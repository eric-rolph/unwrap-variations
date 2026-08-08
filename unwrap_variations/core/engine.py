"""Candidate generation and branching: the coordinator that ties it together.

The engine is the only place that knows the full pipeline:

    extract the mesh once
      -> for each parameter recipe
           -> run a backend to produce UVs
           -> derive the seams that layout actually implies
           -> measure distortion
           -> score
      -> rank, index, hand back

Two invariants drive the design:

1. **Extract once, unwrap many.** Pulling a mesh out of Blender is the single
   most expensive non-solver step, and the geometry does not change between
   candidates -- only the UVs do.  Extracting per candidate would make a
   10-slot run roughly ten times slower for no benefit.

2. **The mesh is put back exactly as it was found.** Backends drive
   ``bpy.ops.uv.*``, which mutates seams, selection and mode.  The engine takes
   a snapshot up front and restores it in a ``finally``, so an exception, a
   cancellation or a backend that misbehaves all leave the artist's mesh
   untouched.  Generation is a measurement, not an edit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..backends import seams as seam_mod
from ..backends.base import UnwrapRequest, get_backend
from .candidate import Candidate, CandidateStore
from .evaluator import evaluate
from .meshdata import UVMeshSample
from .metrics import ScoreWeights, pareto_front, score
from .params import UnwrapParams, mutate, sample_grid

__all__ = [
    "GenerationReport",
    "generate",
    "branch",
    "derive_uv_seams",
    "validate_object",
]

#: Progress callback signature: ``(fraction_0_to_1, human_message)``.
ProgressFn = Callable[[float, str], None]
#: Cancellation probe: return ``True`` to stop after the current candidate.
CancelFn = Callable[[], bool]


@dataclass
class GenerationReport:
    """Everything one generation run produced, successes and failures alike.

    Attributes:
        candidates: Successful candidates, already ranked and indexed.
        seconds: Wall-clock duration of the whole run.
        failures: ``(params, reason)`` for every recipe that did not produce a
            usable layout.  Surfaced in the UI rather than swallowed -- a run
            where 7 of 10 recipes failed is something the artist must see.
        warnings: Non-fatal notes (stale store discarded, packing unavailable,
            a backend skipped because it is not installed, and so on).
        cancelled: ``True`` when the user interrupted the run.
    """

    candidates: list[Candidate] = field(default_factory=list)
    seconds: float = 0.0
    failures: list[tuple[UnwrapParams, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        """Whether the run produced at least one usable candidate."""
        return bool(self.candidates)

    def summary(self) -> str:
        """Return a one-line ASCII summary suitable for the status bar."""
        parts = ["%d candidate(s) in %.2fs" % (len(self.candidates), self.seconds)]
        if self.failures:
            parts.append("%d failed" % len(self.failures))
        if self.cancelled:
            parts.append("cancelled")
        return ", ".join(parts)


def validate_object(obj: Any) -> tuple[bool, str]:
    """Check that ``obj`` is something we can actually unwrap.

    Returns ``(False, reason)`` rather than raising so callers can put the
    reason straight into ``self.report({'ERROR'}, ...)``.
    """
    if obj is None:
        return False, "No active object."
    if getattr(obj, "type", None) != "MESH":
        return False, "Active object %r is not a mesh." % getattr(obj, "name", "?")
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return False, "Object has no mesh data."
    try:
        n_polys = len(mesh.polygons)
        n_loops = len(mesh.loops)
    except (AttributeError, TypeError):
        return False, "Mesh data is not readable."
    if n_polys == 0 or n_loops == 0:
        return False, "Mesh has no faces to unwrap."
    return True, ""


def derive_uv_seams(
    sample: UVMeshSample,
    topo: "seam_mod.EdgeTopology",
    uvs: np.ndarray,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    """Return the ``(E,)`` seam mask that a given UV layout actually implies.

    Rather than reporting the seams we *asked* for, this reads back the seams
    the result genuinely has: an interior edge is a seam exactly when its two
    incident faces do not agree on the UV coordinates of its two endpoints.

    This matters because it is the only way to get a meaningful seam length for
    methods that ignore seams entirely.  ``smart_project`` and ``cube_project``
    invent their own charts, so asking "what seams did we mark?" would report
    zero for layouts that are in fact heavily cut.  Measuring the layout makes
    every method comparable on the same footing, which is the whole point of
    ranking them against each other.

    Args:
        sample: The mesh being measured.
        topo: Edge topology for ``sample`` (built once, reused per candidate).
        uvs: ``(L, 2)`` candidate UVs.
        eps: Coordinate tolerance below which two UVs count as identical.

    Returns:
        ``(E,)`` bool array. Boundary and non-manifold edges are never seams --
        there is no second face to disagree with.
    """
    n_edges = sample.n_edges
    out = np.zeros(n_edges, dtype=bool)
    if n_edges == 0 or sample.n_tris == 0:
        return out

    # Only manifold interior edges (exactly two incident faces) can be seams.
    interior = np.asarray(topo.face_count, dtype=np.int64) == 2
    if not interior.any():
        return out

    idx = np.nonzero(interior)[0]
    fa = np.asarray(topo.face_a, dtype=np.int64)[idx]
    fb = np.asarray(topo.face_b, dtype=np.int64)[idx]

    n_tris = sample.n_tris
    valid = (fa >= 0) & (fa < n_tris) & (fb >= 0) & (fb < n_tris)
    if not valid.all():
        idx, fa, fb = idx[valid], fa[valid], fb[valid]
    if idx.size == 0:
        return out

    ev = np.asarray(sample.edge_verts, dtype=np.int64)[idx]
    tv_a, tl_a = sample.tri_verts[fa], sample.tri_loops[fa]
    tv_b, tl_b = sample.tri_verts[fb], sample.tri_loops[fb]
    rows = np.arange(idx.size)

    def corner_loops(tv: np.ndarray, tl: np.ndarray, vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Loop index of ``vert`` within each face, plus a found mask."""
        eq = tv == vert[:, None]
        found = eq.any(axis=1)
        col = eq.argmax(axis=1)
        return tl[rows, col], found

    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] < sample.n_loops:
        return out

    seam = np.zeros(idx.size, dtype=bool)
    for corner in (0, 1):
        vert = ev[:, corner]
        la, ok_a = corner_loops(tv_a, tl_a, vert)
        lb, ok_b = corner_loops(tv_b, tl_b, vert)
        # A vertex missing from one of the two faces means the triangulation
        # diagonal was matched instead of a real edge; treat it as continuous
        # rather than inventing a seam.
        both = ok_a & ok_b
        delta = np.abs(uv[la] - uv[lb]).max(axis=1)
        seam |= both & (delta > eps)

    out[idx] = seam
    return out


def _resolve_backend_id(params: UnwrapParams, allowed: Sequence[str] | None) -> str:
    """Pick the backend for ``params``, honouring an optional allow-list."""
    wanted = params.backend or "blender_native"
    if allowed and wanted not in allowed:
        return allowed[0]
    return wanted


def _run_recipes(
    obj: Any,
    recipes: Sequence[UnwrapParams],
    *,
    weights: ScoreWeights,
    backends: Sequence[str] | None,
    progress: ProgressFn | None,
    should_cancel: CancelFn | None,
    compute_overlap: bool,
    parent_signature: str | None,
    generation: int,
) -> GenerationReport:
    """Execute a list of recipes against ``obj`` and return a ranked report.

    This is the shared body of :func:`generate` and :func:`branch`; the only
    thing that differs between them is how the recipe list was produced.
    """
    from . import bridge  # imported lazily: only this path needs bpy

    started = time.perf_counter()
    report = GenerationReport()

    ok, why = validate_object(obj)
    if not ok:
        report.warnings.append(why)
        report.seconds = time.perf_counter() - started
        return report

    snapshot = None
    try:
        snapshot = bridge.snapshot_state(obj)
    except Exception as exc:  # noqa: BLE001 - never take Blender down
        report.warnings.append("Could not snapshot mesh state: %s" % exc)

    try:
        sample = bridge.extract_sample(obj)
        problems = sample.validate()
        # "no triangles" is fatal; everything else is worth saying out loud but
        # is survivable -- degenerate faces are excluded from the metrics.
        for problem in problems:
            report.warnings.append("Geometry: %s" % problem)
        if sample.n_tris == 0:
            report.seconds = time.perf_counter() - started
            return report

        topo = seam_mod.build_edge_topology(sample)
        total = max(len(recipes), 1)

        for i, params in enumerate(recipes):
            if should_cancel is not None:
                try:
                    if should_cancel():
                        report.cancelled = True
                        break
                except Exception:  # noqa: BLE001 - a bad probe must not abort
                    pass

            if progress is not None:
                try:
                    progress(i / total, "Unwrapping %d/%d: %s"
                             % (i + 1, total, params.describe()))
                except Exception:  # noqa: BLE001 - UI callback is advisory
                    pass

            backend_id = _resolve_backend_id(params, backends)
            try:
                backend = get_backend(backend_id)
            except Exception as exc:  # noqa: BLE001
                report.failures.append((params, "backend %r unavailable: %s"
                                        % (backend_id, exc)))
                continue

            available, why_not = backend.is_available()
            if not available:
                report.failures.append((params, "%s: %s" % (backend_id, why_not)))
                continue

            result = backend.safe_unwrap(
                UnwrapRequest(obj_name=obj.name, params=params, sample=sample)
            )
            if not result.ok or result.uvs is None:
                report.failures.append((params, result.message or "backend failed"))
                continue

            uvs = np.asarray(result.uvs, dtype=np.float32).reshape(-1, 2)
            if uvs.shape[0] != sample.n_loops:
                report.failures.append(
                    (params, "backend returned %d UVs, expected %d"
                     % (uvs.shape[0], sample.n_loops))
                )
                continue

            try:
                seam_mask = derive_uv_seams(sample, topo, uvs)
                measured = sample.with_uvs(uvs).with_seams(seam_mask)
                metrics = evaluate(measured, compute_overlap=compute_overlap)
                value = score(metrics, weights)
            except Exception as exc:  # noqa: BLE001 - a metric bug must not
                # discard an otherwise valid unwrap; keep it, unranked.
                report.warnings.append("Scoring failed for %s: %s"
                                       % (params.describe(), exc))
                metrics, value = None, 0.0

            report.candidates.append(
                Candidate(
                    index=len(report.candidates),
                    params=params,
                    uvs=uvs,
                    metrics=metrics,
                    score=value,
                    ok=True,
                    message=result.message,
                    parent=parent_signature,
                    generation=generation,
                    seconds=result.seconds,
                )
            )
    except Exception as exc:  # noqa: BLE001 - containment is the whole point
        report.warnings.append("Generation aborted: %s: %s" % (type(exc).__name__, exc))
    finally:
        # Always, always put the mesh back.
        if snapshot is not None:
            try:
                bridge.restore_state(obj, snapshot)
            except Exception as exc:  # noqa: BLE001
                report.warnings.append("Could not restore mesh state: %s" % exc)
        if progress is not None:
            try:
                progress(1.0, "Done")
            except Exception:  # noqa: BLE001
                pass

    _rank_in_place(report.candidates, weights)
    report.seconds = time.perf_counter() - started
    return report


def _rank_in_place(candidates: list[Candidate], weights: ScoreWeights) -> None:
    """Sort candidates best-first, renumber them, and tag the Pareto front.

    Slot numbers follow the ranking so slot 0 is always the best pick.  A
    stable secondary key on the parameter signature keeps the order
    reproducible when two candidates score identically.
    """
    if not candidates:
        return
    candidates.sort(key=lambda c: (-float(c.score or 0.0), c.signature))

    measured = [c for c in candidates if c.metrics is not None]
    front: set[int] = set()
    if measured:
        try:
            picks = pareto_front([c.metrics for c in measured])
            front = {id(measured[i]) for i in picks if 0 <= i < len(measured)}
        except Exception:  # noqa: BLE001 - ranking is advisory, never fatal
            front = set()

    for i, cand in enumerate(candidates):
        cand.index = i
        # Recorded on the candidate so the UI can badge it without recomputing.
        setattr(cand, "pareto", id(cand) in front)


def generate(
    obj: Any,
    *,
    count: int = 10,
    seed: int = 0,
    weights: ScoreWeights | None = None,
    base: UnwrapParams | None = None,
    backends: Sequence[str] | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    compute_overlap: bool = True,
    store: bool = True,
) -> GenerationReport:
    """Produce ``count`` distinct unwrap candidates for ``obj``.

    Args:
        obj: The Blender object to unwrap.  Must be a mesh with faces.
        count: How many candidates to generate.
        seed: Determinism seed -- the same seed reproduces the same grid.
        weights: Scoring weights; defaults to the ``balanced`` preset.
        base: Optional starting recipe whose unsearched fields are inherited.
        backends: Optional allow-list of backend ids.
        progress: Optional ``(fraction, message)`` progress callback.
        should_cancel: Optional probe; return ``True`` to stop early.
        compute_overlap: Whether to run the (more expensive) overlap test.
        store: Whether to write the results into the object's candidate store.

    Returns:
        A :class:`GenerationReport`.  Never raises -- inspect ``warnings`` and
        ``failures`` to find out what went wrong.
    """
    weights = weights or ScoreWeights.preset("balanced")
    recipes = sample_grid(max(int(count), 0), seed=int(seed), base=base)
    report = _run_recipes(
        obj,
        recipes,
        weights=weights,
        backends=backends,
        progress=progress,
        should_cancel=should_cancel,
        compute_overlap=compute_overlap,
        parent_signature=None,
        generation=0,
    )
    if store and report.candidates:
        _commit(obj, report)
    return report


def branch(
    obj: Any,
    parent: Candidate,
    *,
    count: int = 10,
    strength: float = 0.35,
    seed: int = 0,
    weights: ScoreWeights | None = None,
    backends: Sequence[str] | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    compute_overlap: bool = True,
    store: bool = True,
    keep_parent: bool = True,
) -> GenerationReport:
    """Produce ``count`` mutations clustered around ``parent``.

    This is the "10 more like that one" half of the workflow.  The parent is
    carried into the new set by default so the artist can always tell whether a
    branch actually improved on what they started from -- losing the reference
    point is the fastest way to make an iterative tool feel random.

    Args:
        obj: The Blender object to unwrap.
        parent: The candidate to vary around.
        count: How many children to generate.
        strength: Mutation magnitude in ``[0, 1]``.
        seed: Determinism seed.
        weights: Scoring weights; defaults to the ``balanced`` preset.
        backends: Optional allow-list of backend ids.
        progress: Optional progress callback.
        should_cancel: Optional cancellation probe.
        compute_overlap: Whether to run the overlap test.
        store: Whether to write results into the candidate store.
        keep_parent: Whether to re-include the parent in the new set.

    Returns:
        A :class:`GenerationReport`.  Never raises.
    """
    weights = weights or ScoreWeights.preset("balanced")
    if parent is None:
        report = GenerationReport()
        report.warnings.append("No parent candidate selected to branch from.")
        return report

    n_children = max(int(count), 0)
    if keep_parent and n_children > 0:
        n_children -= 1
    recipes = mutate(
        parent.params, count=n_children, strength=float(strength), seed=int(seed)
    )
    if keep_parent:
        recipes = [parent.params, *recipes]

    report = _run_recipes(
        obj,
        recipes,
        weights=weights,
        backends=backends,
        progress=progress,
        should_cancel=should_cancel,
        compute_overlap=compute_overlap,
        parent_signature=parent.signature,
        generation=int(getattr(parent, "generation", 0)) + 1,
    )
    if store and report.candidates:
        _commit(obj, report)
    return report


def _commit(obj: Any, report: GenerationReport) -> None:
    """Write a report's candidates into the object's store, tolerantly."""
    try:
        store = CandidateStore.for_object(obj)
        store.set(report.candidates)
    except Exception as exc:  # noqa: BLE001 - results are still returned
        report.warnings.append("Could not store candidates: %s" % exc)
