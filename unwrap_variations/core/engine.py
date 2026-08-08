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


def dead_layout_reason(uvs: np.ndarray, previous: np.ndarray | None = None) -> str:
    """Return why ``uvs`` is not a usable layout, or ``""`` when it is fine.

    This exists because of a verified and genuinely nasty Blender behaviour:
    **``bpy.ops.uv.unwrap`` returns ``{'FINISHED'}`` even when it fails.**  On a
    closed mesh with no seams it prints "Unwrap failed to solve 1 of 1
    island(s)" to the console, leaves the previous UVs completely untouched, and
    still reports success.  Trusting the operator's return status therefore
    scores a candidate that did nothing at all, and -- worse -- it inherits
    whatever UVs the previous candidate left behind, so it can silently
    duplicate a neighbour's result.

    Six of ten slots on an icosphere were dead this way before this check
    existed, every one of them reported as a valid candidate.

    The check is deliberately at the engine level rather than inside a single
    backend: it protects every backend, including third-party ones, and costs
    a few microseconds against a solver that costs milliseconds.

    Args:
        uvs: ``(L, 2)`` candidate UVs.
        previous: Optional UVs the mesh carried before this attempt.  When the
            result is bit-identical to them, the solver almost certainly did
            nothing.

    Returns:
        A human-readable reason, or the empty string when the layout is usable.
    """
    arr = np.asarray(uvs, dtype=np.float64)
    if arr.size == 0:
        return "backend produced no UVs"
    if not np.isfinite(arr).all():
        return "layout contains NaN or infinite coordinates"

    span = arr.max(axis=0) - arr.min(axis=0)
    if float(span.max()) <= 1e-9:
        # Every UV collapsed onto a single point. This is what an unsolved
        # unwrap leaves behind on a freshly created (zero-filled) UV layer.
        return "unwrap produced a degenerate layout (all UVs coincident)"

    if previous is not None:
        prev = np.asarray(previous, dtype=np.float64)
        if prev.shape == arr.shape and np.array_equal(prev, arr):
            return "unwrap left the UVs unchanged (solver reported success but did nothing)"
    return ""


#: Solvers that need the surface cut open into disks before they can run.
_SEAM_DRIVEN = ("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")


def repair_recipe(
    params: UnwrapParams,
    sample: UVMeshSample,
    topo: "seam_mod.EdgeTopology",
) -> tuple[UnwrapParams, str]:
    """Rewrite a recipe that provably cannot work on *this* mesh.

    A closed surface has no boundary, so a seam-driven solver given an empty
    seam set has no disk to flatten and simply gives up -- and, because Blender
    reports success anyway (see :func:`dead_layout_reason`), the slot is wasted
    silently.  On an icosphere that was six of ten slots.

    Rather than let the artist pay for dead slots, escalate the seam strategy to
    ``DUAL_GRAPH``, which is guaranteed to produce a cut on a closed component
    because it cuts the complement of a spanning forest of the face dual graph.

    This is deliberately conservative: it only fires when the strategy yields
    *no* seams at all on a mesh with *no* boundary, which is exactly the case
    that cannot succeed.  Anything that might work is left untouched.

    Args:
        params: The recipe to check.
        sample: The mesh it will run against.
        topo: Prebuilt edge topology for ``sample``.

    Returns:
        ``(params, note)``. ``note`` is empty when nothing was changed.
    """
    if params.method not in _SEAM_DRIVEN:
        return params, ""
    if sample.n_edges == 0:
        return params, ""
    try:
        if bool(topo.boundary_mask().any()):
            # An open surface can always be flattened without extra cuts.
            return params, ""
        mask = seam_mod.compute_seams(
            sample,
            params.seam_strategy,
            angle_deg=params.seam_angle_deg,
            curvature_threshold=params.curvature_threshold,
            existing=sample.edge_seam,
            seed=params.seed,
        )
        if bool(np.asarray(mask).any()):
            return params, ""
    except Exception:  # noqa: BLE001 - a probe failure must not kill the run
        return params, ""

    # Derive a distinct seed from the original recipe so that several recipes
    # escalating at once produce *different* dual-graph cuts instead of
    # collapsing onto one identical signature and being de-duplicated away.
    # Different seeds give genuinely different (and equally valid) cut sets.
    spread = int(params.signature()[:8], 16) & 0x3FFFFFFF
    return (
        params.replace(seam_strategy="DUAL_GRAPH", seed=spread),
        "%s yields no seams on a closed mesh; escalated to DUAL_GRAPH"
        % params.seam_strategy,
    )


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
        # Tracks the last layout a backend produced, so a solver that
        # silently does nothing and inherits it can be detected.
        previous_uvs: np.ndarray | None = None
        seen_signatures: set[str] = set()

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

            # Rewrite recipes that cannot possibly work on this mesh, and
            # de-duplicate: an escalation can collide with an existing recipe.
            params, note = repair_recipe(params, sample, topo)
            if note:
                report.warnings.append("%s: %s" % (params.describe(), note))
            if params.signature() in seen_signatures:
                report.failures.append(
                    (params, "duplicate of another candidate after repair")
                )
                continue
            seen_signatures.add(params.signature())

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

            # Blender can report success while doing nothing at all; see
            # dead_layout_reason. Compare against the previous candidate's UVs
            # too, because an unsolved unwrap silently inherits them.
            dead = dead_layout_reason(uvs, previous_uvs)
            if dead:
                report.failures.append((params, dead))
                previous_uvs = uvs
                continue
            previous_uvs = uvs

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
        cand.pareto = id(cand) in front  # type: ignore[attr-defined]


#: How many refill rounds `generate` will run before giving up on reaching the
#: requested candidate count. Each round costs real solver time, so this is a
#: guard against pathological meshes where almost every recipe fails.
_MAX_REFILL_ROUNDS = 4


def _merge(into: GenerationReport, part: GenerationReport, cap: int) -> None:
    """Fold one batch's results into the accumulating report, up to `cap`."""
    seen = {c.signature for c in into.candidates}
    for cand in part.candidates:
        if len(into.candidates) >= cap:
            break
        if cand.signature in seen:
            continue
        seen.add(cand.signature)
        into.candidates.append(cand)
    into.failures.extend(part.failures)
    for note in part.warnings:
        if note not in into.warnings:
            into.warnings.append(note)
    into.seconds += part.seconds


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
    target = max(int(count), 0)
    if target == 0:
        return GenerationReport()

    # Deliver `target` USABLE candidates, not `target` attempts. Recipes that a
    # given mesh cannot solve are rejected rather than shown as dead slots, so
    # without a refill pass a closed mesh would hand back a half-empty grid.
    report = GenerationReport()
    tried: set[str] = set()
    round_seed = int(seed)
    for attempt in range(_MAX_REFILL_ROUNDS):
        want = target - len(report.candidates)
        if want <= 0:
            break
        # Over-request on refills: some fraction will fail again.
        batch = sample_grid(
            want if attempt == 0 else want * 2, seed=round_seed, base=base
        )
        batch = [p for p in batch if p.signature() not in tried]
        tried.update(p.signature() for p in batch)
        if not batch:
            round_seed += 977
            continue

        part = _run_recipes(
            obj,
            batch,
            weights=weights,
            backends=backends,
            progress=progress if attempt == 0 else None,
            should_cancel=should_cancel,
            compute_overlap=compute_overlap,
            parent_signature=None,
            generation=0,
        )
        _merge(report, part, target)
        if part.cancelled:
            report.cancelled = True
            break
        round_seed += 977

    _rank_in_place(report.candidates, weights)
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
