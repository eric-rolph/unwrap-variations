"""The post-solver refinement chain: merge charts, repair folds, keep the best.

A backend hands back a layout that is the best its solver could do given the
cuts it was told to make.  On organic meshes that is rarely the best layout
available, for two reasons measured on Suzanne and recorded in
``docs/ORGANIC.md`` section 0:

* **Too many cuts.**  100% of the winning layout's 31.6 units of seam was
  boundary between two islands and none of it was an intra-island slit, so the
  entire seam budget was reachable by merging charts.  Seam length was 44.1% of
  the total penalty.
* **A fold.**  The layout was locally injective everywhere -- all 968 signed UV
  areas positive, zero flipped triangles -- and still overlapped itself, because
  one chart's boundary lapped over a distant part of the same chart.  That cost
  another 41.7% of the penalty, almost all of it the flat ``_HARD_STEP``.

This module runs the two fixes in the order they have to happen:

    backend uvs -> merge charts -> repair folds -> (caller repacks) -> measure

Merging comes first because a merged chart is a *different* chart: certifying
the pre-merge layout says nothing about the post-merge one.

**Rule 2 of the contract is enforced here and only here.**  Every stage is
scored by the frozen :mod:`~unwrap_variations.core.metrics` scorer before and
after, and a stage that does not improve the score is discarded wholesale.  A
stage cannot make a candidate worse, so the chain is safe to leave on by
default and any regression is a bug rather than a trade-off.

Nothing in this module imports :mod:`bpy`.  Repacking needs Blender and is
therefore the caller's job -- see :func:`plan_repack`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..backends import seams as seam_mod
from .meshdata import UVMeshSample
from .metrics import ScoreWeights, UVMetrics, score

__all__ = [
    "RefineOutcome",
    "refine",
    "refine_enabled",
    "plan_repack",
    "chart_seam_mask",
]


#: Stage names, in execution order.  Used for reporting and for the per-stage
#: opt-outs a recipe can carry.
STAGES = ("merge", "repair")

#: Stages whose output changes island *shapes* and must therefore be repacked
#: before it is scored.  Repair re-flattens a chart in place at its original UV
#: area, so it does not disturb the arrangement; merging fuses charts and does.
_NEEDS_REPACK = frozenset({"merge"})


@dataclass
class RefineOutcome:
    """What the chain did to one candidate.

    Attributes:
        uvs: The layout to use.  Always the best-scoring one seen, which may be
            the input unchanged.
        metrics: Metrics of :attr:`uvs`, so the caller need not re-measure.
        score: Score of :attr:`uvs` under the weights the chain was given.
        changed: Whether :attr:`uvs` differs from the input.
        stages_applied: Names of the stages whose output was kept.
        stages_rejected: Names of stages that ran but were discarded because
            they did not improve the score.  This is not a failure -- it is
            Rule 2 working -- but it is worth surfacing, because a stage that is
            always rejected is a stage that is not earning its runtime.
        notes: Human-readable diagnostics, safe to show in a UI.
        seconds: Wall-clock time the whole chain took.
    """

    uvs: np.ndarray
    metrics: UVMetrics | None = None
    score: float = 0.0
    changed: bool = False
    stages_applied: tuple[str, ...] = ()
    stages_rejected: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    seconds: float = 0.0

    def summary(self) -> str:
        """Return a compact one-line description for logs."""
        if not self.stages_applied and not self.stages_rejected:
            return "refine: skipped"
        return "refine: kept=%s rejected=%s score=%.2f" % (
            ",".join(self.stages_applied) or "-",
            ",".join(self.stages_rejected) or "-",
            self.score,
        )


def chart_seam_mask(sample: UVMeshSample, tri_chart: np.ndarray, topo: Any) -> np.ndarray:
    """Return the ``(E,)`` seam mask implied by a chart partition.

    An interior manifold edge is a cut exactly when its two incident triangles
    belong to different charts.  This is the inverse of
    :func:`engine.derive_uv_seams`, which reads cuts back out of a *layout*;
    here the cuts are known and the layout does not exist yet.

    Its purpose is to let a real solver finish the job.  Merging decides *where*
    to cut, which is a combinatorial question our chart code answers well; how
    to flatten the resulting chart is a numerical question Blender's ABF answers
    better than the fallback flattener does.  Handing these seams back to
    ``bpy.ops.uv.unwrap`` gets both.
    """
    n_edges = sample.n_edges
    out = np.zeros(n_edges, dtype=bool)
    if n_edges == 0 or sample.n_tris == 0:
        return out
    counts = np.asarray(topo.face_count, dtype=np.int64)
    interior = counts == 2
    if not interior.any():
        return out
    idx = np.nonzero(interior)[0]
    fa = np.asarray(topo.face_a, dtype=np.int64)[idx]
    fb = np.asarray(topo.face_b, dtype=np.int64)[idx]
    valid = (fa >= 0) & (fa < sample.n_tris) & (fb >= 0) & (fb < sample.n_tris)
    idx, fa, fb = idx[valid], fa[valid], fb[valid]
    if idx.size == 0:
        return out
    labels = np.asarray(tri_chart).reshape(-1)
    out[idx] = labels[fa] != labels[fb]
    return out


def refine_enabled(params: Any) -> bool:
    """Return whether ``params`` asks for any refinement at all.

    A recipe opts out of merging by carrying ``max_stretch < 1.0``, which makes
    every merge unsatisfiable -- that is how the grid keeps an unmerged control
    slot reachable (``docs/ORGANIC.md`` section 5).  Fold repair has no such
    dial because a fold is a defect rather than a preference: there is no
    layout for which leaving the overlap in is the better answer, and Rule 2
    already guarantees the stage cannot cost anything when it fails.
    """
    return True


def refine(
    sample: UVMeshSample,
    uvs: np.ndarray,
    *,
    params: Any = None,
    weights: ScoreWeights | None = None,
    topo: Any = None,
    derive_seams: Callable[..., np.ndarray] | None = None,
    repack: Callable[[np.ndarray], np.ndarray | None] | None = None,
    resolve: Callable[[np.ndarray], np.ndarray | None] | None = None,
    compute_overlap: bool = True,
    stages: Sequence[str] = STAGES,
    time_budget: float = 0.0,
) -> RefineOutcome:
    """Improve ``uvs`` if possible, never worsen it.

    Args:
        sample: The mesh.  Never mutated.
        uvs: ``(L, 2)`` layout the backend produced.
        params: The recipe, read for ``max_stretch``.  May be ``None``.
        weights: Scorer weights.  Defaults to ``balanced``.
        topo: Prebuilt edge topology, shared across candidates by the caller
            because building it per candidate is pure waste.
        derive_seams: ``(sample, topo, uvs) -> seam mask``.  Injected rather
            than imported so this module does not depend on the engine, which
            depends on it.
        repack: Optional ``(uvs) -> uvs | None`` that repacks a layout.
            Injected because packing needs :mod:`bpy` and this module must not.
            **A merged layout has to be repacked before it is judged**, not
            after: merging fuses charts into larger ones that sit where their
            parents used to, so the unpacked result collides with its
            neighbours and scores far worse than what it replaces.  Measured on
            Suzanne before this was wired up, merging cut seam length from
            29.950 to 15.398 and the score still fell from 14.97 to 8.43, so
            every single merge was correctly-but-uselessly discarded.  Without
            a ``repack`` the merge stage is close to dead weight.
        resolve: Optional ``(seam_mask) -> uvs | None`` that re-runs the real
            solver against a given cut set.  Also injected because it needs
            :mod:`bpy`.  Merging answers *where to cut*, which is combinatorial;
            *how to flatten the result* is numerical, and Blender's ABF is
            better at it than the fallback flattener.  Measured on Suzanne:
            merging alone tripled mean angle distortion from 3.75 to 7.94
            degrees, because every merged chart was re-flattened by the
            symmetric-Dirichlet descent rather than by the solver that produced
            the original layout.  Re-solving on the merged cuts keeps the seam
            reduction and hands the flattening back to ABF.
        compute_overlap: Passed through to the evaluator.  Refinement is about
            overlap, so switching this off disables the repair stage's ability
            to tell whether it helped; the chain degrades to merge-only.
        stages: Which stages to consider, in order.
        time_budget: Seconds after which no *new* stage is started.  ``0``
            means no limit.  A stage already running is never interrupted --
            aborting mid-flatten would leave a half-built layout.

    Returns:
        A :class:`RefineOutcome`.  On any internal failure the input layout is
        returned unchanged with the reason in ``notes``: refinement is an
        optimisation, and an optimisation that raises is worse than one that
        declines.
    """
    started = time.perf_counter()
    w = weights if weights is not None else ScoreWeights()
    current = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    notes: list[str] = []
    applied: list[str] = []
    rejected: list[str] = []

    if topo is None:
        try:
            topo = seam_mod.build_edge_topology(sample)
        except Exception as exc:  # noqa: BLE001 - refinement never raises
            return RefineOutcome(
                uvs=np.asarray(uvs), notes=("topology unavailable: %s" % exc,),
                seconds=time.perf_counter() - started,
            )

    measure = _measurer(sample, topo, derive_seams, compute_overlap, w)
    best_uvs = current
    best_score, best_metrics = measure(current)
    if best_metrics is None:
        return RefineOutcome(
            uvs=np.asarray(uvs), notes=("baseline could not be measured",),
            seconds=time.perf_counter() - started,
        )

    for stage in stages:
        if time_budget > 0.0 and (time.perf_counter() - started) >= time_budget:
            notes.append("%s: skipped, time budget spent" % stage)
            continue
        try:
            produced, why = _run_stage(
                stage, sample, best_uvs, params=params, topo=topo, resolve=resolve
            )
        except Exception as exc:  # noqa: BLE001 - a broken stage must not
            # cost the candidate; it just does not get the improvement.
            notes.append("%s: failed, %s: %s" % (stage, type(exc).__name__, exc))
            continue
        if why:
            notes.append("%s: %s" % (stage, why))
        variants = [v for v in (produced if isinstance(produced, list) else [produced])
                    if v is not None]
        if not variants:
            continue

        # A stage may offer several layouts for the same decision -- merging
        # offers its own rebuilt layout and, when a solver is available, the
        # same cuts re-solved by ABF. Which wins is a measurement, not a guess,
        # and repacking has to happen before the measurement rather than after.
        if repack is not None and stage in _NEEDS_REPACK:
            packed_variants = []
            for variant in variants:
                try:
                    packed = repack(variant)
                except Exception as exc:  # noqa: BLE001 - packing is optional
                    packed = None
                    notes.append("%s: repack failed, %s" % (stage, type(exc).__name__))
                if packed is not None:
                    packed_variants.append(packed)
            variants.extend(packed_variants)

        candidate_score, candidate_metrics, produced = float("-inf"), None, None
        for variant in variants:
            v_score, v_metrics = measure(variant)
            if v_metrics is not None and v_score > candidate_score:
                candidate_score, candidate_metrics, produced = v_score, v_metrics, variant
        if len(variants) > 1:
            notes.append("%s: best of %d variants scored %.2f"
                         % (stage, len(variants), candidate_score))
        if candidate_metrics is None:
            notes.append("%s: result could not be measured, discarded" % stage)
            rejected.append(stage)
            continue
        # Rule 2, enforced. The comparison is on the SCORE, not on any single
        # axis: a stage that trades distortion for a removed fold is a good
        # trade -- the flat overlap step alone is worth 0.6 of penalty, roughly
        # 19 points on Suzanne -- and an axis-wise test cannot see that.
        if candidate_score > best_score + _SCORE_EPS:
            notes.append("%s: %.2f -> %.2f" % (stage, best_score, candidate_score))
            best_uvs, best_score, best_metrics = produced, candidate_score, candidate_metrics
            applied.append(stage)
        else:
            notes.append("%s: %.2f -> %.2f, discarded" % (stage, best_score, candidate_score))
            rejected.append(stage)

    changed = best_uvs is not current
    return RefineOutcome(
        uvs=np.asarray(best_uvs, dtype=np.float32) if changed else np.asarray(uvs),
        metrics=best_metrics,
        score=best_score,
        changed=changed,
        stages_applied=tuple(applied),
        stages_rejected=tuple(rejected),
        notes=tuple(notes),
        seconds=time.perf_counter() - started,
    )


#: Score improvement below this is noise, not an improvement.  The scorer is
#: ``100 * exp(-P)``, so this is well under a thousandth of a penalty unit.
_SCORE_EPS = 1e-6


def _measurer(sample, topo, derive_seams, compute_overlap, weights):
    """Return a ``uvs -> (score, metrics)`` closure over the fixed context."""
    from .evaluator import evaluate

    def measure(candidate_uvs):
        try:
            measured = sample.with_uvs(candidate_uvs)
            if derive_seams is not None:
                measured = measured.with_seams(derive_seams(sample, topo, candidate_uvs))
            metrics = evaluate(measured, compute_overlap=compute_overlap)
            return score(metrics, weights), metrics
        except Exception:  # noqa: BLE001 - an unmeasurable layout is simply
            # not a candidate for replacing the incumbent.
            return float("-inf"), None

    return measure


def _run_stage(stage, sample, uvs, *, params, topo, resolve=None):
    """Run one stage and return ``(new_uvs_or_None, note)``."""
    if stage == "merge":
        return _stage_merge(sample, uvs, params=params, topo=topo, resolve=resolve)
    if stage == "repair":
        return _stage_repair(sample, uvs)
    return None, "unknown stage %r" % (stage,)


#: Seconds `merge_charts` may spend screening new candidate pairs on one
#: candidate.  Merging is the expensive stage by a wide margin and its result
#: is always a prefix of the same greedy sequence, so a budget costs quality
#: gracefully rather than producing a different kind of answer.  Measured on
#: Suzanne over ten candidates: unbudgeted merging took 45.6 s of a 65.2 s
#: generate for a best of 48.46; at 2.0 s it takes 16.0 s of 51.6 s for 47.33.
#: Roughly a point of score for a third of the wall-clock.
_MERGE_TIME_BUDGET = 2.0


def _stage_merge(sample, uvs, *, params, topo, resolve=None):
    """Merge adjacent charts under the recipe's distortion bound."""
    from . import charts, injectivity

    max_stretch = _float_attr(params, "max_stretch", 1.25)
    if max_stretch < 1.0:
        # The deliberate unmerged control. Not a failure; do not log it as one.
        return None, ""

    tri_chart = charts.tri_islands(sample, uvs=uvs, topo=topo)
    n_before = int(np.unique(tri_chart).size)
    if n_before < 2:
        return None, "single chart, nothing to merge"

    from . import flatten

    merged, report = charts.merge_charts(
        sample,
        tri_chart,
        max_stretch=max_stretch,
        # Pass the flattener plainly and let `screen_iters` default. Wrapping it
        # to force a cheap budget was actively wrong: the wrapper's
        # `setdefault("iters", ...)` could not override the explicit value
        # merge_charts passes when screening, and *did* fill in the omission
        # that means full quality, so it left screening untouched and quietly
        # degraded the final polish instead -- exactly backwards. The default
        # derives the screening budget as a fraction of the flattener's own,
        # which is also the only version that cannot go stale when that budget
        # changes.
        flatten=flatten.flatten_chart,
        certify=injectivity.certify,
        uvs=uvs,
        topo=topo,
        time_budget=_MERGE_TIME_BUDGET,
    )
    n_after = int(np.unique(merged).size)
    if n_after >= n_before:
        return None, "no merge was admissible at max_stretch=%.2f" % max_stretch

    # merge_charts already rebuilds the layout, pasting each merged chart's
    # flattened UVs into a copy of the input and leaving unmerged charts
    # bit-identical.  Re-deriving it here would duplicate that work and risk
    # disagreeing with what the merge actually accepted.
    layout = getattr(report, "uvs", None)

    # Hand the merged cut set back to the real solver. The chart code decided
    # WHERE to cut; ABF flattens the result better than the fallback descent
    # does, and on Suzanne that difference is worth more than the merge itself.
    # If it declines, the merge's own rebuilt layout still stands.
    note = "%d -> %d charts at max_stretch=%.2f, seam %.3f -> %.3f" % (
        n_before, n_after, max_stretch,
        float(getattr(report, "seam_length_before", float("nan"))),
        float(getattr(report, "seam_length_after", float("nan"))),
    )
    variants = [] if layout is None else [layout]
    if resolve is not None:
        try:
            mask = chart_seam_mask(sample, merged, topo)
            resolved = resolve(mask)
        except Exception as exc:  # noqa: BLE001 - re-solving is a bonus
            resolved, note = None, note + " (re-solve failed: %s)" % type(exc).__name__
        if resolved is not None:
            variants.append(np.asarray(resolved, dtype=np.float64).reshape(-1, 2))
            note += ", re-solved"

    if not variants:
        return None, "merge accepted %d..%d charts but produced no layout" % (n_after, n_before)
    return variants, note


def _stage_repair(sample, uvs):
    """Re-flatten any chart that fails injectivity certification."""
    from . import flatten

    fixed, report = flatten.repair_folds(sample, uvs)
    if not getattr(report, "changed", False):
        return None, getattr(report, "summary", lambda: "no change")()
    return fixed, "repaired %d chart(s)" % int(getattr(report, "charts_repaired", 0) or 0)


def _float_attr(obj, name, default):
    """Read a float attribute from a recipe, tolerating anything at all."""
    try:
        value = float(getattr(obj, name, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def plan_repack(outcome: RefineOutcome) -> bool:
    """Return whether the caller should repack after refinement.

    Repacking needs ``bpy.ops.uv.pack_islands`` and therefore cannot live in
    ``core/``.  Merging changes island shapes, so a merged layout that is not
    repacked leaves part of its gain in the packing term; a layout the chain did
    not change is already packed as the backend left it.
    """
    return bool(outcome.changed and "merge" in outcome.stages_applied)
