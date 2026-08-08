"""UV quality metrics, the scalar score derived from them and Pareto ranking.

This module owns the *vocabulary* of UV quality: :class:`UVMetrics` is the
record every candidate carries, :class:`ScoreWeights` is the artist-facing dial
that says which kind of quality matters, and :func:`score` collapses the two
into the single number the UI sorts by.  Computing the metrics is the
evaluator's job (:mod:`unwrap_variations.core.evaluator`); this module only
defines and interprets them.

Nothing here imports :mod:`bpy`, so the whole scoring model is unit-testable in
plain CPython.

Design notes
------------

*Why an exponential score?*  A weighted sum of penalties is easy to reason
about but has no natural upper bound, so it cannot be presented as "0..100".
Mapping the total penalty ``P >= 0`` through ``100 * exp(-P)`` gives a bounded,
strictly decreasing function.  Because every individual penalty term is a
non-decreasing function of one metric's *badness* and every weight is clamped
to be non-negative, the total penalty is non-decreasing in badness and the
score is therefore **provably monotonic**: making any metric better can never
lower the score.  A weighted-sum-then-clamp scheme loses that property the
moment the clamp bites.

*Why squared packing penalty?*  Real packers land in the 0.6-0.8 efficiency
band; treating 0.7 as "30% bad" would swamp genuine distortion problems.
``(1 - efficiency)**2`` keeps the ordering (it is still monotone in
``packing_efficiency``) while giving realistic layouts a fair shake.

*Monotonicity contract.*  :func:`score` is non-increasing in
``angle_distortion``, ``area_distortion``, ``stretch_l2``, ``seam_ratio``,
``overlap_area``, ``flipped_tris`` and ``degenerate_tris``, and non-decreasing
in ``packing_efficiency`` and ``uv_coverage``, with the remaining fields held
fixed.  ``n_tris`` is bookkeeping, not a quality axis: it is the denominator of
the flipped/degenerate fractions and is expected to be equal when two
candidates for the same mesh are compared.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "UVMetrics",
    "ScoreWeights",
    "SCORE_PRESETS",
    "score",
    "pareto_front",
]


# ---------------------------------------------------------------------------
# The metric record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UVMetrics:
    """Everything measurable about one UV layout.

    All fields default to a neutral "empty mesh" reading so that a partially
    failed evaluation can still return a well-formed record instead of raising.

    Attributes:
        angle_distortion: Area-weighted mean absolute corner-angle deviation
            between 3D and UV space, in **degrees**.  0 for a conformal map.
        angle_distortion_max: Worst single corner-angle deviation, in degrees.
        area_distortion: Mean ``|log2(uv_area_share / surface_area_share)|``
            over non-degenerate triangles.  0 when relative areas are preserved;
            1 means the average triangle is off by a factor of two.
        stretch_l2: Sander et al. (SIGGRAPH 2001) L2 stretch norm, normalised so
            that a perfectly isometric unwrap scores exactly ``1.0``.  Always
            ``>= 1.0`` for a valid layout.
        stretch_linf: The same normalisation applied to the worst-case (largest
            singular value) stretch of any triangle.
        conformal_error: Area-weighted mean of ``sigma_max / sigma_min - 1``.
            0 for an angle-preserving map, independent of scale.
        island_count: Number of connected UV islands.
        seam_length: Total length of seam-marked edges, in world units.
        seam_ratio: ``seam_length / sqrt(total 3D area)`` -- scale free, so a
            model imported in centimetres compares against one in metres.
        packing_efficiency: Used UV area divided by the area of the *union* of
            the per-island bounding boxes, in ``[0, 1]``.
        uv_coverage: Total absolute UV area, measured against the unit square.
        overlap_area: Total pairwise overlapping UV area.  ``0.0`` when clean.
        flipped_tris: Triangles whose UV winding is mirrored relative to 3D.
        degenerate_tris: Triangles excluded from the distortion means because
            their 3D or UV area collapsed (or their coordinates were invalid).
        texel_density_cv: Area-weighted coefficient of variation of
            ``sqrt(uv_area / surface_area)``.  0 means perfectly uniform texel
            density across the model.
        uv_bounds: ``(min_u, min_v, max_u, max_v)``.
        out_of_bounds_tris: Triangles with at least one corner outside the unit
            square.  Expected to be non-zero for deliberate UDIM layouts.
        n_tris: Triangle count of the evaluated sample.
        eval_seconds: Wall-clock time the evaluation took.
    """

    # -- distortion ----------------------------------------------------------
    angle_distortion: float = 0.0
    angle_distortion_max: float = 0.0
    area_distortion: float = 0.0
    stretch_l2: float = 1.0
    stretch_linf: float = 1.0
    conformal_error: float = 0.0
    # -- topology / layout ---------------------------------------------------
    island_count: int = 0
    seam_length: float = 0.0
    seam_ratio: float = 0.0
    packing_efficiency: float = 0.0
    uv_coverage: float = 0.0
    overlap_area: float = 0.0
    flipped_tris: int = 0
    degenerate_tris: int = 0
    texel_density_cv: float = 0.0
    uv_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    out_of_bounds_tris: int = 0
    # -- bookkeeping ---------------------------------------------------------
    n_tris: int = 0
    eval_seconds: float = 0.0

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of every field.

        ``uv_bounds`` becomes a list because JSON has no tuple type; the
        matching :meth:`from_dict` turns it back into a tuple.
        """
        data = asdict(self)
        data["uv_bounds"] = [float(v) for v in self.uv_bounds]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "UVMetrics":
        """Rebuild metrics from a dict, tolerating unknown and missing keys.

        A ``.blend`` written by a newer add-on version must still open in an
        older one, so unrecognised keys are dropped rather than raising, and
        anything unparseable falls back to the field default.
        """
        if not data:
            return cls()
        defaults = cls()
        clean: dict[str, Any] = {}
        for spec in fields(cls):
            if spec.name not in data:
                continue
            raw = data[spec.name]
            fallback = getattr(defaults, spec.name)
            try:
                if spec.name == "uv_bounds":
                    seq = tuple(float(v) for v in raw)
                    clean[spec.name] = seq if len(seq) == 4 else fallback
                elif spec.name in _INT_FIELDS:
                    clean[spec.name] = int(raw)
                else:
                    clean[spec.name] = float(raw)
            except (TypeError, ValueError):
                clean[spec.name] = fallback
        return cls(**clean)

    # -- convenience ---------------------------------------------------------

    @classmethod
    def empty(cls, n_tris: int = 0, degenerate_tris: int = 0) -> "UVMetrics":
        """Return a neutral record used when evaluation cannot proceed.

        Kept explicit rather than relying on the bare constructor so that the
        evaluator's failure path reads as a deliberate choice.
        """
        return cls(n_tris=int(max(0, n_tris)), degenerate_tris=int(max(0, degenerate_tris)))

    def summary(self) -> str:
        """Return a compact one-line description for logs and reports."""
        return (
            "T=%d isl=%d stretch=%.3f/%.3f ang=%.2fdeg area=%.3f "
            "pack=%.3f overlap=%.4g flip=%d degen=%d"
            % (
                self.n_tris,
                self.island_count,
                self.stretch_l2,
                self.stretch_linf,
                self.angle_distortion,
                self.area_distortion,
                self.packing_efficiency,
                self.overlap_area,
                self.flipped_tris,
                self.degenerate_tris,
            )
        )


#: Fields of :class:`UVMetrics` that must round-trip as ``int``.
_INT_FIELDS = frozenset(
    {"island_count", "flipped_tris", "degenerate_tris", "out_of_bounds_tris", "n_tris"}
)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreWeights:
    """Relative importance of each penalty term in :func:`score`.

    Weights are multiplied into penalties, so ``0.0`` disables a term entirely
    and larger values make it matter more.  Negative weights are clamped to
    zero inside :func:`score` -- a negative weight would invert a term and
    destroy the monotonicity guarantee, so it is treated as "off" rather than
    silently producing a nonsense ranking.

    Attributes:
        angle: Weight on angular (conformal) distortion.
        area: Weight on relative area distortion.
        stretch: Weight on the Sander L2 stretch above the isometric ideal.
        seam: Weight on seam length relative to model size.
        packing: Weight on wasted UV space.
        overlap: Weight on overlapping UVs.  Doubles as the hard-failure dial.
        flipped: Weight on mirrored and degenerate triangles.
    """

    angle: float = 1.0
    area: float = 1.0
    stretch: float = 1.2
    seam: float = 0.6
    packing: float = 0.8
    overlap: float = 2.0
    flipped: float = 2.0

    @classmethod
    def preset(cls, name: str) -> "ScoreWeights":
        """Return a named weight preset, falling back to ``balanced``.

        Args:
            name: One of ``balanced``, ``low_distortion``, ``few_seams``,
                ``tight_packing`` or ``game_ready``.  Case-insensitive.

        Returns:
            The matching :class:`ScoreWeights`.  An unknown name never raises;
            it returns ``balanced``, because a bad preference string stored in a
            ``.blend`` must not be able to break the UI.
        """
        key = str(name or "").strip().lower()
        return SCORE_PRESETS.get(key, SCORE_PRESETS["balanced"])

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe dict of every weight."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ScoreWeights":
        """Rebuild weights from a dict, ignoring unknown keys."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        clean: dict[str, float] = {}
        for key, value in dict(data).items():
            if key in known:
                try:
                    clean[key] = float(value)
                except (TypeError, ValueError):
                    pass
        return cls(**clean)


#: Named presets exposed in the UI.  The names are contract (see ARCHITECTURE
#: section 4) so they can be stored in scene properties.
SCORE_PRESETS: dict[str, ScoreWeights] = {
    "balanced": ScoreWeights(),
    # Character / hero assets: distortion is what shows up in a normal map.
    "low_distortion": ScoreWeights(
        angle=1.8, area=1.4, stretch=2.4, seam=0.25, packing=0.35, overlap=2.0, flipped=2.0
    ),
    # Hard-surface work where every visible cut costs an artist a fix-up.
    "few_seams": ScoreWeights(
        angle=0.8, area=0.7, stretch=0.9, seam=2.4, packing=0.5, overlap=2.0, flipped=2.0
    ),
    # Atlasing / lightmaps: texture memory is the scarce resource.
    "tight_packing": ScoreWeights(
        angle=0.6, area=0.8, stretch=0.8, seam=0.3, packing=2.6, overlap=2.5, flipped=2.0
    ),
    # Shipping game asset: overlaps and flips are outright bugs, and texel
    # density uniformity matters as much as raw distortion.
    "game_ready": ScoreWeights(
        angle=1.0, area=1.3, stretch=1.4, seam=0.7, packing=1.4, overlap=4.0, flipped=4.0
    ),
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

#: Degrees of mean angular deviation that cost one penalty unit.
_ANGLE_SCALE = 45.0
#: ``|log2|`` area error that costs one penalty unit.
_AREA_SCALE = 3.0
#: Sander stretch *above* 1.0 that costs one penalty unit.
_STRETCH_SCALE = 2.0
#: Seam ratio that costs one penalty unit.
_SEAM_SCALE = 25.0
#: Flat penalty applied the instant a hard defect appears at all.  This is what
#: makes overlap and flipping "hard" rather than merely expensive: a layout with
#: a single overlapping texel drops a full multiplicative notch below a clean
#: one, no matter how good the rest of it looks.
_HARD_STEP = 0.6
#: Additional penalty proportional to how much of the layout is defective.
_HARD_SLOPE = 4.0
#: Penalty slope for the share of triangles that could not be measured at all.
_DEGENERATE_SLOPE = 2.0
#: Penalty above which the exponential is saturated; keeps ``exp`` in range.
_MAX_PENALTY = 60.0


def _badness(value: Any, fallback: float) -> float:
    """Return ``value`` as a non-negative float, substituting for non-finite.

    A NaN metric means "we could not measure this", which for scoring purposes
    must be treated as bad rather than as zero -- otherwise a broken evaluation
    would outrank a working one.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return number if number > 0.0 else 0.0


def _unit(value: Any, fallback: float) -> float:
    """Return ``value`` clamped into ``[0, 1]``, substituting for non-finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return min(1.0, max(0.0, number))


def score(m: UVMetrics, w: ScoreWeights | None = None) -> float:
    """Collapse metrics into a single ``0..100`` quality number, higher better.

    The score is ``100 * exp(-P)`` where ``P`` is a weighted sum of per-metric
    penalties.  Every penalty term is a non-decreasing function of exactly one
    metric's badness and every weight is clamped non-negative, which makes the
    score provably monotonic (see the module docstring for the exact contract).

    Overlapping UVs, mirrored triangles and unmeasurable (degenerate) triangles
    each carry a flat step penalty on top of a proportional one, so a layout
    with *any* of them can never outrank an otherwise-equal clean layout.

    Args:
        m: The metrics to score.
        w: Weights to apply.  Defaults to :class:`ScoreWeights` (``balanced``).

    Returns:
        A float in ``[0.0, 100.0]``.  An empty mesh scores ``0.0`` because there
        is nothing to be good at.
    """
    weights = w if w is not None else ScoreWeights()
    n_tris = int(m.n_tris) if isinstance(m.n_tris, (int, float)) else 0
    if n_tris <= 0:
        return 0.0

    w_angle = max(0.0, _badness(weights.angle, 0.0))
    w_area = max(0.0, _badness(weights.area, 0.0))
    w_stretch = max(0.0, _badness(weights.stretch, 0.0))
    w_seam = max(0.0, _badness(weights.seam, 0.0))
    w_pack = max(0.0, _badness(weights.packing, 0.0))
    w_overlap = max(0.0, _badness(weights.overlap, 0.0))
    w_flipped = max(0.0, _badness(weights.flipped, 0.0))

    penalty = 0.0

    # -- smooth distortion terms --------------------------------------------
    penalty += w_angle * (_badness(m.angle_distortion, 90.0) / _ANGLE_SCALE)
    penalty += w_area * (_badness(m.area_distortion, 8.0) / _AREA_SCALE)

    stretch = _badness(m.stretch_l2, 16.0)
    penalty += w_stretch * (max(stretch, 1.0) - 1.0) / _STRETCH_SCALE

    penalty += w_seam * (_badness(m.seam_ratio, 50.0) / _SEAM_SCALE)

    # Wasted UV space.  Squared so that the 0.6-0.8 efficiency a real packer
    # achieves is not treated as a catastrophe, while still ordering correctly.
    waste = 1.0 - _unit(m.packing_efficiency, 0.0)
    penalty += w_pack * waste * waste

    # -- hard defects --------------------------------------------------------
    overlap = _badness(m.overlap_area, 1.0)
    # Normalised against the layout's own used area so the term is scale free.
    # Using ``max(coverage, overlap)`` keeps the ratio in [0, 1] and keeps the
    # term monotone in *both* overlap_area (up) and uv_coverage (down).
    coverage = _badness(m.uv_coverage, 0.0)
    denom = max(coverage, overlap, 1e-12)
    overlap_ratio = min(1.0, overlap / denom)
    if overlap > 0.0:
        penalty += w_overlap * (_HARD_STEP + _HARD_SLOPE * overlap_ratio)

    flipped = max(0, int(_badness(m.flipped_tris, float(n_tris))))
    if flipped > 0:
        penalty += w_flipped * (_HARD_STEP + _HARD_SLOPE * min(1.0, flipped / n_tris))

    # Triangles we could not measure are a defect in their own right: without
    # this term an unwrap whose every triangle collapsed would report zero
    # distortion and score a perfect 100.
    degenerate = max(0, int(_badness(m.degenerate_tris, float(n_tris))))
    if degenerate > 0:
        penalty += w_flipped * _DEGENERATE_SLOPE * min(1.0, degenerate / n_tris)

    penalty = min(penalty, _MAX_PENALTY)
    return 100.0 * math.exp(-penalty)


# ---------------------------------------------------------------------------
# Pareto ranking
# ---------------------------------------------------------------------------

#: Chunk size for the pairwise dominance test; keeps the O(n^2) boolean matrix
#: bounded even if a caller hands us an unreasonable number of candidates.
_PARETO_CHUNK = 512


def _objectives(ms: Sequence[UVMetrics]) -> np.ndarray:
    """Return the ``(n, 3)`` minimisation objectives for the Pareto test.

    The three axes are the ones an artist actually trades off against each
    other: distortion, cut count and texture-space efficiency.  Packing is
    negated so that all three are "smaller is better".  Unmeasurable values
    become ``+inf``, which makes them dominated by anything measurable.
    """
    n = len(ms)
    obj = np.empty((n, 3), dtype=np.float64)
    for i, m in enumerate(ms):
        obj[i, 0] = _badness(m.stretch_l2, math.inf)
        obj[i, 1] = _badness(m.seam_ratio, math.inf)
        eff = float(m.packing_efficiency) if _is_finite(m.packing_efficiency) else -math.inf
        obj[i, 2] = -eff
    return obj


def _is_finite(value: Any) -> bool:
    """Return whether ``value`` converts to a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pareto_front(ms: Sequence[UVMetrics]) -> list[int]:
    """Return the indices of the non-dominated candidates.

    Domination is over ``(stretch_l2, seam_ratio, -packing_efficiency)``:
    candidate *i* dominates *j* when it is no worse on all three axes and
    strictly better on at least one.  Ties (identical objective vectors) are
    all kept, since neither dominates the other -- this is what lets the UI show
    "these three are equally good, pick by taste".

    Args:
        ms: The candidate metrics, in candidate order.

    Returns:
        Sorted indices into ``ms`` of the Pareto-optimal candidates.  An empty
        input returns an empty list.
    """
    n = len(ms)
    if n == 0:
        return []
    if n == 1:
        return [0]

    obj = _objectives(ms)
    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, _PARETO_CHUNK):
        stop = min(start + _PARETO_CHUNK, n)
        block = obj[start:stop]  # candidate dominators
        # (block, n, 3) comparisons: does block[i] dominate obj[j]?
        no_worse = (block[:, None, :] <= obj[None, :, :]).all(axis=2)
        better = (block[:, None, :] < obj[None, :, :]).any(axis=2)
        dominated |= (no_worse & better).any(axis=0)
    return [int(i) for i in np.flatnonzero(~dominated)]
