"""Tests for ``unwrap_variations.core.metrics`` -- ranking and scoring.

The score is what turns fifteen numbers into one, and it is the only thing
standing between the user and a wrong recommendation.  Its one non-negotiable
property is **monotonicity**: if layout B is better than layout A on every axis,
B must not score lower.  A score that violates that will, sooner or later, put
a worse unwrap at the top of the list -- and because the number looks
authoritative, the user will believe it.

Monotonicity is property-tested here rather than spot-checked, both field by
field and all fields at once, over randomised metric sets.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random

import pytest

from tests.conftest import import_or_skip

metrics = import_or_skip("unwrap_variations.core.metrics")

UVMetrics = metrics.UVMetrics
ScoreWeights = metrics.ScoreWeights
score = metrics.score
pareto_front = metrics.pareto_front

#: Metric fields where a *smaller* value is unambiguously a better unwrap.
LOWER_IS_BETTER = (
    "angle_distortion",
    "angle_distortion_max",
    "area_distortion",
    "stretch_l2",
    "stretch_linf",
    "conformal_error",
    "seam_length",
    "seam_ratio",
    "overlap_area",
    "flipped_tris",
    "degenerate_tris",
    "texel_density_cv",
    "out_of_bounds_tris",
)

#: Metric fields where a *larger* value is unambiguously better.
HIGHER_IS_BETTER = ("packing_efficiency", "uv_coverage")

#: Deliberately excluded from the monotonicity sweep: ``island_count`` trades
#: off against distortion (more cuts means less stretch), so "fewer islands is
#: better" is a preference, not a fact, and pinning it here would be asserting
#: an opinion rather than a property.
AMBIGUOUS = ("island_count",)

PRESET_NAMES = ("balanced", "low_distortion", "few_seams", "tight_packing", "game_ready")


def make_metrics(**kw) -> "metrics.UVMetrics":
    """Build a plausible mid-quality metrics record, overridden by ``kw``."""
    base = {
        "angle_distortion": 5.0,
        "angle_distortion_max": 30.0,
        "area_distortion": 0.15,
        "stretch_l2": 1.05,
        "stretch_linf": 1.8,
        "conformal_error": 0.2,
        "island_count": 8,
        "seam_length": 12.0,
        "seam_ratio": 1.5,
        "packing_efficiency": 0.6,
        "uv_coverage": 0.55,
        "overlap_area": 0.0,
        "flipped_tris": 0,
        "degenerate_tris": 0,
        "texel_density_cv": 0.2,
        "uv_bounds": (0.0, 0.0, 1.0, 1.0),
        "out_of_bounds_tris": 0,
        "n_tris": 1000,
        "eval_seconds": 0.01,
    }
    base.update(kw)
    return UVMetrics(**base)


def random_metrics(rng: random.Random) -> "metrics.UVMetrics":
    """Build a random but plausible metrics record."""
    return make_metrics(
        angle_distortion=rng.uniform(0.0, 45.0),
        angle_distortion_max=rng.uniform(0.0, 170.0),
        area_distortion=rng.uniform(0.0, 3.0),
        stretch_l2=1.0 + rng.uniform(0.0, 4.0),
        stretch_linf=1.0 + rng.uniform(0.0, 12.0),
        conformal_error=rng.uniform(0.0, 5.0),
        island_count=rng.randint(1, 200),
        seam_length=rng.uniform(0.0, 200.0),
        seam_ratio=rng.uniform(0.0, 20.0),
        packing_efficiency=rng.uniform(0.05, 0.95),
        uv_coverage=rng.uniform(0.05, 0.95),
        overlap_area=rng.choice([0.0, 0.0, rng.uniform(0.0, 0.5)]),
        flipped_tris=rng.choice([0, 0, rng.randint(1, 400)]),
        degenerate_tris=rng.choice([0, 0, rng.randint(1, 200)]),
        texel_density_cv=rng.uniform(0.0, 2.0),
        out_of_bounds_tris=rng.choice([0, rng.randint(1, 500)]),
    )


def improve(m, factor: float = 0.5):
    """Return a strictly better version of ``m`` on every unambiguous axis."""
    changes = {}
    for field in LOWER_IS_BETTER:
        value = getattr(m, field)
        if field == "stretch_l2" or field == "stretch_linf":
            changes[field] = 1.0 + (value - 1.0) * factor
        elif isinstance(value, int):
            changes[field] = int(value * factor)
        else:
            changes[field] = value * factor
    for field in HIGHER_IS_BETTER:
        value = float(getattr(m, field))
        changes[field] = value + (1.0 - value) * (1.0 - factor)
    return dataclasses.replace(m, **changes)


# ---------------------------------------------------------------------------
# Basic score behaviour.
# ---------------------------------------------------------------------------


def test_score_is_bounded():
    """The score is a percentage: never below 0, never above 100."""
    rng = random.Random(0)
    for _ in range(400):
        value = score(random_metrics(rng), ScoreWeights())
        assert 0.0 <= value <= 100.0, "score %r escaped 0..100" % value
        assert math.isfinite(value)


def test_perfect_metrics_score_near_100():
    """A flawless layout scores at or near the maximum."""
    perfect = make_metrics(
        angle_distortion=0.0,
        angle_distortion_max=0.0,
        area_distortion=0.0,
        stretch_l2=1.0,
        stretch_linf=1.0,
        conformal_error=0.0,
        seam_length=0.0,
        seam_ratio=0.0,
        packing_efficiency=1.0,
        uv_coverage=1.0,
        overlap_area=0.0,
        flipped_tris=0,
        degenerate_tris=0,
        texel_density_cv=0.0,
        out_of_bounds_tris=0,
    )
    assert score(perfect, ScoreWeights()) == pytest.approx(100.0, abs=1e-6)


def test_terrible_metrics_score_far_below_a_good_one():
    """The bad-unwrap control is ranked far below the good one."""
    good = make_metrics()
    bad = make_metrics(
        angle_distortion=14.0,
        area_distortion=0.94,
        stretch_l2=3.08,
        stretch_linf=10.8,
        conformal_error=3.0,
        packing_efficiency=0.2,
        overlap_area=0.78,
        flipped_tris=1104,
        n_tris=2208,
    )
    assert score(bad, ScoreWeights()) < score(good, ScoreWeights()) * 0.5


def test_empty_mesh_scores_zero():
    """Nothing to be good at means no score, not a free 100."""
    assert score(UVMetrics(n_tris=0), ScoreWeights()) == 0.0


def test_score_accepts_default_weights():
    """``score(m)`` without weights uses the balanced preset."""
    m = make_metrics()
    assert score(m) == pytest.approx(score(m, ScoreWeights()))


# ---------------------------------------------------------------------------
# Monotonicity -- the property that matters.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_uniformly_better_metrics_never_score_lower(preset):
    """Improving every unambiguous axis at once never lowers the score.

    This is the headline guarantee.  If it fails, the ranking can put a worse
    unwrap above a better one, and every recommendation the add-on makes is
    suspect.
    """
    weights = ScoreWeights.preset(preset)
    rng = random.Random(1234)
    for _ in range(500):
        worse = random_metrics(rng)
        better = improve(worse)
        s_worse = score(worse, weights)
        s_better = score(better, weights)
        assert s_better >= s_worse - 1e-9, (
            "preset %r: improving every metric lowered the score from %.6f to "
            "%.6f\n  worse  = %r\n  better = %r" % (preset, s_worse, s_better, worse, better)
        )


@pytest.mark.parametrize("field", LOWER_IS_BETTER)
def test_lowering_one_metric_never_lowers_the_score(field):
    """Each "smaller is better" metric is individually monotone.

    Held one at a time, because a bug that cancels out across several fields at
    once still shows up here.
    """
    weights = ScoreWeights()
    rng = random.Random(hash(field) & 0xFFFF)
    for _ in range(120):
        worse = random_metrics(rng)
        value = getattr(worse, field)
        if field in ("stretch_l2", "stretch_linf"):
            improved = 1.0 + (float(value) - 1.0) * 0.4
        elif isinstance(value, int):
            improved = value // 2
        else:
            improved = float(value) * 0.4
        better = dataclasses.replace(worse, **{field: improved})
        assert score(better, weights) >= score(worse, weights) - 1e-9, (
            "reducing %s from %r to %r lowered the score (%.9f -> %.9f)"
            % (field, value, improved, score(worse, weights), score(better, weights))
        )


@pytest.mark.parametrize("field", HIGHER_IS_BETTER)
def test_raising_one_metric_never_lowers_the_score(field):
    """Each "larger is better" metric is individually monotone."""
    weights = ScoreWeights()
    rng = random.Random(hash(field) & 0xFFFF)
    for _ in range(120):
        worse = random_metrics(rng)
        value = float(getattr(worse, field))
        better = dataclasses.replace(worse, **{field: min(1.0, value + 0.3)})
        assert score(better, weights) >= score(worse, weights) - 1e-9, (
            "raising %s from %r lowered the score" % (field, value)
        )


def test_overlap_is_a_hard_penalty():
    """Any overlap at all costs materially more than a rounding error.

    Overlapping UVs are not a matter of degree to a texture painter: they mean
    two parts of the model share texels, and painting one paints the other.
    """
    clean = make_metrics(overlap_area=0.0)
    dirty = make_metrics(overlap_area=1e-4)
    very_dirty = make_metrics(overlap_area=0.5)
    assert score(dirty) < score(clean)
    assert score(very_dirty) < score(dirty)
    assert score(dirty) < score(clean) * 0.95, (
        "a layout with overlapping UVs scored %.3f against a clean %.3f -- the "
        "step penalty is too small to keep it out of the top slot"
        % (score(dirty), score(clean))
    )


def test_flipped_faces_are_a_hard_penalty():
    """A single mirrored triangle is a real defect, not a rounding difference."""
    clean = make_metrics(flipped_tris=0)
    one = make_metrics(flipped_tris=1)
    many = make_metrics(flipped_tris=500)
    assert score(one) < score(clean) * 0.95
    assert score(many) < score(one)


def test_fully_degenerate_layout_cannot_outrank_a_real_one():
    """A collapsed unwrap reports zero distortion; it must not win because of it.

    ``evaluate`` reports the neutral ``stretch_l2 = 1.0`` when no triangle is
    measurable, which on the distortion axes looks *perfect*.  The only thing
    stopping a completely broken candidate from topping the list is the
    degenerate-triangle penalty, so that penalty is load-bearing.
    """
    collapsed = make_metrics(
        angle_distortion=0.0,
        area_distortion=0.0,
        stretch_l2=1.0,
        stretch_linf=1.0,
        conformal_error=0.0,
        texel_density_cv=0.0,
        degenerate_tris=1000,
        n_tris=1000,
        packing_efficiency=0.0,
        uv_coverage=0.0,
    )
    real = make_metrics()
    assert score(collapsed) < score(real), (
        "a layout whose every triangle collapsed scored %.3f against a real "
        "unwrap's %.3f" % (score(collapsed), score(real))
    )


def test_score_is_deterministic():
    """Same metrics, same weights, same float."""
    m = make_metrics()
    w = ScoreWeights.preset("game_ready")
    assert score(m, w) == score(m, w)


def test_score_never_raises_on_garbage():
    """Hostile or non-finite metric values are contained, not propagated."""
    for bad in (
        make_metrics(stretch_l2=float("nan")),
        make_metrics(angle_distortion=float("inf")),
        make_metrics(packing_efficiency=float("-inf")),
        make_metrics(overlap_area=float("nan"), uv_coverage=float("nan")),
        make_metrics(n_tris=-5),
        make_metrics(flipped_tris=-1),
    ):
        value = score(bad, ScoreWeights())
        assert math.isfinite(value)
        assert 0.0 <= value <= 100.0


# ---------------------------------------------------------------------------
# Weights.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_exists(name):
    """All five named presets from the contract resolve to real weights."""
    w = ScoreWeights.preset(name)
    assert isinstance(w, ScoreWeights)
    for field in dataclasses.fields(ScoreWeights):
        assert float(getattr(w, field.name)) >= 0.0


def test_presets_are_actually_different():
    """The presets express different priorities, not the same one five times."""
    seen = {ScoreWeights.preset(n).to_dict()["stretch"] for n in PRESET_NAMES}
    tuples = {tuple(sorted(ScoreWeights.preset(n).to_dict().items())) for n in PRESET_NAMES}
    assert len(tuples) >= 4, "only %d distinct presets" % len(tuples)
    assert seen  # at least the stretch weight varies somewhere


def test_preset_names_are_case_insensitive_and_unknown_falls_back():
    """A stale preference string in a ``.blend`` must not be able to raise."""
    assert ScoreWeights.preset("BALANCED") == ScoreWeights.preset("balanced")
    assert ScoreWeights.preset("  game_ready ") == ScoreWeights.preset("game_ready")
    assert ScoreWeights.preset("no_such_preset") == ScoreWeights.preset("balanced")
    assert ScoreWeights.preset("") == ScoreWeights.preset("balanced")
    assert ScoreWeights.preset(None) == ScoreWeights.preset("balanced")


def test_zero_weight_disables_a_term():
    """Setting a weight to zero removes that metric from the score entirely."""
    off = ScoreWeights(angle=0.0)
    a = make_metrics(angle_distortion=0.0)
    b = make_metrics(angle_distortion=40.0)
    assert score(a, off) == pytest.approx(score(b, off))
    on = ScoreWeights(angle=1.0)
    assert score(a, on) > score(b, on)


def test_negative_weights_are_clamped_not_inverted():
    """A negative weight is treated as "off"; inverting a term breaks ranking."""
    negative = ScoreWeights(stretch=-5.0)
    zero = ScoreWeights(stretch=0.0)
    m = make_metrics(stretch_l2=2.5)
    assert score(m, negative) == pytest.approx(score(m, zero))


def test_weights_round_trip_through_dict():
    """Weights survive ``to_dict``/``from_dict`` and a JSON hop."""
    for name in PRESET_NAMES:
        w = ScoreWeights.preset(name)
        assert ScoreWeights.from_dict(json.loads(json.dumps(w.to_dict()))) == w


def test_weights_from_dict_is_tolerant():
    """Unknown keys, missing keys and junk values fall back to defaults."""
    assert ScoreWeights.from_dict(None) == ScoreWeights()
    assert ScoreWeights.from_dict({}) == ScoreWeights()
    assert ScoreWeights.from_dict({"nonsense": 1}) == ScoreWeights()
    got = ScoreWeights.from_dict({"angle": "not a number", "area": 3.0})
    assert got.area == 3.0
    assert got.angle == ScoreWeights().angle


# ---------------------------------------------------------------------------
# UVMetrics serialisation.
# ---------------------------------------------------------------------------


def test_metrics_round_trip_through_dict():
    """Metrics survive the trip into a ``.blend`` ID property and back."""
    rng = random.Random(7)
    for _ in range(50):
        m = random_metrics(rng)
        again = UVMetrics.from_dict(json.loads(json.dumps(m.to_dict())))
        assert again == m, "round trip changed %r -> %r" % (m, again)


def test_metrics_to_dict_is_json_safe():
    """Every value in the dict is a JSON scalar or list, including ``uv_bounds``."""
    data = make_metrics().to_dict()
    text = json.dumps(data)
    assert isinstance(json.loads(text), dict)


def test_metrics_from_dict_is_tolerant():
    """Unknown and missing keys are survivable -- forwards compatibility."""
    data = make_metrics().to_dict()
    data["a_field_from_the_future"] = [1, 2, 3]
    del data["stretch_linf"]
    m = UVMetrics.from_dict(data)
    assert m.stretch_l2 == pytest.approx(1.05)
    assert UVMetrics.from_dict({}) == UVMetrics()
    assert UVMetrics.from_dict(None) == UVMetrics()


def test_metrics_summary_is_ascii():
    """The one-line summary is safe for a Windows cp1252 console."""
    make_metrics().summary().encode("ascii")


@pytest.mark.xfail(
    reason=(
        "BUG in core/metrics.py UVMetrics.from_dict: the int/float conversions "
        "sit inside a `try` that catches only (TypeError, ValueError). "
        "int(float('inf')) and float(<huge python int>) both raise "
        "OverflowError, which escapes. Same class as the ParamSpec.clamp bug. "
        "Reachable from disk: json emits and accepts `Infinity`, and a JSON "
        "integer literal of 400 digits parses to a Python int that float() "
        "cannot take. Fix: add OverflowError to the except clause (and "
        "saturate non-finite values toward the matching bound, as "
        "ParamSpec.clamp now does)."
    ),
    raises=OverflowError,
    strict=False,
)
@pytest.mark.parametrize(
    "payload",
    [
        '{"n_tris": Infinity}',
        '{"flipped_tris": -Infinity}',
        '{"island_count": Infinity}',
        '{"degenerate_tris": Infinity}',
        '{"out_of_bounds_tris": Infinity}',
        '{"stretch_l2": %s}' % ("9" * 400),
        '{"angle_distortion": -%s}' % ("9" * 400),
    ],
)
def test_metrics_from_dict_contains_non_finite_values(payload):
    """Deserialising metrics is a trust boundary and must never raise.

    ``from_dict`` promises that "a ``.blend`` written by a newer add-on version
    must still open in an older one".  An exception here does not degrade a
    candidate, it propagates out of the file-load path.
    """
    m = UVMetrics.from_dict(json.loads(payload))
    assert isinstance(m, UVMetrics)
    assert math.isfinite(float(m.stretch_l2))
    assert int(m.n_tris) >= 0
    # And whatever came back must still be scoreable.
    assert 0.0 <= score(m, ScoreWeights()) <= 100.0


@pytest.mark.xfail(
    reason=(
        "BUG in core/metrics.py ScoreWeights.from_dict: float(value) inside a "
        "`try` catching only (TypeError, ValueError); a JSON integer literal "
        "too large for a float raises OverflowError and escapes. Fix: add "
        "OverflowError to the except clause."
    ),
    raises=OverflowError,
    strict=False,
)
def test_weights_from_dict_contains_an_oversized_integer():
    """A stored weight too large to be a float falls back, it does not raise."""
    w = ScoreWeights.from_dict(json.loads('{"angle": %s}' % ("9" * 400)))
    assert math.isfinite(w.angle)


def test_weights_from_dict_contains_infinity():
    """A non-finite weight is accepted here and neutralised inside ``score``."""
    w = ScoreWeights.from_dict(json.loads('{"angle": Infinity}'))
    value = score(make_metrics(), w)
    assert math.isfinite(value)
    assert 0.0 <= value <= 100.0


# ---------------------------------------------------------------------------
# pareto_front.
# ---------------------------------------------------------------------------


def test_pareto_front_of_nothing_is_empty():
    """An empty candidate set has an empty front, not an exception."""
    assert pareto_front([]) == []


def test_pareto_front_of_one_is_that_one():
    """A single candidate is trivially non-dominated."""
    assert pareto_front([make_metrics()]) == [0]


def test_pareto_front_excludes_a_dominated_candidate():
    """A candidate worse on all three axes is dominated and drops out."""
    best = make_metrics(stretch_l2=1.01, seam_ratio=0.5, packing_efficiency=0.9)
    worst = make_metrics(stretch_l2=2.0, seam_ratio=3.0, packing_efficiency=0.2)
    front = pareto_front([best, worst])
    assert front == [0]


def test_pareto_front_keeps_genuine_trade_offs():
    """Candidates that each win on a different axis are all kept."""
    low_stretch = make_metrics(stretch_l2=1.01, seam_ratio=5.0, packing_efficiency=0.4)
    few_seams = make_metrics(stretch_l2=2.0, seam_ratio=0.2, packing_efficiency=0.4)
    tight_pack = make_metrics(stretch_l2=2.0, seam_ratio=5.0, packing_efficiency=0.95)
    front = pareto_front([low_stretch, few_seams, tight_pack])
    assert sorted(front) == [0, 1, 2]


def test_pareto_front_keeps_ties():
    """Identical candidates do not dominate each other, so both survive."""
    m = make_metrics()
    assert sorted(pareto_front([m, m])) == [0, 1]


def test_pareto_front_always_contains_the_best_on_each_axis():
    """Whatever else it does, the single best candidate per axis is in the front."""
    rng = random.Random(99)
    ms = [random_metrics(rng) for _ in range(60)]
    front = set(pareto_front(ms))
    assert front
    best_stretch = min(range(len(ms)), key=lambda i: ms[i].stretch_l2)
    best_seam = min(range(len(ms)), key=lambda i: ms[i].seam_ratio)
    best_pack = max(range(len(ms)), key=lambda i: ms[i].packing_efficiency)
    for i in (best_stretch, best_seam, best_pack):
        assert i in front, "candidate %d is best on an axis but not in the front" % i


def test_pareto_front_returns_valid_sorted_unique_indices():
    """The result is a clean index list, safe to feed straight into a UI list."""
    rng = random.Random(5)
    ms = [random_metrics(rng) for _ in range(40)]
    front = pareto_front(ms)
    assert front == sorted(front)
    assert len(set(front)) == len(front)
    assert all(0 <= i < len(ms) for i in front)


def test_pareto_front_tolerates_non_finite_metrics():
    """A failed candidate with NaN metrics is dominated, never a crash."""
    good = make_metrics()
    broken = make_metrics(stretch_l2=float("nan"), seam_ratio=float("nan"))
    front = pareto_front([good, broken])
    assert 0 in front
    assert 1 not in front
