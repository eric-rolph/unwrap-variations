"""Tests for ``unwrap_variations.core.params`` -- the search space.

Two properties matter more than anything else here.

**Determinism.**  A grid is the user's exploration.  If ``sample_grid(10, seed=7)``
returns different recipes on Tuesday than it did on Monday, the candidate they
liked and wrote down is gone, and re-running a "reproducible" seed proves
nothing.

**Uniqueness.**  Every duplicate recipe in a 10-slot grid is a wasted unwrap --
seconds of solver time spent producing a result the user has already seen.  The
contract requires de-duplication by ``signature()``, and ``mutate()`` must never
hand back the parent it was asked to vary.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from unwrap_variations.core.params import (
    ARCHETYPES,
    METHODS,
    PARAM_SPACE,
    SEAM_STRATEGIES,
    ParamSpec,
    UnwrapParams,
    mutate,
    sample_grid,
)

# ---------------------------------------------------------------------------
# Signatures.
# ---------------------------------------------------------------------------


def test_signature_is_twelve_hex_chars():
    """The signature is a stable 12-character hex string, as the contract says."""
    sig = UnwrapParams().signature()
    assert len(sig) == 12
    assert all(c in "0123456789abcdef" for c in sig)


def test_signature_is_stable_across_calls_and_instances():
    """Identical recipes hash identically; the hash never drifts between calls."""
    a = UnwrapParams(method="CONFORMAL", island_margin=0.02)
    b = UnwrapParams(method="CONFORMAL", island_margin=0.02)
    assert a.signature() == a.signature() == b.signature()


def test_signature_distinguishes_meaningful_changes():
    """A change that alters the resulting UVs alters the signature."""
    base = UnwrapParams(method="ANGLE_BASED", seam_strategy="ANGLE", seam_angle_deg=30.0)
    assert base.signature() != base.replace(method="CONFORMAL").signature()
    assert base.signature() != base.replace(seam_angle_deg=60.0).signature()
    assert base.signature() != base.replace(island_margin=0.05).signature()


def test_signature_ignores_irrelevant_fields():
    """Fields with no effect for the chosen method do not perturb the signature.

    ``CUBE_PROJECT`` ignores seams entirely, so two cube projections that differ
    only in ``seam_strategy`` are the *same recipe* and must collapse to one
    grid slot rather than burning two.
    """
    a = UnwrapParams(method="CUBE_PROJECT", seam_strategy="SHARP")
    b = UnwrapParams(method="CUBE_PROJECT", seam_strategy="DUAL_GRAPH")
    assert a.signature() == b.signature()


def test_signature_tolerates_float_noise():
    """Values differing only by float noise hash the same."""
    a = UnwrapParams(island_margin=0.01)
    b = UnwrapParams(island_margin=0.01 + 1e-12)
    assert a.signature() == b.signature()


# ---------------------------------------------------------------------------
# Serialisation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_round_trip_through_dict(seed):
    """``from_dict(to_dict(p)) == p`` for every recipe a grid can produce."""
    for params in sample_grid(12, seed=seed):
        again = UnwrapParams.from_dict(params.to_dict())
        assert again == params, "round trip changed %r -> %r" % (params, again)
        assert again.signature() == params.signature()


def test_to_dict_is_json_safe():
    """The dict form survives a real ``json`` round trip -- it goes into a .blend."""
    params = sample_grid(1, seed=3)[0]
    text = json.dumps(params.to_dict(), sort_keys=True)
    assert UnwrapParams.from_dict(json.loads(text)) == params


def test_from_dict_tolerates_unknown_keys():
    """Unknown keys are dropped, not raised on.

    A ``.blend`` written by a newer add-on carries fields this build has never
    heard of.  Refusing to open the file would be the wrong trade.
    """
    data = UnwrapParams(method="CONFORMAL").to_dict()
    data["future_field"] = 42
    data["another"] = {"nested": [1, 2, 3]}
    params = UnwrapParams.from_dict(data)
    assert params.method == "CONFORMAL"


def test_from_dict_tolerates_missing_keys():
    """Missing keys fall back to defaults."""
    params = UnwrapParams.from_dict({"method": "CUBE_PROJECT"})
    assert params.method == "CUBE_PROJECT"
    assert params.island_margin == UnwrapParams().island_margin


def test_from_dict_tolerates_empty_and_none():
    """An empty or absent payload yields defaults rather than an exception."""
    assert UnwrapParams.from_dict({}) == UnwrapParams()
    assert UnwrapParams.from_dict(None) == UnwrapParams()


def test_from_dict_clamps_hostile_values():
    """Deserialisation is a trust boundary: it clamps, it does not trust."""
    params = UnwrapParams.from_dict(
        {
            "method": "NOT_A_METHOD",
            "seam_strategy": "NONSENSE",
            "angle_limit_deg": 1e9,
            "island_margin": -5.0,
            "iterations": 10**9,
            "curvature_threshold": float("nan"),
            "margin_method": "WAT",
        }
    )
    assert params.method in METHODS
    assert params.seam_strategy in SEAM_STRATEGIES
    assert PARAM_SPACE["angle_limit_deg"].low <= params.angle_limit_deg <= 89.0
    assert 0.0 <= params.island_margin <= 0.2
    assert 1 <= params.iterations <= 100
    assert math.isfinite(params.curvature_threshold)
    assert params.margin_method in PARAM_SPACE["margin_method"].choices


# ---------------------------------------------------------------------------
# Clamping.
# ---------------------------------------------------------------------------


def test_clamped_forces_out_of_range_into_range():
    """Every numeric field is pulled back inside its declared bounds."""
    wild = UnwrapParams(
        angle_limit_deg=-1000.0,
        island_margin=99.0,
        area_weight=17.0,
        iterations=-4,
        seam_angle_deg=1e6,
        curvature_threshold=-2.0,
        minimize_stretch_iters=999,
    ).clamped()
    for name, spec in PARAM_SPACE.items():
        value = getattr(wild, name)
        if spec.kind in ("float", "int"):
            assert spec.low <= value <= spec.high, "%s=%r escaped [%r, %r]" % (
                name,
                value,
                spec.low,
                spec.high,
            )
        elif spec.kind == "enum":
            assert value in spec.choices


def test_clamped_replaces_garbage_types():
    """Strings, ``None`` and NaN where numbers belong become legal numbers."""
    junk = UnwrapParams(
        angle_limit_deg="banana",
        island_margin=None,
        iterations="12",
        curvature_threshold=float("inf"),
        method=None,
        seam_strategy=object(),
    ).clamped()
    assert isinstance(junk.angle_limit_deg, float)
    assert math.isfinite(junk.angle_limit_deg)
    assert isinstance(junk.island_margin, float)
    assert math.isfinite(junk.island_margin)
    assert junk.iterations == 12
    assert math.isfinite(junk.curvature_threshold)
    assert junk.method in METHODS
    assert junk.seam_strategy in SEAM_STRATEGIES


def test_clamped_is_idempotent():
    """Clamping twice changes nothing the second time."""
    once = UnwrapParams(angle_limit_deg=1e9).clamped()
    assert once.clamped() == once


def test_replace_clamps():
    """``replace`` funnels through the same clamp, so no caller can bypass it."""
    params = UnwrapParams().replace(angle_limit_deg=500.0, method="ZZZ")
    assert params.angle_limit_deg <= 89.0
    assert params.method in METHODS


@pytest.mark.parametrize("name,spec", sorted(PARAM_SPACE.items()))
def test_every_param_spec_clamps_hostile_input(name, spec):
    """Each :class:`ParamSpec` survives NaN, None, strings and huge magnitudes."""
    for hostile in (None, "x", float("nan"), -1e300, 1e300, [], {}, b"\x00"):
        value = spec.clamp(hostile)
        if spec.kind == "enum":
            assert value in spec.choices
        elif spec.kind == "bool":
            assert isinstance(value, bool)
        else:
            assert isinstance(value, (int, float))
            assert math.isfinite(float(value))
            assert spec.low <= value <= spec.high


@pytest.mark.parametrize("field", ["iterations", "minimize_stretch_iters"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinite_values_clamp_into_int_fields(field, value):
    """An infinite value in an int field must clamp, not raise.

    ``clamped()`` is the trust boundary between stored data and the Blender
    operators, and the contract says it forces garbage into range.  Infinity is
    garbage; the only question is whether it is *contained* garbage.

    Regression test.  ``ParamSpec.clamp`` used to call ``int(value)`` inside a
    ``try`` that caught only ``(TypeError, ValueError)``; ``int(float('inf'))``
    raises ``OverflowError``, which escaped.  That was reachable from disk --
    ``json.dumps`` writes ``Infinity`` and ``json.loads`` reads it back -- so a
    single stored infinity made a whole candidate set fail to load.
    """
    params = UnwrapParams(**{field: value}).clamped()
    spec = PARAM_SPACE[field]
    assert spec.low <= getattr(params, field) <= spec.high


@pytest.mark.parametrize("name,spec", sorted(PARAM_SPACE.items()))
def test_non_finite_values_saturate_toward_the_matching_bound(name, spec):
    """Infinities saturate in the direction they point; NaN falls back to the low bound.

    The *direction* is the part worth pinning.  Clamping everything to the low
    bound would also "not raise", and would silently turn a corrupted
    ``iterations = Infinity`` into 1 -- a value that looks deliberate, produces
    a plausible-but-wrong unwrap, and gives the user nothing to notice.
    """
    if spec.kind in ("float", "int"):
        assert spec.clamp(float("inf")) == spec.high, (
            "%s: +inf should saturate to the high bound %r" % (name, spec.high)
        )
        assert spec.clamp(float("-inf")) == spec.low, (
            "%s: -inf should saturate to the low bound %r" % (name, spec.low)
        )
        assert spec.clamp(float("nan")) == spec.low, (
            "%s: NaN carries no direction, so it takes the low bound %r"
            % (name, spec.low)
        )
    elif spec.kind == "enum":
        for value in (float("inf"), float("-inf"), float("nan")):
            assert spec.clamp(value) in spec.choices


@pytest.mark.parametrize("value", [10**400, -(10**400), 10**4000])
def test_oversized_python_ints_are_contained(value):
    """An int too large to convert to a float lands in range rather than raising.

    ``float(10**400)`` raises ``OverflowError``; a narrow ``except`` on the load
    path turns that into a ``.blend`` that will not open.
    """
    for name, spec in PARAM_SPACE.items():
        got = spec.clamp(value)
        if spec.kind in ("float", "int"):
            assert spec.low <= got <= spec.high, "%s: %r escaped its range" % (name, got)
            assert math.isfinite(float(got))
        elif spec.kind == "enum":
            assert got in spec.choices


def test_infinite_values_survive_a_json_round_trip():
    """The exact path that used to break: ``Infinity`` written and read back.

    ``json`` accepts and emits ``Infinity`` by default, so a metric or parameter
    that went non-finite once is on disk forever.  Loading it must clamp.
    """
    payload = json.dumps(
        {"iterations": float("inf"), "angle_limit_deg": float("-inf"),
         "curvature_threshold": float("nan"), "minimize_stretch_iters": float("inf")}
    )
    assert "Infinity" in payload
    params = UnwrapParams.from_dict(json.loads(payload))
    assert params.iterations == PARAM_SPACE["iterations"].high
    assert params.minimize_stretch_iters == PARAM_SPACE["minimize_stretch_iters"].high
    assert params.angle_limit_deg == PARAM_SPACE["angle_limit_deg"].low
    assert math.isfinite(params.curvature_threshold)
    assert params.clamped() == params


def test_param_space_names_are_real_fields():
    """Every searched name is an actual field of :class:`UnwrapParams`."""
    fields = set(UnwrapParams.__dataclass_fields__)
    for name in PARAM_SPACE:
        assert name in fields, "PARAM_SPACE has %r, which is not a field" % name


def test_param_spec_is_frozen():
    """``ParamSpec`` is immutable -- the space cannot be mutated at runtime."""
    spec = PARAM_SPACE["island_margin"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.weight = 99.0  # type: ignore[misc]
    assert isinstance(spec, ParamSpec)


# ---------------------------------------------------------------------------
# sample_grid.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 5, 10, 24])
@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_sample_grid_is_deterministic(count, seed):
    """The same seed always yields byte-identical recipes, in the same order."""
    first = sample_grid(count, seed=seed)
    second = sample_grid(count, seed=seed)
    assert [p.signature() for p in first] == [p.signature() for p in second]
    assert first == second


@pytest.mark.parametrize("count", [1, 2, 5, 10, 24, 60])
def test_sample_grid_returns_unique_signatures(count):
    """No slot is wasted on a duplicate recipe."""
    grid = sample_grid(count, seed=7)
    sigs = [p.signature() for p in grid]
    assert len(set(sigs)) == len(sigs), "duplicate recipes in a %d-slot grid" % count
    assert len(grid) == count, "asked for %d recipes, got %d" % (count, len(grid))


@pytest.mark.parametrize("count", [0, -1, -100])
def test_sample_grid_rejects_nonpositive_counts(count):
    """Zero or negative counts yield an empty list, not an exception."""
    assert sample_grid(count, seed=0) == []


def test_sample_grid_differs_between_seeds():
    """Different seeds explore different regions of the space."""
    a = {p.signature() for p in sample_grid(10, seed=1)}
    b = {p.signature() for p in sample_grid(10, seed=2)}
    assert a != b


def test_sample_grid_starts_from_archetypes():
    """The first slots are the curated, production-sensible recipes.

    A grid whose first entries are random noise looks broken to a user even when
    it is not, so the recognisable layouts come first.
    """
    grid = sample_grid(len(ARCHETYPES), seed=0)
    archetype_sigs = {
        UnwrapParams().replace(seed=0, **a).signature() for a in ARCHETYPES
    }
    assert {p.signature() for p in grid} <= archetype_sigs


def test_sample_grid_honours_a_base():
    """Fields that are not searched are inherited from ``base``."""
    base = UnwrapParams(backend="xatlas")
    for params in sample_grid(6, seed=4, base=base):
        assert params.backend == "xatlas"


def test_sample_grid_output_is_clamped():
    """Nothing a grid produces can be out of range."""
    for params in sample_grid(40, seed=9):
        assert params.clamped() == params, "%r came out unclamped" % (params,)


def test_sample_grid_covers_several_methods():
    """A 10-slot grid is genuinely diverse, not ten tweaks of one solver."""
    methods = {p.method for p in sample_grid(10, seed=0)}
    assert len(methods) >= 3, "only %d distinct methods in a 10-slot grid" % len(methods)


# ---------------------------------------------------------------------------
# mutate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", [0.0, 0.1, 0.35, 0.75, 1.0])
def test_mutate_never_returns_the_parent(strength):
    """Branching must never hand back the recipe it was asked to vary.

    A child identical to its parent is a wasted solver run and, worse, makes the
    branch UI look broken -- the user asked for variations and got a copy.
    """
    parent = UnwrapParams(method="ANGLE_BASED", seam_strategy="ANGLE").clamped()
    children = mutate(parent, count=12, strength=strength, seed=3)
    parent_sig = parent.signature()
    for child in children:
        assert child.signature() != parent_sig, (
            "mutate(strength=%g) returned the parent unchanged" % strength
        )


@pytest.mark.parametrize("seed", [0, 1, 99])
def test_mutate_is_deterministic(seed):
    """The same parent, strength and seed always give the same children."""
    parent = sample_grid(1, seed=0)[0]
    a = mutate(parent, count=8, strength=0.4, seed=seed)
    b = mutate(parent, count=8, strength=0.4, seed=seed)
    assert [p.signature() for p in a] == [p.signature() for p in b]
    assert a == b


def test_mutate_returns_unique_children():
    """Children are de-duplicated against each other and the parent."""
    parent = UnwrapParams().clamped()
    children = mutate(parent, count=16, strength=0.5, seed=5)
    sigs = [c.signature() for c in children]
    assert len(set(sigs)) == len(sigs)
    assert parent.signature() not in sigs


@pytest.mark.parametrize("count", [0, -3])
def test_mutate_rejects_nonpositive_counts(count):
    """Zero or negative counts yield an empty list."""
    assert mutate(UnwrapParams(), count=count, strength=0.3, seed=0) == []


def test_mutate_output_is_clamped():
    """Even at maximum strength, no child escapes its bounds."""
    parent = UnwrapParams().clamped()
    for child in mutate(parent, count=30, strength=1.0, seed=11):
        assert child.clamped() == child
        for name, spec in PARAM_SPACE.items():
            value = getattr(child, name)
            if spec.kind in ("float", "int"):
                assert spec.low <= value <= spec.high, "%s=%r out of range" % (name, value)


def test_mutate_clamps_hostile_strength():
    """Out-of-range ``strength`` is clamped, not propagated into ``random.gauss``."""
    parent = UnwrapParams().clamped()
    for strength in (-1.0, 5.0):
        children = mutate(parent, count=4, strength=strength, seed=2)
        assert len(children) == 4
        for child in children:
            assert child.clamped() == child


def test_low_strength_mutations_stay_near_the_parent():
    """At low strength the children are siblings, not strangers.

    Measured by how often the solver itself changes: at strength 0.05 a jump to
    a different method should be rare, otherwise "refine this one" does not
    refine anything.
    """
    parent = UnwrapParams(method="ANGLE_BASED", seam_strategy="ANGLE").clamped()
    children = mutate(parent, count=40, strength=0.02, seed=17)
    same_method = sum(1 for c in children if c.method == parent.method)
    assert same_method >= len(children) * 0.5, (
        "only %d of %d low-strength children kept the parent's method"
        % (same_method, len(children))
    )


def test_high_strength_mutations_explore_widely():
    """At maximum strength the method itself flips, so deep branches do not stall."""
    parent = UnwrapParams(method="ANGLE_BASED").clamped()
    children = mutate(parent, count=40, strength=1.0, seed=17)
    methods = {c.method for c in children}
    assert len(methods) >= 2


# ---------------------------------------------------------------------------
# describe().
# ---------------------------------------------------------------------------


def test_describe_is_short_and_ascii():
    """The UI label fits and contains no character a cp1252 console will choke on."""
    for params in sample_grid(24, seed=0) + mutate(
        UnwrapParams(), count=24, strength=1.0, seed=1
    ):
        label = params.describe()
        assert isinstance(label, str)
        assert 0 < len(label) <= 40, "describe() returned %d chars: %r" % (len(label), label)
        label.encode("ascii")


def test_describe_distinguishes_methods():
    """Different solvers read differently in the list."""
    labels = {
        UnwrapParams(method=m).describe() for m in METHODS
    }
    assert len(labels) == len(METHODS)
