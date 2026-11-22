"""Tests for ``unwrap_variations.core.evaluator`` -- the metric engine.

This is the file the whole project leans on.  If the evaluator is wrong, every
score, every ranking and the entire "best of N beats one-shot" product claim is
wrong with it, silently and plausibly.  So the evaluator is checked three ways:

1. **Against provable invariants.**  Mesh-wide normalised L2 stretch is
   mathematically ``>= 1.0``; a value below that is always a bug, never a good
   unwrap.  Checked on every corpus mesh and on randomised nonsense.
2. **Against hand-computed analytic cases.**  An isometric unwrap must score
   exactly 1.0; a 2x1 rectangle squashed into a unit UV square must score
   exactly ``sqrt(2.5/2) = 1.118034``.  No implementation detail can hide from
   a number worked out on paper.
3. **Against an independent naive oracle.**  ``tests/reference.py`` implements
   the same formulae with plain Python loops.  The fast path and the slow path
   must agree to floating-point tolerance on real Blender meshes.

Plus the hard operational requirement from ``docs/ARCHITECTURE.md`` section 4:
``evaluate`` must never raise, on anything.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tests import reference as ref
from tests.conftest import (
    CORPUS_NAMES,
    all_pathological,
    load_corpus,
    make_all_identical,
    make_empty,
    make_flat_grid,
    make_nan_uvs,
    make_overlapping_pair,
    make_random_mesh,
    make_unit_triangle,
    make_zero_area_mix,
)

#: Below this the mesh-wide L2 is a bug, not a rounding artefact.  The invariant
#: is exact; the tolerance only absorbs float64 summation noise.
L2_FLOOR = 1.0 - 1e-9


# ---------------------------------------------------------------------------
# 1. The provable invariant.
# ---------------------------------------------------------------------------


def _usable_tris(sample) -> int:
    """Return how many triangles can contribute to a distortion mean."""
    from unwrap_variations.core.meshdata import degenerate_mask

    if sample.n_tris == 0:
        return 0
    return int((~degenerate_mask(sample)).sum())


def _assert_neutral_stretch(value: float, label: str) -> None:
    """Assert a mesh with nothing measurable reports a *neutral* stretch.

    Two conventions are defensible when no triangle survives the degeneracy
    filter: ``0.0`` as a "not measured" sentinel, or ``1.0`` as the neutral
    isometric reading (which is what ``UVMetrics``' field defaults use).  What
    is **not** acceptable is anything strictly between them: a value like 0.3
    would make a completely collapsed candidate look better than a flawless one
    and float straight to the top of the ranking.
    """
    assert value in (0.0, 1.0), (
        "%s: stretch_l2 is %.12f with no measurable triangles. Expected the "
        "0.0 sentinel or the neutral 1.0; anything in between out-ranks a "
        "perfect unwrap." % (label, value)
    )


def test_l2_invariant_on_corpus(evaluator, corpus_sample):
    """Mesh-wide normalised L2 stretch is provably >= 1.0 on every real mesh.

    Per triangle AM-GM gives ``L2(T)^2 = (G^2+g^2)/2 >= G*g = A3d/Auv``; summing
    area-weighted and applying Cauchy-Schwarz gives
    ``sum(A3d^2/Auv) >= (sum A3d)^2 / sum Auv = sum A3d``.  Hence ``L2 >= 1``.

    The classic way to break this is to compute the normalisation constant
    ``k = sqrt(sum A3d / sum Auv)`` over *all* triangles while accumulating the
    stretch sums over only the non-degenerate subset.  The two triangle sets
    must be identical.
    """
    m = evaluator.evaluate(corpus_sample, compute_overlap=False)
    assert m.stretch_l2 >= L2_FLOOR, (
        "L2 %.12f < 1.0 on %s -- the normalisation constant and the accumulated "
        "sums are being computed over different triangle subsets"
        % (m.stretch_l2, corpus_sample.name)
    )
    assert m.stretch_linf >= m.stretch_l2 - 1e-9, (
        "Linf (%.12f) must be at least L2 (%.12f): the worst triangle cannot be "
        "better than the area-weighted mean" % (m.stretch_linf, m.stretch_l2)
    )


@pytest.mark.parametrize("mode", ["chaotic", "jittered", "spiky"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_l2_invariant_on_random_meshes(evaluator, mode, seed):
    """The L2 >= 1.0 invariant holds on randomised nonsense too.

    Random UVs on random triangles are not a "good" unwrap by any measure, but
    the invariant is a theorem, not a quality heuristic -- it must survive
    flipped triangles, collapsed triangles and eight orders of magnitude of
    scale variation.
    """
    sample = make_random_mesh(seed, mode=mode)
    m = evaluator.evaluate(sample, compute_overlap=False)
    if _usable_tris(sample) == 0:
        _assert_neutral_stretch(m.stretch_l2, "%s/%d" % (mode, seed))
        return
    assert m.stretch_l2 >= L2_FLOOR, "L2 %.12f < 1.0 on %s/%d" % (
        m.stretch_l2,
        mode,
        seed,
    )


def test_l2_invariant_survives_uv_scaling(evaluator):
    """A global UV scale is normalised away, so it cannot push L2 below 1.0."""
    sample = make_random_mesh(3, mode="jittered")
    base = evaluator.evaluate(sample, compute_overlap=False).stretch_l2
    assert base > 1.0
    for scale in (1e-3, 0.5, 2.0, 1e3, 1e6):
        scaled = sample.with_uvs(sample.uvs * scale)
        got = evaluator.evaluate(scaled, compute_overlap=False).stretch_l2
        assert got >= L2_FLOOR
        assert got == pytest.approx(base, rel=1e-6), (
            "UV scale %g changed L2 from %.9f to %.9f; the metric is supposed to "
            "be normalised and therefore scale-free" % (scale, base, got)
        )


@pytest.mark.parametrize("uv_scale", [1e-6, 1e-3, 1e3, 1e6])
def test_extreme_uv_scaling_does_not_fake_degeneracy(evaluator, uv_scale):
    """Shrinking a whole UV layout does not turn it into a "degenerate" mesh.

    The degeneracy floor is *relative* to the mesh's own mean triangle area, not
    an absolute 1e-12.  That matters for real work: a 0.1 mm CAD or
    photogrammetry part has per-triangle areas around 5e-15, so an absolute
    floor would report a perfectly good unwrap of it as 100 % degenerate and
    score it at zero.  Scaling the UVs by a million in either direction must
    change nothing at all.
    """
    sample = make_random_mesh(3, mode="jittered")
    base = evaluator.evaluate(sample, compute_overlap=False)
    scaled = sample.with_uvs(sample.uvs * uv_scale)
    got = evaluator.evaluate(scaled, compute_overlap=False)
    assert got.degenerate_tris == base.degenerate_tris == 0, (
        "UV scale %g reported %d degenerate triangles against %d at scale 1"
        % (uv_scale, got.degenerate_tris, base.degenerate_tris)
    )
    assert got.stretch_l2 == pytest.approx(base.stretch_l2, rel=1e-6)
    # And the naive oracle, which models the same relative floor, agrees.
    want = ref.reference_metrics(scaled, do_overlap=False)
    assert want["degenerate_tris"] == 0
    assert got.stretch_l2 == pytest.approx(want["stretch_l2"], rel=1e-8)


def test_a_sub_millimetre_model_is_not_reported_as_degenerate(evaluator):
    """A tiny-but-valid model measures exactly like the same model scaled up.

    The regression this guards: with an absolute 1e-12 area floor, every
    triangle of a 0.1 mm part falls below it, the whole mesh is classified
    degenerate, and a flawless unwrap scores zero.
    """
    normal = make_flat_grid(8, size=1.0)
    tiny = dataclasses.replace(normal, verts=normal.verts * 1e-4)
    tiny = tiny.with_uvs(normal.uvs * 1e-4)

    a = evaluator.evaluate(normal, compute_overlap=False)
    b = evaluator.evaluate(tiny, compute_overlap=False)
    assert b.degenerate_tris == 0, (
        "a 0.1 mm model reported %d of %d triangles degenerate"
        % (b.degenerate_tris, tiny.n_tris)
    )
    assert b.stretch_l2 == pytest.approx(a.stretch_l2, rel=1e-9)
    assert b.stretch_l2 == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Analytic cases.
# ---------------------------------------------------------------------------


def test_reference_analytic_selftest():
    """The oracle itself matches three hand-computed cases before we trust it."""
    # Isometric.
    big, small = ref.stretch_singular_values(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0), (1, 0), (0, 1)]
    )
    assert big == pytest.approx(1.0, abs=1e-12)
    assert small == pytest.approx(1.0, abs=1e-12)

    # 3D is 2x1, UV is 1x1 -> raw singular values 2 and 1.
    big, small = ref.stretch_singular_values(
        [(0, 0, 0), (2, 0, 0), (0, 1, 0)], [(0, 0), (1, 0), (0, 1)]
    )
    assert big == pytest.approx(2.0, abs=1e-12)
    assert small == pytest.approx(1.0, abs=1e-12)
    k = math.sqrt(1.0 / 0.5)  # 3D area 1.0, UV area 0.5
    assert big / k == pytest.approx(math.sqrt(2.0), abs=1e-9)
    assert small / k == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-9)
    assert (big / k) * (small / k) == pytest.approx(1.0, abs=1e-12)
    l2 = math.sqrt(((big / k) ** 2 + (small / k) ** 2) / 2.0)
    assert l2 == pytest.approx(1.118034, abs=1e-6)

    # A uniform UV scale cancels completely.
    big, small = ref.stretch_singular_values(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0), (7, 0), (0, 7)]
    )
    k = math.sqrt(0.5 / 24.5)
    assert math.sqrt(((big / k) ** 2 + (small / k) ** 2) / 2.0) == pytest.approx(1.0, abs=1e-12)

    # A mirrored triangle has negative signed UV area.
    assert ref.tri_area_uv_signed((0, 0), (0, 1), (1, 0)) < 0


@pytest.mark.parametrize("uv_scale", [1.0, 0.25, 7.0, 1000.0])
def test_isometric_unwrap_scores_exactly_one(evaluator, uv_scale):
    """An isometric unwrap scores L2 == Linf == 1.0, at any global UV scale."""
    sample = make_flat_grid(16, uv_scale=uv_scale)
    m = evaluator.evaluate(sample, compute_overlap=False)
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-9)
    assert m.stretch_linf == pytest.approx(1.0, abs=1e-9)
    assert m.angle_distortion == pytest.approx(0.0, abs=1e-9)
    assert m.area_distortion == pytest.approx(0.0, abs=1e-9)
    assert m.flipped_tris == 0
    assert m.degenerate_tris == 0


def test_isometric_cube_corpus_scores_exactly_one(evaluator):
    """``cube_project`` is a real Blender isometric projection: exactly 1.0."""
    m = evaluator.evaluate(load_corpus("cube_project"), compute_overlap=False)
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-9)
    assert m.stretch_linf == pytest.approx(1.0, abs=1e-9)


def test_rect_to_square_analytic(evaluator, rect_to_square):
    """A 2x1 rectangle in a 1x1 UV square: L2 == 1.118034, Linf == sqrt(2).

    Worked out by hand: the map squashes by 2 along one axis, so after the area
    normalisation the singular values are ``sqrt(2)`` and ``1/sqrt(2)`` (product
    1, area preserved) and ``L2 = sqrt((2 + 0.5)/2)``.
    """
    m = evaluator.evaluate(rect_to_square, compute_overlap=False)
    assert m.stretch_l2 == pytest.approx(1.118034, abs=1e-6)
    assert m.stretch_linf == pytest.approx(math.sqrt(2.0), abs=1e-9)
    # Area is preserved everywhere, only shape is distorted.
    assert m.area_distortion == pytest.approx(0.0, abs=1e-9)
    assert m.flipped_tris == 0


def test_uniform_uv_scale_is_normalised_away(evaluator):
    """Scaling every UV by 7 changes nothing: normalisation cancels it entirely."""
    base = make_flat_grid(8, uv_scale=1.0)
    scaled = base.with_uvs(base.uvs * 7.0)
    a = evaluator.evaluate(base, compute_overlap=False)
    b = evaluator.evaluate(scaled, compute_overlap=False)
    assert b.stretch_l2 == pytest.approx(1.0, abs=1e-9)
    assert b.stretch_l2 == pytest.approx(a.stretch_l2, rel=1e-12)
    assert b.stretch_linf == pytest.approx(a.stretch_linf, rel=1e-12)


def test_mesh_scale_invariance(evaluator):
    """Scaling the 3D mesh by 1000 leaves every scale-free metric untouched."""
    base = load_corpus("hardsurface")
    big = dataclasses.replace(base, verts=base.verts * 1000.0)
    a = evaluator.evaluate(base, compute_overlap=False)
    b = evaluator.evaluate(big, compute_overlap=False)
    assert b.stretch_l2 == pytest.approx(a.stretch_l2, rel=1e-9)
    assert b.stretch_linf == pytest.approx(a.stretch_linf, rel=1e-9)
    assert b.angle_distortion == pytest.approx(a.angle_distortion, abs=1e-9)
    assert b.area_distortion == pytest.approx(a.area_distortion, rel=1e-9)
    assert b.seam_ratio == pytest.approx(a.seam_ratio, rel=1e-9)


def test_bad_unwrap_is_flagged_loudly(evaluator):
    """``sphere_planar`` is the control: a planar projection of a sphere.

    Exactly half of its 2208 triangles are mirrored and it self-overlaps
    heavily.  A metric suite that does not scream at this is not measuring
    anything.
    """
    good = evaluator.evaluate(load_corpus("sphere_smart"), compute_overlap=True)
    bad = evaluator.evaluate(load_corpus("sphere_planar"), compute_overlap=True)
    assert bad.flipped_tris == 1104
    assert good.flipped_tris == 0
    assert bad.stretch_l2 > 2.0 > good.stretch_l2
    assert bad.overlap_area > 0.0
    assert good.overlap_area == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Fast vs naive.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_fast_matches_naive_reference(evaluator, name):
    """The vectorised evaluator must agree with the naive oracle.

    ``tests/reference.py`` computes the same quantities with plain Python loops
    and the textbook formulae.  Any disagreement beyond float noise means one of
    them is wrong, and the loop version is the one that was validated against
    hand-computed cases.
    """
    sample = load_corpus(name)
    want = ref.reference_metrics(sample, do_overlap=False)
    got = evaluator.evaluate(sample, compute_overlap=False)

    assert got.n_tris == want["tris"]
    assert got.stretch_l2 == pytest.approx(want["stretch_l2"], rel=1e-9), (
        "%s: stretch_l2 fast=%.12f naive=%.12f" % (name, got.stretch_l2, want["stretch_l2"])
    )
    assert got.stretch_linf == pytest.approx(want["stretch_linf"], rel=1e-9), (
        "%s: stretch_linf fast=%.12f naive=%.12f"
        % (name, got.stretch_linf, want["stretch_linf"])
    )
    assert got.flipped_tris == want["flipped_usable_tris"], (
        "%s: flipped fast=%d naive=%d" % (name, got.flipped_tris, want["flipped_usable_tris"])
    )
    assert got.degenerate_tris == want["degenerate_tris"], (
        "%s: degenerate fast=%d naive=%d" % (name, got.degenerate_tris, want["degenerate_tris"])
    )
    assert got.island_count == want["island_count"], (
        "%s: islands fast=%d naive=%d" % (name, got.island_count, want["island_count"])
    )
    assert got.seam_length == pytest.approx(want["seam_length"], rel=1e-9, abs=1e-12)
    assert got.angle_distortion == pytest.approx(want["angle_distortion"], rel=1e-6, abs=1e-9), (
        "%s: angle_distortion fast=%.9f naive=%.9f (spec: area-weighted mean of "
        "the absolute per-corner angle deviation, in degrees)"
        % (name, got.angle_distortion, want["angle_distortion"])
    )
    _assert_matches_one_weighting(
        "area_distortion",
        name,
        got.area_distortion,
        want["area_distortion"],
        want["area_distortion_unweighted"],
    )


def _assert_matches_one_weighting(field, name, got, weighted, unweighted):
    """Assert a mean matches the oracle under one of its two legal weightings.

    ``docs/ARCHITECTURE.md`` pins ``angle_distortion`` as *area-weighted* but
    leaves ``area_distortion`` and ``conformal_error`` as an unqualified
    "mean".  Both readings are defensible, so rather than loosen the tolerance
    into uselessness this pins the value to one of exactly two hand-computable
    numbers.  A third number is a bug.
    """
    if got == pytest.approx(weighted, rel=1e-6, abs=1e-9):
        return
    if got == pytest.approx(unweighted, rel=1e-6, abs=1e-9):
        return
    raise AssertionError(
        "%s: %s fast=%.9f matches neither the area-weighted oracle (%.9f) nor "
        "the unweighted one (%.9f)" % (name, field, got, weighted, unweighted)
    )


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_conformal_error_matches_reference(evaluator, name):
    """Conformal error matches the oracle and is never negative.

    ``sigma_max/sigma_min - 1`` is zero for a perfectly conformal map and grows
    without bound as triangles shear, so it is bounded below by 0.
    """
    sample = load_corpus(name)
    want = ref.reference_metrics(sample, do_overlap=False)
    got = evaluator.evaluate(sample, compute_overlap=False).conformal_error
    assert got >= -1e-12
    assert math.isfinite(got)
    _assert_matches_one_weighting(
        "conformal_error",
        name,
        got,
        want["conformal_error"],
        want["conformal_error_unweighted"],
    )


@pytest.mark.parametrize("mode", ["jittered", "chaotic"])
@pytest.mark.parametrize("seed", [21, 22, 23])
def test_fast_matches_naive_on_random_meshes(evaluator, mode, seed):
    """Fast/naive agreement is not a corpus artefact: it holds on random input too."""
    sample = make_random_mesh(seed, n_tris=40, mode=mode)
    want = ref.reference_metrics(sample, do_overlap=False)
    got = evaluator.evaluate(sample, compute_overlap=False)
    assert got.stretch_l2 == pytest.approx(want["stretch_l2"], rel=1e-8)
    assert got.stretch_linf == pytest.approx(want["stretch_linf"], rel=1e-8)
    assert got.degenerate_tris == want["degenerate_tris"]
    # A triangle with no UV area is not meaningfully "flipped", so counting
    # mirror flips over all triangles or over only the measurable ones are both
    # defensible.  They coincide on every real mesh (no corpus mesh has a
    # degenerate triangle); only random junk can tell them apart.
    assert got.flipped_tris in (want["flipped_tris"], want["flipped_usable_tris"]), (
        "flipped fast=%d, naive over all tris=%d, naive over usable tris=%d"
        % (got.flipped_tris, want["flipped_tris"], want["flipped_usable_tris"])
    )


def test_deterministic(evaluator):
    """Same input, same floats -- twice, bit for bit."""
    sample = load_corpus("sphere_smart")
    a = evaluator.evaluate(sample, compute_overlap=True)
    b = evaluator.evaluate(sample, compute_overlap=True)
    for field in (
        "stretch_l2",
        "stretch_linf",
        "angle_distortion",
        "area_distortion",
        "conformal_error",
        "overlap_area",
        "packing_efficiency",
        "texel_density_cv",
    ):
        assert getattr(a, field) == getattr(b, field), "%s is not deterministic" % field
    assert a.island_count == b.island_count
    assert a.flipped_tris == b.flipped_tris


# ---------------------------------------------------------------------------
# 4. Overlap correctness.
# ---------------------------------------------------------------------------


def test_edge_adjacent_triangles_report_zero_overlap(evaluator):
    """Two triangles sharing an edge overlap in a set of measure zero.

    This is the single most important overlap test in the suite.  A
    separating-axis boolean answers "yes, overlapping" for every edge-adjacent
    pair; on a 2048-triangle flat grid that produced 8773 false positives before
    the reference was fixed to measure intersection *area*.
    """
    sample = make_overlapping_pair("adjacent")
    m = evaluator.evaluate(sample, compute_overlap=True)
    assert m.overlap_area == pytest.approx(0.0, abs=1e-12), (
        "edge-adjacent triangles reported %g of overlap; overlap must be "
        "measured as intersection AREA, not a separating-axis boolean"
        % m.overlap_area
    )


def test_flat_grid_has_zero_overlap(evaluator, flat_grid):
    """A 2048-triangle isometric grid is perfectly packed: zero overlap.

    Every interior triangle here is edge-adjacent to three neighbours and
    vertex-adjacent to several more.  The correct answer is exactly 0.0.
    """
    m = evaluator.evaluate(flat_grid, compute_overlap=True)
    assert m.overlap_area == pytest.approx(0.0, abs=1e-9), (
        "flat grid reported %g of self-overlap; the true answer is 0 and every "
        "reported unit is an edge-adjacency false positive" % m.overlap_area
    )


def test_partial_overlap_is_detected(evaluator):
    """Deliberately overlapping triangles are found, with the right area."""
    sample = make_overlapping_pair("partial")
    m = evaluator.evaluate(sample, compute_overlap=True)
    assert m.overlap_area > 1e-6, "a 0.25-unit triangle overlap went undetected"
    assert m.overlap_area == pytest.approx(0.25, rel=0.02)


def test_containment_is_detected(evaluator):
    """One triangle wholly inside another is an overlap of the inner one's area.

    Containment is the case an edge-intersection test misses entirely: no two
    edges cross, yet the triangles overlap completely.
    """
    sample = make_overlapping_pair("contained")
    m = evaluator.evaluate(sample, compute_overlap=True)
    assert m.overlap_area > 1e-6, (
        "a triangle wholly contained in another was reported as non-overlapping; "
        "the test is probably looking for edge crossings only"
    )
    assert m.overlap_area == pytest.approx(0.02, rel=0.05)


def test_identical_triangles_overlap_fully(evaluator):
    """Two coincident triangles overlap by their whole area."""
    sample = make_overlapping_pair("identical")
    m = evaluator.evaluate(sample, compute_overlap=True)
    assert m.overlap_area == pytest.approx(0.5, rel=0.05)


@pytest.mark.parametrize(
    "name", ["cube_project", "nonmanifold", "hardsurface", "sphere_smart", "sphere_planar"]
)
def test_overlap_matches_reference(evaluator, name):
    """Overlap area agrees with the exact polygon-clipping oracle.

    Both implementations sum the intersection area over triangle **pairs**, so
    the numbers must match exactly -- except that the evaluator clamps its total
    to the used UV area, a documented decision that keeps the figure
    interpretable when many charts stack on the same spot.  ``cube_project`` is
    the one corpus mesh where the clamp bites: all six faces project into the
    same unit square, so the raw pairwise sum is 15.0 against 6.0 of actual UV
    area.

    The oracle is O(T^2) with an exact Sutherland-Hodgman clip per candidate
    pair, so this is only tractable on the corpus, not on a 500k-triangle mesh.
    """
    sample = load_corpus(name)
    want = ref.reference_metrics(sample, do_overlap=True, overlap_limit=5000)
    got = evaluator.evaluate(sample, compute_overlap=True)
    assert want["overlap_area"] is not None

    expected = min(want["overlap_area"], want["uv_area_sum"])
    if expected == 0.0:
        assert got.overlap_area == pytest.approx(0.0, abs=1e-12), (
            "%s: fast=%g naive=0.0 -- false-positive overlap, most likely from "
            "treating edge adjacency as intersection" % (name, got.overlap_area)
        )
    else:
        assert got.overlap_area > 0.0, (
            "%s: fast=0.0 naive=%g -- missed real overlap" % (name, want["overlap_area"])
        )
        assert got.overlap_area == pytest.approx(expected, rel=1e-6), (
            "%s: overlap_area fast=%.9f expected=%.9f (naive pairwise sum %.9f "
            "clamped to the %.9f of used UV area)"
            % (name, got.overlap_area, expected, want["overlap_area"], want["uv_area_sum"])
        )


def test_compute_overlap_false_skips_the_work(evaluator):
    """``compute_overlap=False`` reports zero and never raises."""
    m = evaluator.evaluate(load_corpus("sphere_planar"), compute_overlap=False)
    assert m.overlap_area == 0.0
    # Everything else is still computed.
    assert m.n_tris == 2208
    assert m.flipped_tris == 1104


def test_overlap_is_not_quadratic(evaluator):
    """Overlap detection scales sub-quadratically, i.e. it uses a spatial grid.

    A naive O(T^2) pass on 8192 triangles is 33.5 million pair tests; a uniform
    grid is linear in practice.  Quadrupling the triangle count must not
    quadruple-squared the time.  The bound is deliberately loose -- this is
    catching an algorithmic mistake, not benchmarking.
    """
    import time

    small = make_flat_grid(32)  # 2048 tris
    large = make_flat_grid(64)  # 8192 tris

    evaluator.evaluate(small, compute_overlap=True)  # warm any lazy import
    t0 = time.perf_counter()
    evaluator.evaluate(small, compute_overlap=True)
    t_small = time.perf_counter() - t0
    t0 = time.perf_counter()
    evaluator.evaluate(large, compute_overlap=True)
    t_large = time.perf_counter() - t0

    # 4x the triangles. O(T^2) would be ~16x; allow a very generous 10x.
    assert t_large < max(t_small * 10.0, 5.0), (
        "overlap detection took %.3fs on 8192 tris vs %.3fs on 2048 -- that "
        "growth looks quadratic; docs/ARCHITECTURE.md section 4 requires a "
        "uniform grid / spatial hash" % (t_large, t_small)
    )


# ---------------------------------------------------------------------------
# 5. Degenerate-input safety.
# ---------------------------------------------------------------------------


def _assert_sane(m, label: str) -> None:
    """Assert every float on a metrics record is finite and sensibly signed."""
    for field in (
        "angle_distortion",
        "angle_distortion_max",
        "area_distortion",
        "stretch_l2",
        "stretch_linf",
        "conformal_error",
        "seam_length",
        "seam_ratio",
        "packing_efficiency",
        "uv_coverage",
        "overlap_area",
        "texel_density_cv",
        "eval_seconds",
    ):
        value = float(getattr(m, field))
        assert math.isfinite(value), "%s: %s is %r, which will poison ranking" % (
            label,
            field,
            value,
        )
        assert value >= -1e-12, "%s: %s is negative (%g)" % (label, field, value)
    for field in ("island_count", "flipped_tris", "degenerate_tris", "n_tris",
                  "out_of_bounds_tris"):
        value = getattr(m, field)
        assert int(value) >= 0, "%s: %s is negative (%r)" % (label, field, value)
    assert len(m.uv_bounds) == 4
    assert all(math.isfinite(float(v)) for v in m.uv_bounds), (
        "%s: uv_bounds contains a non-finite value %r" % (label, m.uv_bounds)
    )


@pytest.mark.parametrize("compute_overlap", [True, False])
def test_pathological_meshes_never_raise(evaluator, compute_overlap):
    """``evaluate`` must survive every kind of broken input.

    ``docs/ARCHITECTURE.md`` section 4 is unambiguous: never raises on degenerate
    input.  A traceback escaping here becomes a traceback inside a Blender
    operator, and Blender operators that raise leave the mesh half-written.
    """
    for sample in all_pathological():
        m = evaluator.evaluate(sample, compute_overlap=compute_overlap)
        _assert_sane(m, sample.name)


def test_evaluate_emits_no_numpy_warnings(evaluator):
    """A clean run leaves nothing in the console, even on broken geometry.

    Regression test.  The vectorised path used to emit numpy RuntimeWarnings on
    degenerate input -- "overflow encountered in divide" from the
    conformal-error ratio, and "invalid value encountered in minimum/maximum"
    from the island-bounds reduction when a UV is NaN.  The values were always
    finite, so this was noise rather than wrongness -- but it was noise printed
    into Blender's console on exactly the mesh the user is already confused
    about.  Now contained by ``np.errstate`` inside the evaluator.
    """
    import warnings

    samples = all_pathological() + [
        make_random_mesh(seed, mode=mode)
        for mode in ("spiky", "chaotic", "jittered")
        for seed in range(8)
    ]
    for sample in samples:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            evaluator.evaluate(sample, compute_overlap=True)
        runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime, "%s produced %s" % (
            sample.name,
            [str(w.message) for w in runtime],
        )


def test_empty_mesh(evaluator):
    """A mesh with no triangles evaluates to a zeroed record, not an exception."""
    m = evaluator.evaluate(make_empty(), compute_overlap=True)
    assert m.n_tris == 0
    _assert_neutral_stretch(m.stretch_l2, "empty")
    assert m.island_count == 0
    assert m.flipped_tris == 0
    _assert_sane(m, "empty")


def test_single_triangle(evaluator):
    """One isometric triangle is a valid mesh and scores exactly 1.0."""
    m = evaluator.evaluate(make_unit_triangle(), compute_overlap=True)
    assert m.n_tris == 1
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-9)
    assert m.island_count == 1
    assert m.overlap_area == pytest.approx(0.0, abs=1e-15)


def test_all_identical_vertices(evaluator):
    """Every triangle collapsed to a point: all degenerate, no crash, no NaN."""
    sample = make_all_identical(6)
    m = evaluator.evaluate(sample, compute_overlap=True)
    assert m.degenerate_tris == sample.n_tris
    _assert_neutral_stretch(m.stretch_l2, "all_identical")
    _assert_sane(m, "all_identical")


def test_zero_area_triangles_are_excluded_not_fatal(evaluator):
    """Degenerate triangles are counted and excluded from the means.

    ``make_zero_area_mix`` has one good triangle, one collinear (zero 3D area)
    and one with a collapsed UV triangle.  The good one is isometric, so the
    reported stretch must be exactly 1.0 -- proof the bad two were excluded from
    the sums *and* from the normalisation constant.
    """
    m = evaluator.evaluate(make_zero_area_mix(), compute_overlap=True)
    assert m.n_tris == 3
    assert m.degenerate_tris == 2
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-9)
    _assert_sane(m, "zero_area_mix")


def test_nan_and_inf_uvs(evaluator):
    """NaN and Inf UVs are contained, never propagated into the metrics."""
    m = evaluator.evaluate(make_nan_uvs(), compute_overlap=True)
    assert m.degenerate_tris > 0
    _assert_sane(m, "nan_uvs")
    assert m.stretch_l2 >= L2_FLOOR


def test_mesh_with_no_uv_layer(evaluator):
    """A mesh whose UVs are all zeros (no UV layer) evaluates cleanly.

    ``core.bridge.extract_sample`` returns zeroed UVs when the mesh has no UV
    layer at all, so this is a real input, not a hypothetical one.
    """
    grid = make_flat_grid(4)
    sample = grid.with_uvs(np.zeros_like(grid.uvs))
    m = evaluator.evaluate(sample, compute_overlap=True)
    assert m.degenerate_tris == m.n_tris
    _assert_sane(m, "no_uv_layer")


def test_huge_uv_magnitudes(evaluator):
    """UVs far outside the unit square do not overflow or produce NaN."""
    grid = make_flat_grid(6)
    sample = grid.with_uvs(grid.uvs * 1e8 + 1e9)
    m = evaluator.evaluate(sample, compute_overlap=True)
    _assert_sane(m, "huge_uvs")
    assert m.stretch_l2 == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 6. Layout metrics.
# ---------------------------------------------------------------------------


def test_uv_bounds(evaluator):
    """``uv_bounds`` is ``(min_u, min_v, max_u, max_v)`` of the UV layer."""
    grid = make_flat_grid(4)
    uvs = np.array(grid.uvs)
    uvs = uvs * 0.5 + np.array([0.25, 0.1])
    m = evaluator.evaluate(grid.with_uvs(uvs), compute_overlap=False)
    umin, vmin, umax, vmax = m.uv_bounds
    assert umin == pytest.approx(0.25, abs=1e-9)
    assert vmin == pytest.approx(0.1, abs=1e-9)
    assert umax == pytest.approx(0.75, abs=1e-9)
    assert vmax == pytest.approx(0.6, abs=1e-9)


def test_out_of_bounds_tris(evaluator):
    """Triangles outside the 0..1 UDIM tile are counted."""
    grid = make_flat_grid(4)
    inside = evaluator.evaluate(grid, compute_overlap=False)
    assert inside.out_of_bounds_tris == 0
    shifted = grid.with_uvs(grid.uvs + 5.0)
    outside = evaluator.evaluate(shifted, compute_overlap=False)
    assert outside.out_of_bounds_tris == grid.n_tris


def test_packing_efficiency_bounds(evaluator, corpus_sample):
    """Packing efficiency is a ratio in 0..1 on every real mesh."""
    m = evaluator.evaluate(corpus_sample, compute_overlap=False)
    assert 0.0 <= m.packing_efficiency <= 1.0 + 1e-9, (
        "%s: packing_efficiency %g is outside 0..1" % (corpus_sample.name, m.packing_efficiency)
    )


def test_packing_efficiency_of_a_full_square(evaluator):
    """A grid filling its own bounding box packs at ~100 %."""
    m = evaluator.evaluate(make_flat_grid(8), compute_overlap=False)
    assert m.packing_efficiency == pytest.approx(1.0, abs=1e-6)


def test_texel_density_cv_is_zero_for_uniform_maps(evaluator):
    """An isometric map has identical texel density everywhere: CV == 0."""
    m = evaluator.evaluate(make_flat_grid(8), compute_overlap=False)
    assert m.texel_density_cv == pytest.approx(0.0, abs=1e-9)


def test_texel_density_cv_is_positive_for_uneven_maps(evaluator):
    """A bad unwrap has wildly varying texel density."""
    m = evaluator.evaluate(load_corpus("sphere_planar"), compute_overlap=False)
    assert m.texel_density_cv > 0.05


def test_seam_ratio_is_scale_free(evaluator):
    """``seam_ratio`` divides by sqrt(total 3D area), so mesh scale cancels."""
    base = load_corpus("hardsurface")
    seams = np.zeros(base.n_edges, dtype=bool)
    seams[::7] = True
    base = base.with_seams(seams)
    scaled = dataclasses.replace(base, verts=base.verts * 37.0)
    a = evaluator.evaluate(base, compute_overlap=False)
    b = evaluator.evaluate(scaled, compute_overlap=False)
    assert a.seam_ratio > 0.0
    assert b.seam_ratio == pytest.approx(a.seam_ratio, rel=1e-9)
    assert b.seam_length == pytest.approx(a.seam_length * 37.0, rel=1e-9)


def test_island_count_matches_reference(evaluator, corpus_sample):
    """Island counting welds loops on ``(vertex, uv)``, same as the oracle."""
    want = ref.reference_metrics(corpus_sample, do_overlap=False)["island_count"]
    got = evaluator.evaluate(corpus_sample, compute_overlap=False).island_count
    assert got == want, "%s: islands fast=%d naive=%d" % (corpus_sample.name, got, want)


def test_eval_seconds_is_recorded(evaluator):
    """``eval_seconds`` is populated and plausible."""
    m = evaluator.evaluate(load_corpus("hardsurface"), compute_overlap=False)
    assert m.eval_seconds >= 0.0
    assert m.eval_seconds < 60.0


# ---------------------------------------------------------------------------
# 7. Performance budget.
# ---------------------------------------------------------------------------


@pytest.mark.perf
@pytest.mark.slow
def test_large_mesh_performance(evaluator):
    """500k triangles in well under a second, excluding overlap.

    ``docs/ARCHITECTURE.md`` section 4 sets a 250 ms budget.  The assertion here
    is deliberately looser (1.5 s) because CI runners vary by several times;
    what it actually catches is a Python-level loop having crept into the hot
    path, which costs two orders of magnitude, not twenty percent.
    """
    import time

    sample = make_flat_grid(500)  # 500,000 triangles
    assert sample.n_tris == 500_000
    evaluator.evaluate(make_flat_grid(8), compute_overlap=False)  # warm up
    t0 = time.perf_counter()
    m = evaluator.evaluate(sample, compute_overlap=False)
    elapsed = time.perf_counter() - t0
    print("\n500k triangles evaluated in %.3f s" % elapsed)
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-6)
    assert elapsed < 1.5, (
        "500k triangles took %.3f s; the 250 ms budget in "
        "docs/ARCHITECTURE.md section 4 implies a fully vectorised path" % elapsed
    )


def test_overlap_saturates_rather_than_reporting_clean_when_the_hash_gives_up(evaluator):
    """A layout too dense to index must not be reported as overlap-free.

    ``_overlap_area`` builds a uniform spatial hash and doubles the cell size up
    to eight times to keep the insert count bounded.  It used to ``return 0.0``
    when that budget was exhausted -- which says "perfectly clean" about the one
    layout shape most likely to be catastrophically dirty, because exhausting
    the budget means triangles are stacked on top of each other.

    That was not theoretical.  ``injectivity.certify`` treats
    ``overlap_area == 0.0`` as proof of injectivity, so a stack of coincident
    triangles certified as *globally injective* while carrying an enormous real
    overlap.  Saturating to the used area is the same rule the function already
    applies once accumulated overlap reaches that area, moved to the case where
    the measurement cannot start rather than cannot finish.
    """
    from unwrap_variations.core import evaluator as ev

    # Every triangle on the same spot: real overlap is maximal.
    n = 64
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    uv = np.repeat(tri[None, :, :], n, axis=0)
    lo = uv.min(axis=1)
    hi = uv.max(axis=1)
    total_uv_area = 0.5 * n

    original = ev._OVERLAP_MAX_INSERTS
    try:
        # Force the doubling loop to exhaust, which is what a genuinely dense
        # layout does on the real budget.
        ev._OVERLAP_MAX_INSERTS = 1
        saturated = ev._overlap_area(uv, lo, hi, total_uv_area)
    finally:
        ev._OVERLAP_MAX_INSERTS = original

    assert saturated == pytest.approx(total_uv_area), (
        "an unindexable layout reported %r; 0.0 here is a false clean bill of "
        "health that injectivity.certify believes" % saturated
    )
    assert saturated > 0.0


def test_organic_baseline_fixture_still_carries_the_contracts_numbers(evaluator):
    """The layout ``docs/ORGANIC.md`` section 0 is measured against is pinned.

    Section 0's figures were originally quoted as "the winner of
    ``engine.generate(count=10, seed=1)``".  That stopped being reproducible the
    moment the ``CONCAVE`` strategy entered the recipe grid, and two separate
    pieces of work had acceptance criteria evaporate underneath them as a
    result.  The layout itself is the thing worth pinning; the procedure that
    happened to produce it is not.

    If this test fails, the fixture changed -- not the engine.  Section 0 and
    every acceptance criterion derived from it are measured against these exact
    numbers, so they move together or not at all.
    """
    from tests.conftest import load_corpus

    sample = load_corpus("suzanne_organic_baseline")
    assert sample.n_tris == 968
    assert sample.n_loops == 1968
    assert sample.n_verts == 507

    m = evaluator.evaluate(sample)
    assert m.island_count == 14
    assert m.flipped_tris == 0
    assert m.degenerate_tris == 0
    assert m.overlap_area == pytest.approx(0.0013420951, rel=1e-6), (
        "ORGANIC.md 0.2 and 0.3 both quote this figure; it is the whole reason "
        "the fold work exists"
    )
    assert m.stretch_l2 == pytest.approx(1.0164, abs=5e-4)
