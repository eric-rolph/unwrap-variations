"""Tests for ``unwrap_variations.core.injectivity`` -- the injectivity certificate.

This module is a *proof engine*, so it is tested like one.  The claim it makes
is not "probably fine": for a disk chart, ``(no flips) and (simple boundary)``
is a theorem, and a wrong answer here would let ``repair_folds`` sign off on a
broken bake.  Hence four independent lines of attack:

1. **Hand-computable answers.**  A square crosses itself 0 times, a bowtie
   once, a pentagram exactly 5.  A spiral strip whose tip stops *inside* its own
   shank crosses exactly twice, and that number is derived on paper in
   :func:`make_spiral_fold`'s docstring, not read off the implementation.
2. **The complementary blindness.**  ``docs/ORGANIC.md`` section 0.3 exists
   because a flip test cannot see Suzanne's defect.  The fan-with-an-everted-
   centre case is the mirror image -- flips with a perfectly simple boundary --
   so the two tests together pin that neither check subsumes the other.
3. **An independent naive oracle.**  ``tests/reference.py`` re-implements every
   step with dictionaries, Python loops and exact :class:`fractions.Fraction`
   arithmetic, sharing no code with the module under test.  They must agree on
   *which* segment pairs cross, not merely on how many.
4. **Consistency with the evaluator.**  ``overlap_area > 0`` and
   "globally injective" cannot both be true, ever.  The two are measured by
   completely different machinery (convex clipping against boundary
   combinatorics) and the certificate is worthless if they can disagree.

Plus the operational requirements: never raise, be deterministic, and stay
inside the 500 ms / 100k-segment budget from ``docs/ORGANIC.md`` section 4.1.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tests import reference as ref
from tests.conftest import (
    CORPUS_NAMES,
    SMALL_CORPUS_NAMES,
    all_pathological,
    import_or_skip,
    load_corpus,
    make_flat_grid,
    make_overlapping_pair,
    make_random_mesh,
)
from unwrap_variations.core.meshdata import make_sample, tri_areas_uv

#: Corpus meshes whose boundary is small enough for the quadratic oracle.
ORACLE_CORPUS = CORPUS_NAMES

#: Section 4.1's budget: 100k boundary segments in under 500 ms.
PERF_SEGMENTS = 100_000
PERF_BUDGET_S = 0.5


@pytest.fixture(scope="session")
def inj():
    """The injectivity module under test."""
    return import_or_skip("unwrap_variations.core.injectivity")


# ---------------------------------------------------------------------------
# Mesh builders.  Local to this file on purpose: they encode the specific
# defects this module has to distinguish, and each one's expected answer is
# derived in its docstring.
# ---------------------------------------------------------------------------


def _strip_from_rings(uv: np.ndarray, verts: np.ndarray, steps: int, name: str, *, cw: bool):
    """Build a quad strip from interleaved (outer, inner) corner arrays."""
    tris = []
    for i in range(steps):
        outer_a, inner_a = 2 * i, 2 * i + 1
        outer_b, inner_b = 2 * i + 2, 2 * i + 3
        if cw:
            tris.append((outer_a, inner_a, inner_b))
            tris.append((outer_a, inner_b, outer_b))
        else:
            tris.append((outer_a, outer_b, inner_b))
            tris.append((outer_a, inner_b, inner_a))
    tri_verts = np.asarray(tris, dtype=np.int32)
    tri_loops = np.arange(tri_verts.size, dtype=np.int32).reshape(-1, 3)
    loop_verts = tri_verts.reshape(-1)
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uv[loop_verts],
        loop_verts=loop_verts,
        name=name,
    )


def make_spiral_fold(
    steps: int = 120,
    r0: float = 2.0,
    half: float = 0.25,
    extra: float = 0.5,
    *,
    cw: bool = True,
) -> object:
    """Build a flat ribbon whose UV layout laps over itself **exactly twice**.

    This is Suzanne's defect in miniature and the reason this module exists: a
    chart that is locally injective everywhere -- every triangle keeps its
    orientation -- while a distant part of it lies on top of another part.  A
    flip test scores it perfectly clean.

    The ribbon follows the spiral centreline ``P(t) = R(t) * (cos t, sin t)``
    with ``R(t) = r0 - k t`` and half-width ``w``, over ``t in [0, 2 pi + f]``.
    The Jacobian of ``(t, v) -> (R(t) + v w)(cos t, sin t)`` has determinant
    ``-w (R(t) + v w)``, which never changes sign while the radius stays
    positive, so **no triangle can flip** however tightly it winds.

    Choosing ``2 pi k = w`` makes the second lap sit half a width inside the
    first, and the crossing count then follows by hand.  Writing the outer and
    inner boundary radii as ``R(t) + w`` and ``R(t) - w``:

    * the **end cap**, a radial segment at angle ``f``, spans
      ``[R0 - f k - 2w, R0 - f k]``.  Lap one's inner edge at that angle sits at
      ``R0 - f k - w``, the midpoint -- inside the cap, so they cross.  Lap
      one's outer edge sits at ``R0 - f k + w``, above the cap's top -- no
      crossing.  **One crossing**, and the tip stops inside the shank rather
      than punching through it.
    * the **outer edge on lap two** is at ``R(t + 2 pi) + w = R0 - t k``, which
      lies strictly between lap one's inner (``R0 - t k - w``) and outer
      (``R0 - t k + w``) edges for the whole overlap, so it crosses neither.  At
      angle zero it does pass through the **start cap**, whose radial span is
      ``[R0 - w, R0 + w]``.  **One crossing**.
    * the **inner edge on lap two** is at ``R0 - t k - 2w``, below lap one's
      inner edge and below the start cap, so it crosses nothing.

    Total: **exactly 2**.

    Args:
        steps: Quads along the ribbon.
        r0: Starting centreline radius.
        half: Ribbon half-width, ``w`` above.
        extra: Overlap angle ``f``, in radians, past a full turn.
        cw: Wind the triangles so the whole chart is mirrored in UV.  A
            uniformly mirrored chart is still injective, so ``flipped`` must
            stay 0 either way -- which is what makes this the case that pins the
            "opposite the chart majority" rule.

    Returns:
        A :class:`~unwrap_variations.core.meshdata.UVMeshSample`.
    """
    k = half / (2.0 * math.pi)
    total = 2.0 * math.pi + extra
    t = np.linspace(0.0, total, steps + 1)
    radius = r0 - k * t
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    uv = np.empty((2 * (steps + 1), 2), dtype=np.float64)
    uv[0::2, 0] = (radius + half) * cos_t
    uv[0::2, 1] = (radius + half) * sin_t
    uv[1::2, 0] = (radius - half) * cos_t
    uv[1::2, 1] = (radius - half) * sin_t

    # The 3D surface is the ribbon unrolled flat: isometric, so nothing in 3D
    # is degenerate and the defect is unambiguously a UV-space one.
    arc = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(radius * cos_t), np.diff(radius * sin_t)))]
    )
    verts = np.zeros((2 * (steps + 1), 3), dtype=np.float64)
    verts[0::2, 0] = arc
    verts[0::2, 1] = half
    verts[1::2, 0] = arc
    verts[1::2, 1] = -half
    return _strip_from_rings(uv, verts, steps, "spiral_fold", cw=cw)


def make_everted_fan(centre=(1.5, 0.0), n: int = 6) -> object:
    """Build a triangle fan whose centre vertex is dragged outside its own ring.

    The complement of :func:`make_spiral_fold`: the boundary polygon is the
    untouched regular ``n``-gon and is perfectly simple, while the two triangles
    on the far side of the ring invert.  A boundary-only check scores it clean,
    so this is the case that stops ``boundary_crossings`` being mistaken for the
    whole certificate.

    Args:
        centre: UV position of the fan's hub.  Inside the ring gives a clean
            chart; ``(1.5, 0)`` sits outside it and everts exactly two
            triangles.
        n: Ring vertices.

    Returns:
        A :class:`~unwrap_variations.core.meshdata.UVMeshSample`.
    """
    ring = [(math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n)) for i in range(n)]
    uv = np.asarray([tuple(centre), *ring], dtype=np.float64)
    verts = np.asarray([(0.0, 0.0, 0.0), *[(x, y, 0.0) for x, y in ring]], dtype=np.float64)
    tri_verts = np.asarray(
        [(0, i + 1, (i + 1) % n + 1) for i in range(n)], dtype=np.int32
    )
    tri_loops = np.arange(tri_verts.size, dtype=np.int32).reshape(-1, 3)
    loop_verts = tri_verts.reshape(-1)
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uv[loop_verts],
        loop_verts=loop_verts,
        name="everted_fan",
    )


def make_both_defects() -> object:
    """Build a chart carrying a fold **and** an everted triangle at once.

    A spiral fold with one interior UV vertex dragged across its neighbours, so
    ``locally_injective`` and the boundary test must both fail.  A certificate
    that reports only the first defect it finds gets this wrong.
    """
    sample = make_spiral_fold()
    uv = np.array(sample.uvs, dtype=np.float64)
    # Loop 1 is an inner-edge corner of the first quad; pushing it far across
    # the ribbon inverts the triangles that use it without moving any boundary
    # vertex of the far lap.
    target = uv[1]
    uv[sample.tri_loops[0][1]] = target + np.array([0.0, 2.0])
    return sample.with_uvs(uv)


def make_bowtie(nested: bool = False) -> object:
    """Build two triangular lobes joined at a single shared vertex.

    The shape that gets past a naive disk test: Euler characteristic
    ``V - E + F = 5 - 6 + 2 = 1`` and a *single* boundary cycle, yet not a disk,
    because that cycle passes through the shared vertex twice.

    Args:
        nested: When True the second lobe is laid inside the first in UV.  That
            is the dangerous version -- the two boundaries touch only at the
            pinch, so nothing crosses transversally and only the overlap
            measurement and the pinch test see the failure.

    Returns:
        A :class:`~unwrap_variations.core.meshdata.UVMeshSample`.
    """
    verts = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    tri_verts = np.asarray([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    tri_loops = np.arange(6, dtype=np.int32).reshape(-1, 3)
    loop_verts = tri_verts.reshape(-1)
    second = (
        [(0.0, 0.0), (0.5, 0.1), (0.1, 0.5)]
        if nested
        else [(0.0, 0.0), (-1.0, 0.0), (0.0, -1.0)]
    )
    uvs = np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), *second], dtype=np.float64)
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=loop_verts,
        name="bowtie_nested" if nested else "bowtie",
    )


def make_touching_squares(mode: str = "edge") -> tuple[np.ndarray, list[list[int]]]:
    """Return ``(uv, loops)`` for two unit squares that touch but never cross.

    Args:
        mode: ``"edge"`` shares a full edge, ``"corner"`` touches at one point,
            ``"tee"`` puts one square's corner exactly on the other's edge
            midpoint, and ``"collinear"`` overlaps along part of an edge.

    Returns:
        Coordinates and two closed polylines.  Every mode must report **zero**
        crossings: touching is not crossing, which is the convention that keeps
        this module consistent with the evaluator's overlap measurement.
    """
    first = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    if mode == "edge":
        second = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)]
    elif mode == "corner":
        second = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
    elif mode == "tee":
        second = [(1.0, 0.5), (2.0, 0.5), (2.0, 1.5), (1.0, 1.5)]
    elif mode == "collinear":
        second = [(1.0, 0.25), (2.0, 0.25), (2.0, 0.75), (1.0, 0.75)]
    else:
        raise ValueError("unknown mode %r" % mode)
    uv = np.asarray(first + second, dtype=np.float64)
    return uv, [[0, 1, 2, 3], [4, 5, 6, 7]]


def random_polylines(seed: int) -> tuple[np.ndarray, list[list[int]]]:
    """Return random closed polylines, deliberately full of crossings.

    Every third seed places its points on a coarse integer lattice, which
    manufactures exactly-collinear triples, shared positions and T-junctions by
    the hundred -- the configurations a floating-point predicate gets wrong.
    """
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    loops: list[list[int]] = []
    for _ in range(int(rng.integers(1, 4))):
        count = int(rng.integers(3, 18))
        base = sum(len(loop) for loop in loops)
        if seed % 3 == 0:
            block = rng.integers(0, 5, size=(count, 2)).astype(np.float64)
        elif seed % 3 == 1:
            block = np.round(rng.uniform(-1.0, 1.0, size=(count, 2)), 1)
        else:
            block = rng.uniform(-1.0, 1.0, size=(count, 2))
        points.append(block)
        loops.append(list(range(base, base + count)))
    return np.concatenate(points), loops


def canonical_loops(loops, uv) -> list:
    """Return loops as rotation- and direction-independent coordinate tuples.

    Two correct boundary extractors may start the same cycle at a different
    vertex, walk it the other way round, or pick a different loop index for the
    same welded UV vertex.  Comparing canonicalised *positions* asks the only
    question that matters: is it the same curve.
    """
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    out = []
    for loop in loops:
        seq = [tuple(points[int(i)]) for i in loop]
        rotations = [tuple(seq[k:] + seq[:k]) for k in range(len(seq))]
        backwards = seq[::-1]
        rotations += [tuple(backwards[k:] + backwards[:k]) for k in range(len(backwards))]
        out.append(min(rotations))
    return sorted(out)


# ---------------------------------------------------------------------------
# 1. Hand-computable crossing counts.
# ---------------------------------------------------------------------------


def test_square_boundary_does_not_cross_itself(inj):
    """The simplest possible simple closed curve reports zero crossings."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert inj.boundary_crossings(uv, [[0, 1, 2, 3]]) == 0


def test_bowtie_crosses_once(inj):
    """A quadrilateral traversed in bowtie order crosses itself exactly once."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert inj.boundary_crossings(uv, [[0, 1, 2, 3]]) == 1


def test_pentagram_crosses_five_times(inj):
    """The star polygon {5/2} has exactly five self-intersections.

    Five vertices, each chord skipping one -- every pair of non-adjacent chords
    crosses once, and there are exactly five such pairs.  The angular offset
    keeps the coordinates off any axis so nothing is accidentally collinear.
    """
    angle = [2.0 * math.pi * k / 5.0 + 0.3 for k in range(5)]
    uv = np.array([[math.cos(a), math.sin(a)] for a in angle])
    assert inj.boundary_crossings(uv, [[0, 2, 4, 1, 3]]) == 5


def test_two_separate_squares_do_not_cross(inj):
    """Disjoint loops in the same call cross zero times."""
    uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [6.0, 6.0],
            [5.0, 6.0],
        ]
    )
    assert inj.boundary_crossings(uv, [[0, 1, 2, 3], [4, 5, 6, 7]]) == 0


def test_square_inside_square_crosses_zero_times(inj):
    """Containment produces **no** boundary crossing -- the documented blind spot.

    This is why ``certify`` does not rest on ``boundary_crossings`` alone: a
    chart nested wholly inside another is a genuine injectivity failure that no
    crossing test can see, so the exact overlap measurement has to carry it.
    """
    uv = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ]
    )
    assert inj.boundary_crossings(uv, [[0, 1, 2, 3], [4, 5, 6, 7]]) == 0


def test_crossing_pairs_identify_the_right_segments(inj):
    """The bowtie's crossing is between its first and third segments."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pairs = inj.crossing_pairs(uv, [[0, 1, 2, 3]])
    assert pairs.shape == (1, 2)
    assert pairs.tolist() == [[1, 3]]


@pytest.mark.parametrize("mode", ["edge", "corner", "tee", "collinear"])
def test_touching_is_not_crossing(inj, mode):
    """Shared edges, shared corners, T-junctions and collinear overlap count zero.

    This is the convention that keeps the certificate consistent with
    ``evaluator._overlap_area``, whose separating-axis test also treats touching
    as separated -- otherwise every pair of adjacent triangles in a mesh would
    read as a defect.
    """
    uv, loops = make_touching_squares(mode)
    assert inj.boundary_crossings(uv, loops) == 0
    assert ref.naive_boundary_crossings(uv, loops) == 0


def test_degenerate_and_repeated_vertices_do_not_cross(inj):
    """Zero-length segments and a pinched loop produce no crossings.

    A polyline that visits the same point twice touches itself; a segment of
    zero length has no interior for anything to cross.  Both must read zero
    rather than raising or counting a phantom.
    """
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    assert inj.boundary_crossings(uv, [[0, 1, 2, 3, 4, 5]]) == 0


def test_collinear_boundary_does_not_cross(inj):
    """A polyline doubling back along its own line never crosses."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert inj.boundary_crossings(uv, [[0, 2, 1, 3]]) == 0


# ---------------------------------------------------------------------------
# 2. The four canonical charts.
# ---------------------------------------------------------------------------


def test_clean_disk_is_certified(inj):
    """An isometric grid is a disk with a simple boundary: proved injective."""
    sample = make_flat_grid(8)
    report = inj.certify(sample)
    assert report.flipped == 0
    assert report.degenerate == 0
    assert report.boundary_crossings == 0
    assert report.non_disk_charts == 0
    assert report.overlap_area == 0.0
    assert report.locally_injective
    assert report.globally_injective
    assert report.charts_checked == 1


@pytest.mark.parametrize("cw", [True, False])
def test_spiral_fold_crosses_exactly_twice(inj, cw):
    """The fold is invisible to a flip test and obvious to the boundary test.

    Every triangle keeps its orientation -- the ribbon's Jacobian determinant
    cannot change sign -- so ``flipped`` is 0 and ``locally_injective`` is True,
    exactly as measured on Suzanne (``docs/ORGANIC.md`` section 0.3).  The
    boundary crosses twice, by the derivation in :func:`make_spiral_fold`.
    """
    sample = make_spiral_fold(cw=cw)
    signed = tri_areas_uv(sample)
    # Clockwise winding mirrors the *whole* chart; that is still injective, so
    # it must not be reported as a flip.
    assert (signed < 0).all() if cw else (signed > 0).all()

    report = inj.certify(sample)
    assert report.flipped == 0
    assert report.degenerate == 0
    assert report.locally_injective
    assert report.boundary_crossings == 2
    assert report.non_disk_charts == 0
    assert report.overlap_area > 0.0
    assert not report.globally_injective


def test_everted_fan_flips_without_crossing(inj):
    """Flips with a perfectly simple boundary: the complementary failure.

    Together with :func:`test_spiral_fold_crosses_exactly_twice` this pins that
    neither half of the certificate subsumes the other.
    """
    report = inj.certify(make_everted_fan())
    assert report.flipped == 2
    assert report.boundary_crossings == 0
    assert not report.locally_injective
    assert not report.globally_injective


def test_clean_fan_is_certified(inj):
    """The same fan with its hub inside the ring is proved injective."""
    report = inj.certify(make_everted_fan(centre=(0.2, -0.1)))
    assert report.flipped == 0
    assert report.boundary_crossings == 0
    assert report.globally_injective


def test_both_defects_are_reported_together(inj):
    """A chart that folds *and* everts reports both, not whichever came first."""
    report = inj.certify(make_both_defects())
    assert report.flipped > 0
    assert report.boundary_crossings > 0
    assert not report.locally_injective
    assert not report.globally_injective


def test_closed_chart_cannot_be_certified(inj):
    """A chart with no boundary at all is never certified.

    ``sphere_planar`` welds into a single closed component, so it has zero
    boundary loops and ``boundary_crossings`` is trivially 0.  The simple-
    boundary theorem needs a *disk*; without ``non_disk_charts`` the report
    would claim a proof from an empty boundary.
    """
    sample = load_corpus("sphere_planar")
    report = inj.certify(sample)
    assert inj.boundary_loops(sample) == []
    assert report.boundary_crossings == 0
    assert report.non_disk_charts == 1
    assert not report.globally_injective


@pytest.mark.parametrize("nested", [False, True])
def test_bowtie_is_never_certified(inj, nested):
    """A bowtie has ``chi == 1`` and one boundary loop, and is not a disk.

    Without the pinch test this is the shape that walks straight through the
    certificate: the arithmetic invariant and the loop count both say "disk",
    and when one lobe is laid *inside* the other their boundaries touch only at
    the pinch, so ``boundary_crossings`` is zero as well.  Only the repeated UV
    vertex in the boundary walk -- and, for the nested case, the overlap
    measurement -- distinguishes it.
    """
    sample = make_bowtie(nested=nested)
    report = inj.certify(sample)
    assert report.charts_checked == 1
    assert report.boundary_crossings == 0
    assert report.locally_injective
    assert report.non_disk_charts == 1
    assert not report.globally_injective
    if nested:
        assert report.overlap_area > 0.0
    else:
        # Side by side, the map really is injective; refusing it is the
        # deliberate conservative direction, not an accident.
        assert report.overlap_area == 0.0
    # And exactly one boundary loop, of six vertices, visiting the pinch twice.
    loops = inj.boundary_loops(sample)
    assert len(loops) == 1
    assert len(loops[0]) == 6


def test_degenerate_triangles_break_local_injectivity(inj):
    """A collapsed UV triangle has no local inverse and is reported as such."""
    from tests.conftest import make_zero_area_mix

    report = inj.certify(make_zero_area_mix())
    assert report.degenerate >= 1
    assert not report.locally_injective
    assert not report.globally_injective


# ---------------------------------------------------------------------------
# 3. Boundary extraction.
# ---------------------------------------------------------------------------


def test_grid_boundary_is_one_loop_of_the_right_length(inj):
    """An ``n x n`` grid's chart boundary is a single loop of ``4n`` vertices."""
    for n in (2, 5, 8):
        loops = inj.boundary_loops(make_flat_grid(n))
        assert len(loops) == 1
        assert len(loops[0]) == 4 * n


def test_boundary_loop_vertices_are_welded_positions(inj):
    """Consecutive boundary entries are distinct UV positions forming a cycle."""
    sample = make_flat_grid(4)
    (loop,) = inj.boundary_loops(sample)
    points = sample.uvs[np.asarray(loop)]
    assert len(np.unique(points, axis=0)) == len(loop)
    closing = np.roll(points, -1, axis=0) - points
    # Every step is one grid cell along one axis: the walk really is adjacent.
    assert np.allclose(np.abs(closing).min(axis=1), 0.0)
    assert np.allclose(np.abs(closing).max(axis=1), 0.25)


def test_separate_islands_get_separate_charts_and_loops(inj):
    """Two disjoint UV triangles are two charts with one boundary loop each."""
    sample = make_overlapping_pair("partial")
    labels = inj.chart_labels(sample)
    assert sorted(labels.tolist()) == [0, 1]
    grouped = inj.boundary_loops_by_chart(sample)
    assert sorted(grouped) == [0, 1]
    assert all(len(loops) == 1 and len(loops[0]) == 3 for loops in grouped.values())


def test_supplied_chart_labels_are_honoured(inj):
    """Calling two separate islands one chart makes that chart a non-disk.

    ``core/charts.py`` certifies a *proposed* merge before the UVs reflect it,
    so a caller-supplied labelling has to change the answer.  Two disjoint
    regions called one chart have two boundary loops and Euler characteristic 2,
    which is exactly the "not a disk" the merge stage must reject.
    """
    sample = make_overlapping_pair("partial")
    merged = np.zeros(sample.n_tris, dtype=np.int64)
    report = inj.certify(sample, tri_chart=merged)
    assert report.charts_checked == 1
    assert report.non_disk_charts == 1
    assert not report.globally_injective
    assert len(inj.boundary_loops_by_chart(sample, merged)[0]) == 2


def test_crossings_localise_to_the_guilty_chart(inj):
    """``crossings_by_chart`` names the chart, not just the layout.

    A clean grid parked next to a folded ribbon: the total is the ribbon's, and
    the report has to say which one owns it or ``repair_folds`` would have to
    re-flatten everything.
    """
    fold = make_spiral_fold()
    grid = make_flat_grid(4)
    shifted = grid.uvs + np.array([20.0, 20.0])
    verts = np.concatenate([fold.verts, grid.verts + np.array([100.0, 0.0, 0.0])])
    tri_verts = np.concatenate([fold.tri_verts, grid.tri_verts + fold.n_verts])
    tri_loops = np.concatenate([fold.tri_loops, grid.tri_loops + fold.n_loops])
    uvs = np.concatenate([fold.uvs, shifted])
    loop_verts = np.concatenate([fold.loop_verts, grid.loop_verts + fold.n_verts])
    combined = make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=loop_verts,
        name="fold_plus_grid",
    )
    counts = inj.crossings_by_chart(combined)
    assert sum(counts.values()) == 2
    assert sorted(counts.values()) == [0, 2]


def test_chart_count_matches_the_evaluators_island_count(inj, evaluator, corpus_sample):
    """``charts_checked`` and ``UVMetrics.island_count`` must never disagree.

    They are computed by different code from different welding rules; if they
    can drift, "chart 7 is broken" means nothing to anything downstream.
    """
    report = inj.certify(corpus_sample, compute_overlap=False)
    metrics = evaluator.evaluate(corpus_sample, compute_overlap=False)
    assert report.charts_checked == metrics.island_count


# ---------------------------------------------------------------------------
# 4. Agreement with the independent naive oracle.
# ---------------------------------------------------------------------------

_ORACLE_FIELDS = (
    "flipped",
    "degenerate",
    "boundary_crossings",
    "locally_injective",
    "charts_checked",
    "non_disk_charts",
)


def _assert_agrees(inj, sample, *, do_overlap: bool) -> None:
    """Assert the fast certificate and the naive oracle agree field by field."""
    fast = inj.certify(sample, compute_overlap=do_overlap)
    slow = ref.naive_certify(sample, do_overlap=do_overlap)
    mismatch = {
        field: (getattr(fast, field), slow[field])
        for field in _ORACLE_FIELDS
        if getattr(fast, field) != slow[field]
    }
    assert not mismatch, "%s: fast/naive disagree on %s" % (sample.name, mismatch)
    if do_overlap:
        # The certificate's ``overlap_area`` is the evaluator's, which
        # deliberately **saturates** at the total used UV area so a
        # catastrophically stacked layout stays interpretable (see
        # ``evaluator._overlap_area``).  The oracle sums without a cap, so on
        # ``cube_project`` -- whose projection stacks six faces on one spot --
        # it reports 15 against the evaluator's 6.  What must hold is that the
        # two agree on whether there is *any* overlap, and that saturation only
        # ever reduces the figure.
        assert (fast.overlap_area > 0.0) == (slow["overlap_area"] > 0.0)
        assert fast.overlap_area <= slow["overlap_area"] + 1e-12
    fast_loops = inj.boundary_loops(sample)
    slow_loops = ref.naive_boundary_loops(sample)
    assert canonical_loops(fast_loops, sample.uvs) == canonical_loops(slow_loops, sample.uvs)
    # Both promise a canonical form, so the *same* lists, not merely the same
    # curves -- which is what makes the segment ids below comparable.
    assert fast_loops == slow_loops
    fast_pairs = sorted(map(tuple, inj.crossing_pairs(sample.uvs, fast_loops).tolist()))
    slow_pairs = sorted(ref.naive_crossing_pairs(sample.uvs, slow_loops, limit=20_000))
    assert fast_pairs == slow_pairs
    # Chart *numbering*, not just the partition: "chart 9 is folded" has to mean
    # the same chart to both, or localisation is worthless.
    assert inj.chart_labels(sample).tolist() == ref.naive_chart_labels(sample)
    assert inj.crossings_by_chart(sample) == slow["chart_crossings"]


@pytest.mark.parametrize("name", ORACLE_CORPUS)
def test_oracle_agreement_on_corpus(inj, name):
    """The two implementations agree on every committed corpus mesh."""
    sample = load_corpus(name)
    _assert_agrees(inj, sample, do_overlap=name in SMALL_CORPUS_NAMES)


@pytest.mark.parametrize("mode", ["chaotic", "jittered", "spiky"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_oracle_agreement_on_random_meshes(inj, mode, seed):
    """Agreement survives randomised nonsense, including collapsed triangles."""
    _assert_agrees(inj, make_random_mesh(seed, mode=mode), do_overlap=False)


@pytest.mark.parametrize(
    "builder",
    [
        make_spiral_fold,
        make_everted_fan,
        make_both_defects,
        make_bowtie,
        lambda: make_bowtie(nested=True),
        lambda: make_flat_grid(6),
        lambda: make_overlapping_pair("identical"),
        lambda: make_overlapping_pair("contained"),
        lambda: make_overlapping_pair("adjacent"),
    ],
)
def test_oracle_agreement_on_built_cases(inj, builder):
    """Agreement on the hand-built defect cases, overlap area included."""
    _assert_agrees(inj, builder(), do_overlap=True)


@pytest.mark.parametrize("seed", list(range(24)))
def test_oracle_agreement_on_random_polylines(inj, seed):
    """The crossing sets -- not just the counts -- match on random polylines.

    Every third seed lives on a coarse integer lattice, so these cases are dense
    in exact collinearity and shared endpoints: the configurations where a
    tolerance-based predicate quietly invents or loses a crossing.  The
    bounding-box rejection inside the oracle is checked here too, by running it
    both ways.
    """
    uv, loops = random_polylines(seed)
    fast = sorted(map(tuple, inj.crossing_pairs(uv, loops).tolist()))
    slow = sorted(ref.naive_crossing_pairs(uv, loops))
    slow_unfiltered = sorted(ref.naive_crossing_pairs(uv, loops, use_bbox=False))
    assert slow == slow_unfiltered, "the oracle's bounding-box rejection changed the answer"
    assert fast == slow
    assert inj.boundary_crossings(uv, loops) == len(slow)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_exact_predicate_matches_exact_rational_arithmetic(inj, seed):
    """The filtered float64 predicate agrees with exact rational arithmetic.

    Reaching into ``_orient2d_sign`` on purpose: its exactness is the module's
    central claim, and the cases that break a naive predicate -- a query point
    within an ULP or two of the line -- barely survive to the public surface.
    The triples below are built *on* the line and then nudged by a few ULPs, so
    the true sign is known to be delicate and the float64 evaluation has no
    margin at all.
    """
    rng = np.random.default_rng(seed)
    count = 4000
    a = rng.uniform(-1.0, 1.0, size=(count, 2))
    direction = rng.uniform(-1.0, 1.0, size=(count, 2))
    b = a + direction
    t = rng.uniform(0.2, 0.8, size=(count, 1))
    c = a + t * direction
    steps = rng.integers(-3, 4, size=(count, 2))
    c = np.nextafter(c, np.where(steps >= 0, np.inf, -np.inf))
    for _ in range(3):
        more = rng.integers(0, 2, size=(count, 2)).astype(bool)
        c = np.where(more, np.nextafter(c, np.where(steps >= 0, np.inf, -np.inf)), c)

    fast = inj._orient2d_sign(a, b, c)
    slow = [ref.naive_orient2d_sign(a[i], b[i], c[i]) for i in range(count)]
    assert fast.tolist() == slow


def test_oracle_rejects_degenerate_clip_triangles():
    """Regression: the oracle's own clipper used to return the subject's area.

    ``tri_overlap_area`` clips the subject against each of the clip triangle's
    half-planes, and a *zero-length* edge's half-plane test reduces to
    ``0 >= -1e-15`` -- vacuously true, so that edge clips nothing.  Collapse all
    three and the subject came back untouched: a unit triangle "overlapping" a
    point five units away by its whole area.  It went unnoticed because every
    ``reference_metrics`` call in the suite passes ``do_overlap=False``, and it
    would have shown up as the fast implementation being "wrong" on the first
    corpus mesh with a collapsed triangle.
    """
    unit = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    collapsed = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
    edge_collapsed = [(5.0, 5.0), (5.0, 5.0), (6.0, 6.0)]
    assert ref.tri_overlap_area(unit, collapsed) == 0.0
    assert ref.tri_overlap_area(collapsed, unit) == 0.0
    assert ref.tri_overlap_area(unit, edge_collapsed) == 0.0
    # ... while a genuine overlap is untouched by the guard.
    assert ref.tri_overlap_area(unit, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]) > 0.0


def test_exactly_collinear_points_report_zero(inj):
    """Points constructed to lie exactly on a line report sign 0, not noise."""
    scale = 2.0**-20
    a = np.array([[0.0, 0.0]] * 40, dtype=np.float64)
    b = np.array([[1.0, 3.0]] * 40, dtype=np.float64)
    steps = (np.arange(40, dtype=np.float64) + 1.0) * scale
    c = np.stack([steps, 3.0 * steps], axis=1)
    assert inj._orient2d_sign(a, b, c).tolist() == [0] * 40


# ---------------------------------------------------------------------------
# 5. Consistency with the evaluator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_overlap_implies_not_certified_on_corpus(inj, evaluator, name):
    """Positive overlap area and "globally injective" can never coexist.

    Two entirely different machines measure this -- exact convex clipping over a
    spatial hash against boundary combinatorics -- so a layout the scorer
    penalises for overlap must never come back certified.
    """
    sample = load_corpus(name)
    metrics = evaluator.evaluate(sample, compute_overlap=True)
    report = inj.certify(sample)
    if metrics.overlap_area > 0.0:
        assert not report.globally_injective
    assert report.overlap_area == pytest.approx(metrics.overlap_area, rel=1e-12, abs=1e-18)


@pytest.mark.parametrize("kind", ["partial", "contained", "identical"])
def test_overlapping_pairs_are_never_certified(inj, kind):
    """Two islands that genuinely overlap are refused, crossings or not.

    ``contained`` is the case with **zero** boundary crossings, so it only fails
    because the overlap measurement is part of the verdict.
    """
    report = inj.certify(make_overlapping_pair(kind))
    assert report.overlap_area > 0.0
    assert not report.globally_injective


def test_adjacent_triangles_are_certified(inj):
    """Two triangles sharing a UV edge are clean, not a defect."""
    report = inj.certify(make_overlapping_pair("adjacent"))
    assert report.overlap_area == 0.0
    assert report.boundary_crossings == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_overlap_implies_not_certified_on_random_meshes(inj, evaluator, seed):
    """The same implication on randomised nonsense."""
    sample = make_random_mesh(seed, mode="chaotic")
    metrics = evaluator.evaluate(sample, compute_overlap=True)
    report = inj.certify(sample)
    if metrics.overlap_area > 0.0:
        assert not report.globally_injective


# ---------------------------------------------------------------------------
# 6. Robustness and determinism.
# ---------------------------------------------------------------------------


def test_certify_never_raises_on_pathological_meshes(inj):
    """Every deliberately broken mesh returns a report instead of an exception."""
    for sample in all_pathological():
        report = inj.certify(sample)
        assert isinstance(report.boundary_crossings, int)
        assert isinstance(report.globally_injective, bool)


def test_certify_never_raises_on_corrupt_indices(inj):
    """Out-of-range loop indices are dropped and counted, never dereferenced."""
    sample = make_flat_grid(4)
    broken = np.array(sample.tri_loops, dtype=np.int32)
    broken[0] = [999_999, 1, 2]
    corrupt = make_sample(
        sample.verts,
        sample.tri_verts,
        tri_loops=broken,
        uvs=sample.uvs,
        loop_verts=sample.loop_verts,
        name="corrupt",
    )
    report = inj.certify(corrupt)
    assert report.degenerate >= 1
    assert not report.globally_injective


def test_certify_survives_nan_uvs(inj):
    """NaN and Inf UVs are degenerate, not a crash and not a silent pass."""
    from tests.conftest import make_nan_uvs

    report = inj.certify(make_nan_uvs())
    assert report.degenerate >= 1
    assert not report.globally_injective


def test_empty_mesh_is_vacuously_clean(inj):
    """A mesh with no triangles has nothing to fold."""
    from tests.conftest import make_empty

    report = inj.certify(make_empty())
    assert report.charts_checked == 0
    assert report.boundary_crossings == 0
    assert report.globally_injective


def test_certify_is_deterministic(inj):
    """Identical input gives an identical report, every time."""
    sample = make_spiral_fold()
    first = inj.certify(sample)
    for _ in range(3):
        assert inj.certify(sample) == first


def test_crossing_pairs_are_sorted_and_unique(inj):
    """The diagnostic pair list is canonical, so it can be diffed."""
    uv, loops = random_polylines(4)
    pairs = inj.crossing_pairs(uv, loops)
    assert (pairs[:, 0] < pairs[:, 1]).all()
    flat = pairs[:, 0] * (pairs.max() + 1) + pairs[:, 1]
    assert (np.diff(flat) > 0).all()


def test_report_round_trips_through_a_dict(inj):
    """``to_dict``/``from_dict`` preserve the report exactly."""
    report = inj.certify(make_spiral_fold())
    assert inj.InjectivityReport.from_dict(report.to_dict()) == report
    assert inj.InjectivityReport.from_dict(None) == inj.InjectivityReport()


def test_report_exposes_the_contract_fields(inj):
    """Every field ``docs/ORGANIC.md`` section 4.1 names is present."""
    fields = set(inj.InjectivityReport.__dataclass_fields__)
    assert {
        "flipped",
        "degenerate",
        "boundary_crossings",
        "overlap_area",
        "locally_injective",
        "globally_injective",
        "charts_checked",
    } <= fields


def test_compute_overlap_false_skips_the_measurement(inj):
    """Turning the overlap pass off reports 0.0 and keeps the rest identical."""
    sample = make_spiral_fold()
    with_overlap = inj.certify(sample)
    without = inj.certify(sample, compute_overlap=False)
    assert with_overlap.overlap_area > 0.0
    assert without.overlap_area == 0.0
    assert without.boundary_crossings == with_overlap.boundary_crossings
    assert not without.globally_injective


# ---------------------------------------------------------------------------
# 7. The performance budget.
# ---------------------------------------------------------------------------


def _rose_curve(n: int) -> tuple[np.ndarray, list[list[int]]]:
    """Return a smooth simple closed curve with ``n`` segments."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    radius = 1.0 + 0.3 * np.sin(5.0 * t)
    uv = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    return uv, [list(range(n))]


@pytest.mark.perf
def test_hundred_thousand_segments_within_budget(inj):
    """100k boundary segments in under 500 ms -- ``docs/ORGANIC.md`` section 4.1.

    Measured as the best of three, because the machine is shared with whatever
    else is running; the budget is generous enough that a genuine regression to
    ``O(n^2)`` would blow it by orders of magnitude rather than percent.
    """
    uv, loops = _rose_curve(PERF_SEGMENTS)
    best = math.inf
    for _ in range(3):
        started = time.perf_counter()
        crossings = inj.boundary_crossings(uv, loops)
        best = min(best, time.perf_counter() - started)
    assert crossings == 0
    assert best < PERF_BUDGET_S, "100k segments took %.3f s, budget %.3f s" % (
        best,
        PERF_BUDGET_S,
    )


@pytest.mark.perf
def test_many_charts_within_budget(inj):
    """A packed atlas shape -- 2000 small loops -- stays inside the same budget.

    The grid has to cope with a layout that is mostly empty space between many
    small, dense curves, not just one long smooth one.
    """
    points: list[np.ndarray] = []
    loops: list[list[int]] = []
    per_loop = 50
    for i in range(2000):
        cx, cy = divmod(i, 50)
        angle = np.linspace(0.0, 2.0 * math.pi, per_loop, endpoint=False)
        block = np.stack(
            [cx + 0.45 * np.cos(angle), cy + 0.45 * np.sin(angle)], axis=1
        )
        base = sum(len(loop) for loop in loops)
        points.append(block)
        loops.append(list(range(base, base + per_loop)))
    uv = np.concatenate(points)
    started = time.perf_counter()
    crossings = inj.boundary_crossings(uv, loops)
    elapsed = time.perf_counter() - started
    assert crossings == 0
    assert elapsed < PERF_BUDGET_S, "%d segments took %.3f s" % (uv.shape[0], elapsed)
