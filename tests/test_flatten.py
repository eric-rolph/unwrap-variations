"""Tests for ``unwrap_variations.core.flatten`` -- injective flattening.

``docs/ORGANIC.md`` section 0.3 is the fact this module exists to handle, and it
is also the fact that makes it hard to test honestly: Suzanne's layout has
**zero flipped triangles** and still overlaps itself, so any test that only
counts negative signed areas would pass on a broken layout.  The suite is
therefore built around the guarantee chain rather than around the symptom:

1. **Tutte produces no flips.**  Checked on many procedurally generated disk
   triangulations, deliberately including the nasty ones -- slivers, high
   valence, near-degenerate fans, non-convex boundaries.
2. **The descent never introduces a flip and never raises the energy.**  Both
   are invariants of the algorithm, not statistical properties, so they are
   asserted at *every* iteration count on *every* random start.
3. **The gradient is the gradient.**  Pinned against central differences; a
   sign error there would still descend *something*, just not the energy the
   docstring claims.
4. **A layout whose boundary self-intersects with every triangle positively
   oriented is detected and repaired.**  This is the Suzanne defect, built by
   hand at a size that can be reasoned about.
5. **Rule 2 holds.**  A clean chart comes back bit-identical; a repair that does
   not improve the score is discarded.
6. **It is deterministic and it never raises**, on anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import (
    all_pathological,
    import_or_skip,
    load_corpus,
    make_flat_grid,
)
from unwrap_variations.core.meshdata import make_sample, tri_areas_uv


@pytest.fixture(scope="module")
def flatten():
    """The flattener module under test."""
    return import_or_skip("unwrap_variations.core.flatten")


# ---------------------------------------------------------------------------
# Chart builders.  Procedural, deterministic, and deliberately unpleasant.
# ---------------------------------------------------------------------------


def grid_chart(flatten, n: int = 6, *, warp: float = 0.0, seed: int = 0, jitter: float = 0.0):
    """An ``n x n`` triangulated square patch, optionally warped and jittered.

    A disk with a convex boundary in 3D but an arbitrary interior once warped.
    ``jitter`` moves vertices around inside the patch, which is what produces
    the sliver triangles that break a careless mean-value weight.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    if jitter:
        inner = np.zeros_like(gx, dtype=bool)
        inner[1:-1, 1:-1] = True
        gx = gx + np.where(inner, rng.uniform(-jitter, jitter, gx.shape), 0.0)
        gy = gy + np.where(inner, rng.uniform(-jitter, jitter, gy.shape), 0.0)
    gz = warp * np.sin(3.0 * gx) * np.cos(3.0 * gy) if warp else np.zeros_like(gx)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = (i + 1) * (n + 1) + j
            c = (i + 1) * (n + 1) + j + 1
            d = i * (n + 1) + j + 1
            tris.append((a, b, c))
            tris.append((a, c, d))
    tri = np.asarray(tris, dtype=np.int32)
    return flatten.Chart(
        verts=verts,
        tris=tri,
        loops=np.full(tri.shape, -1, dtype=np.int32),
        name="grid%d" % n,
    )


def fan_chart(flatten, spokes: int = 24, *, radius_jitter: float = 0.0, seed: int = 0):
    """A triangle fan: one interior vertex of very high valence.

    High valence is where a hand-rolled Laplacian assembly tends to go wrong,
    and a jittered radius makes the fan's triangles wildly unequal.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, spokes, endpoint=False)
    radii = 1.0 + (rng.uniform(-radius_jitter, radius_jitter, spokes) if radius_jitter else 0.0)
    rim = np.stack([radii * np.cos(angles), radii * np.sin(angles), np.zeros(spokes)], axis=1)
    verts = np.concatenate([np.zeros((1, 3)), rim])
    tris = np.array(
        [[0, 1 + i, 1 + (i + 1) % spokes] for i in range(spokes)], dtype=np.int32
    )
    return flatten.Chart(
        verts=verts,
        tris=tris,
        loops=np.full(tris.shape, -1, dtype=np.int32),
        name="fan%d" % spokes,
    )


def sliver_chart(flatten, n: int = 20, *, thickness: float = 1e-4):
    """A long thin strip: every triangle is a sliver, by construction.

    Near-degenerate triangles are where ``tan(theta/2)`` and the cotangent both
    blow up, so this is the shape that decides whether the weights are stable.
    """
    verts = []
    for k in range(n + 1):
        verts.append([k / n, 0.0, 0.0])
        verts.append([k / n, thickness, 0.0])
    tris = []
    for k in range(n):
        a, b, c, d = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        tris.append((a, c, d))
        tris.append((a, d, b))
    tri = np.asarray(tris, dtype=np.int32)
    return flatten.Chart(
        verts=np.asarray(verts, dtype=np.float64),
        tris=tri,
        loops=np.full(tri.shape, -1, dtype=np.int32),
        name="sliver",
    )


def notched_chart(flatten, n: int = 8):
    """An L-shaped patch: a disk with a strongly non-convex boundary."""
    chart = grid_chart(flatten, n)
    tris = np.asarray(chart.tris, dtype=np.int64)
    centroid = chart.verts[tris].mean(axis=1)
    keep = ~((centroid[:, 0] > 0.5) & (centroid[:, 1] > 0.5))
    kept = tris[keep]
    used, local = np.unique(kept.reshape(-1), return_inverse=True)
    local = local.reshape(-1, 3).astype(np.int32)
    return flatten.Chart(
        verts=chart.verts[used],
        tris=local,
        loops=np.full(local.shape, -1, dtype=np.int32),
        name="notched",
    )


def all_charts(flatten):
    """Every procedural chart, each with a stated reason it is hard."""
    charts = [
        (grid_chart(flatten, 3), "smallest non-trivial disk"),
        (grid_chart(flatten, 6), "plain disk"),
        (grid_chart(flatten, 8, warp=0.6), "curved disk, not developable"),
        (grid_chart(flatten, 6, jitter=0.35, seed=1), "jittered: slivers"),
        (grid_chart(flatten, 6, jitter=0.45, seed=2), "heavily jittered"),
        (grid_chart(flatten, 7, warp=1.0, jitter=0.3, seed=3), "curved and jittered"),
        (fan_chart(flatten, 6), "low-valence fan"),
        (fan_chart(flatten, 48), "valence-48 hub"),
        (fan_chart(flatten, 32, radius_jitter=0.7, seed=4), "ragged high-valence fan"),
        (sliver_chart(flatten), "every triangle a sliver"),
        (sliver_chart(flatten, thickness=1e-7), "near-degenerate slivers"),
        (notched_chart(flatten), "non-convex L-shaped boundary"),
    ]
    return charts


def folded_strip_sample(n: int = 48, turns: float = 1.35):
    """A flat 3D strip whose UV layout is coiled past a full turn.

    **This is the Suzanne defect, made small enough to reason about.**  Every
    triangle keeps a positive signed UV area -- the coil is gentle per triangle
    -- but the strip laps over itself, so the chart boundary self-intersects and
    the layout is not globally injective.  A ``signed_area < 0`` check reports
    a perfectly clean layout here, which is exactly the trap
    ``docs/ORGANIC.md`` section 0.3 documents.
    """
    length, width = 2.0, 0.25
    r_in, r_out = 0.18, 0.30
    verts = []
    for k in range(n + 1):
        verts.append([k / n * length, 0.0, 0.0])
        verts.append([k / n * length, width, 0.0])
    total = -turns * 2.0 * math.pi
    ring = []
    for k in range(n + 1):
        a = (k / n) * total
        ring.append((0.5 + r_in * math.cos(a), 0.5 + r_in * math.sin(a)))
        ring.append((0.5 + r_out * math.cos(a), 0.5 + r_out * math.sin(a)))

    tri_verts, tri_loops, loop_verts, uvs = [], [], [], []
    for k in range(n):
        a, b, c, d = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        for tri in ((a, c, d), (a, d, b)):
            base = len(loop_verts)
            tri_verts.append(list(tri))
            tri_loops.append([base, base + 1, base + 2])
            for v in tri:
                loop_verts.append(v)
                uvs.append(ring[v])
    return make_sample(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tri_verts, dtype=np.int32),
        tri_loops=np.asarray(tri_loops, dtype=np.int32),
        uvs=np.asarray(uvs, dtype=np.float64),
        loop_verts=np.asarray(loop_verts, dtype=np.int32),
        name="folded_strip",
    )


def signed_areas(flatten, uv, chart):
    """Signed UV areas of a chart layout."""
    return flatten._signed_areas(np.asarray(uv, float), np.asarray(chart.tris, np.int64))


# ---------------------------------------------------------------------------
# 1. Topology
# ---------------------------------------------------------------------------


def test_every_procedural_chart_is_a_disk(flatten):
    """The builders must actually build disks, or the rest of the suite is void."""
    for chart, why in all_charts(flatten):
        topo = flatten.chart_topology(chart)
        assert topo.is_disk, "%s (%s) is not a disk: %s" % (chart.name, why, topo.problems)
        assert topo.euler == 1
        assert len(topo.loops) == 1


def test_closed_surface_is_not_a_disk(flatten):
    """A closed mesh has no boundary and cannot be Tutte-embedded."""
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    tris = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    chart = flatten.Chart(corners, tris, np.full(tris.shape, -1, np.int32), "tetra")
    topo = flatten.chart_topology(chart)
    assert not topo.is_disk
    assert any("boundary" in p for p in topo.problems)


def test_annulus_is_not_a_disk(flatten):
    """Two boundary loops means Tutte has no single convex boundary to use."""
    chart = grid_chart(flatten, 6)
    tris = np.asarray(chart.tris, dtype=np.int64)
    centroid = chart.verts[tris].mean(axis=1)
    keep = ~(
        (centroid[:, 0] > 0.3) & (centroid[:, 0] < 0.7)
        & (centroid[:, 1] > 0.3) & (centroid[:, 1] < 0.7)
    )
    kept = tris[keep]
    used, local = np.unique(kept.reshape(-1), return_inverse=True)
    local = local.reshape(-1, 3).astype(np.int32)
    ring = flatten.Chart(
        chart.verts[used], local, np.full(local.shape, -1, np.int32), "annulus"
    )
    topo = flatten.chart_topology(ring)
    assert not topo.is_disk
    assert topo.euler == 0


def test_boundary_loop_is_wound_with_the_triangles(flatten):
    """The extracted loop must agree with the triangles' orientation.

    If it does not, mapping it counter-clockwise onto the Tutte circle mirrors
    the entire chart and every triangle comes out flipped -- a failure that
    looks like a solver bug and is really an indexing one.
    """
    chart = grid_chart(flatten, 4)
    loop = np.asarray(flatten.chart_topology(chart).loops[0], dtype=np.int64)
    pts = chart.verts[loop][:, :2]
    twice_area = float(
        np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    )
    assert twice_area > 0.0


# ---------------------------------------------------------------------------
# 2. Tutte: the guarantee
# ---------------------------------------------------------------------------


def test_tutte_produces_zero_flips_on_every_chart(flatten):
    """Tutte's theorem in force: a convex boundary plus positive weights cannot fold.

    This is the base of the whole guarantee chain.  If a single flip appears
    here, everything downstream is building on sand -- the descent's step cap
    only *preserves* injectivity, it cannot create it.
    """
    for chart, why in all_charts(flatten):
        uv = flatten.tutte_embed(chart)
        assert np.isfinite(uv).all(), "%s produced non-finite UVs" % chart.name
        areas = signed_areas(flatten, uv, chart)
        assert (areas > 0.0).all(), "%s (%s): %d of %d triangles are not positive" % (
            chart.name,
            why,
            int((areas <= 0.0).sum()),
            areas.size,
        )


@pytest.mark.parametrize("seed", range(12))
def test_tutte_zero_flips_on_random_disks(flatten, seed):
    """Same guarantee over randomised geometry, including degenerate corners."""
    chart = grid_chart(flatten, 5 + seed % 4, warp=0.2 * (seed % 5), jitter=0.4, seed=seed)
    uv = flatten.tutte_embed(chart)
    areas = signed_areas(flatten, uv, chart)
    assert (areas > 0.0).all(), "seed %d: %d non-positive" % (
        seed,
        int((areas <= 0.0).sum()),
    )


def test_tutte_boundary_is_convex_and_simple(flatten):
    """The boundary lands on a circle, so it cannot self-intersect.

    Radii are measured from the circle's own centre, the origin -- *not* from
    the mean of the boundary points, which is a different point entirely because
    chord-length spacing distributes them unevenly around the circle.
    """
    chart = grid_chart(flatten, 6, warp=0.5)
    topo = flatten.chart_topology(chart)
    uv = flatten.tutte_embed(chart)
    loop = np.asarray(topo.loops[0], dtype=np.int64)
    ring = uv[loop]

    radii = np.linalg.norm(ring, axis=1)
    assert radii.std() / radii.mean() < 1e-9, "boundary is not on a circle"

    # Convexity: every consecutive turn goes the same way, and it goes
    # anticlockwise, which is what makes the interior triangles come out
    # positively oriented.
    nxt = np.roll(ring, -1, axis=0)
    after = np.roll(ring, -2, axis=0)
    turns = (nxt[:, 0] - ring[:, 0]) * (after[:, 1] - nxt[:, 1]) - (
        nxt[:, 1] - ring[:, 1]
    ) * (after[:, 0] - nxt[:, 0])
    assert (turns > 0.0).all(), "boundary polygon is not convex anticlockwise"
    assert flatten._boundary_crossings(uv, [list(topo.loops[0])]) == 0


def test_tutte_falls_back_to_a_projection_for_a_non_disk(flatten):
    """A non-disk gets a finite, usable layout and no exception.

    Returning a projection rather than raising is what lets ``repair_folds``
    report "this chart cannot be flattened" instead of taking a Blender
    operator down with it.
    """
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tris = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    chart = flatten.Chart(corners, tris, np.full(tris.shape, -1, np.int32), "tetra")
    uv = flatten.tutte_embed(chart)
    assert uv.shape == (4, 2)
    assert np.isfinite(uv).all()


def test_tutte_is_deterministic(flatten):
    """Bit-identical output for bit-identical input."""
    chart = grid_chart(flatten, 7, warp=0.4, jitter=0.2, seed=5)
    assert np.array_equal(flatten.tutte_embed(chart), flatten.tutte_embed(chart))


# ---------------------------------------------------------------------------
# 3. The descent: the two invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iters", [1, 2, 4, 8, 16, 32, 64])
def test_descent_never_introduces_a_flip(flatten, iters):
    """A step that cannot flip a triangle cannot break local injectivity.

    Asserted at every iteration count rather than only at the end: a scheme that
    flips on iteration 7 and happens to unflip by iteration 32 has still lost
    the invariant, and would lose it for good on a different chart.
    """
    for chart, why in all_charts(flatten):
        start = flatten.tutte_embed(chart)
        if not (signed_areas(flatten, start, chart) > 0.0).all():
            continue
        moved = flatten.minimize_distortion(start, chart, iters=iters)
        areas = signed_areas(flatten, moved, chart)
        assert (areas > 0.0).all(), "%s (%s) flipped %d triangles after %d iters" % (
            chart.name,
            why,
            int((areas <= 0.0).sum()),
            iters,
        )


@pytest.mark.parametrize("seed", range(10))
def test_descent_never_flips_from_a_random_valid_start(flatten, seed):
    """The invariant is about the *step*, so it must hold from any valid start."""
    rng = np.random.default_rng(seed)
    chart = grid_chart(flatten, 5, warp=0.3, jitter=0.25, seed=seed)
    start = flatten.tutte_embed(chart)
    # Perturb, but only as far as the layout stays flip-free -- otherwise the
    # premise of the invariant is not met and the test would be vacuous.
    for scale in (0.2, 0.1, 0.05, 0.02, 0.0):
        candidate = start + rng.normal(0.0, scale * float(np.abs(start).max()), start.shape)
        if (signed_areas(flatten, candidate, chart) > 0.0).all():
            start = candidate
            break
    moved = flatten.minimize_distortion(start, chart, iters=24)
    assert (signed_areas(flatten, moved, chart) > 0.0).all()


def test_descent_energy_decreases_monotonically(flatten):
    """The energy is non-increasing in the iteration count, on every chart.

    A backtracking line search only accepts a step that lowers the energy, so
    this is an invariant of the algorithm rather than a tendency.  It is also
    the property that makes the checkpointed ladder in ``flatten_chart`` sound:
    the *energy* is monotone even though *certification* is not.
    """
    for chart, why in all_charts(flatten):
        start = flatten.tutte_embed(chart)
        previous = flatten.symmetric_dirichlet_energy(start, chart)
        for iters in (1, 2, 4, 8, 16, 32, 64):
            energy = flatten.symmetric_dirichlet_energy(
                flatten.minimize_distortion(start, chart, iters=iters), chart
            )
            assert energy <= previous + 1e-9, (
                "%s (%s): energy rose from %.9f to %.9f at %d iterations"
                % (chart.name, why, previous, energy, iters)
            )
            previous = energy


def test_descent_actually_reduces_distortion(flatten):
    """Not just non-increasing -- it must get materially closer to isometric.

    A no-op would satisfy monotonicity perfectly.  4.0 is the isometric floor
    for this energy (both singular values 1), and a flat grid is exactly
    developable, so anything above about 4.01 means the descent is not working.
    """
    chart = grid_chart(flatten, 8)
    start = flatten.tutte_embed(chart)
    assert flatten.symmetric_dirichlet_energy(start, chart) > 4.5
    settled = flatten.minimize_distortion(start, chart, iters=64)
    assert flatten.symmetric_dirichlet_energy(settled, chart) < 4.01


def test_gradient_matches_central_differences(flatten):
    """Pin the analytic gradient against finite differences.

    A sign error in the barrier term still descends *something* -- just not the
    symmetric Dirichlet energy the docstring claims -- and every other test in
    this file would keep passing.
    """
    chart = grid_chart(flatten, 4, warp=0.4, jitter=0.15, seed=7)
    rest = flatten._Rest(chart)
    uv = flatten.tutte_embed(chart)
    value, grad = flatten._energy_and_gradient(uv, rest)
    assert math.isfinite(value)

    step = 1e-6 * max(float(np.abs(uv).max()), 1.0)
    rng = np.random.default_rng(0)
    for index in rng.choice(uv.shape[0], size=min(8, uv.shape[0]), replace=False):
        for axis in (0, 1):
            plus = uv.copy()
            minus = uv.copy()
            plus[index, axis] += step
            minus[index, axis] -= step
            numeric = (flatten._energy(plus, rest) - flatten._energy(minus, rest)) / (2 * step)
            assert numeric == pytest.approx(grad[index, axis], rel=2e-4, abs=1e-6)


def test_descent_leaves_an_already_optimal_layout_alone(flatten):
    """An isometric layout is a fixed point: nothing to gain, so nothing moves."""
    chart = grid_chart(flatten, 5)
    isometric = chart.verts[:, :2].copy()
    moved = flatten.minimize_distortion(isometric, chart, iters=16)
    assert np.abs(moved - isometric).max() < 1e-6


def test_descent_with_a_fixed_boundary_holds_the_boundary(flatten):
    """``free_boundary=False`` must genuinely pin the boundary vertices."""
    chart = grid_chart(flatten, 5, warp=0.4)
    topo = flatten.chart_topology(chart)
    start = flatten.tutte_embed(chart)
    moved = flatten.minimize_distortion(start, chart, iters=16, free_boundary=False)
    loop = np.asarray(topo.loops[0], dtype=np.int64)
    assert np.array_equal(moved[loop], start[loop])
    assert not np.array_equal(moved, start)


def test_descent_rejects_an_unknown_energy(flatten):
    """An unimplemented energy name is a programming error, reported loudly."""
    chart = grid_chart(flatten, 3)
    with pytest.raises(ValueError):
        flatten.minimize_distortion(flatten.tutte_embed(chart), chart, energy="arap")


def test_descent_rejects_a_mismatched_layout(flatten):
    """UVs that do not match the chart are a bug, not geometry."""
    chart = grid_chart(flatten, 3)
    with pytest.raises(ValueError):
        flatten.minimize_distortion(np.zeros((3, 2)), chart)


def test_descent_is_deterministic(flatten):
    """Bit-identical output for bit-identical input."""
    chart = grid_chart(flatten, 6, warp=0.3, jitter=0.2, seed=8)
    start = flatten.tutte_embed(chart)
    first = flatten.minimize_distortion(start, chart, iters=20)
    second = flatten.minimize_distortion(start, chart, iters=20)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# 4. flatten_chart: the certificate
# ---------------------------------------------------------------------------


def test_flatten_chart_certifies_every_procedural_chart(flatten):
    """Every disk must come back certified globally injective."""
    for chart, why in all_charts(flatten):
        result = flatten.flatten_chart(chart)
        assert result.ok, "%s (%s) failed: %s (stage %s, crossings %d)" % (
            chart.name,
            why,
            result.message,
            result.stage,
            result.boundary_crossings,
        )
        assert result.flipped == 0
        assert result.boundary_crossings == 0
        assert np.isfinite(result.uv).all()


def test_flatten_chart_never_returns_an_uncertified_layout(flatten):
    """The contract's hard rule: a worse-but-valid answer, never an invalid one.

    Either ``ok`` is True and the certificate is clean, or ``ok`` is False and
    the caller is told.  There is no third state where ``ok`` is True and the
    layout is broken.
    """
    for chart, _ in all_charts(flatten):
        result = flatten.flatten_chart(chart)
        if result.ok:
            assert (result.flipped, result.degenerate, result.boundary_crossings) == (0, 0, 0)


def test_flatten_chart_aligns_onto_a_target_and_preserves_its_area(flatten):
    """With a target, the result lands on it and reproduces its UV area exactly."""
    chart = grid_chart(flatten, 6, warp=0.3)
    tris = np.asarray(chart.tris, dtype=np.int64)
    target = chart.verts[:, :2] * 0.25 + np.array([0.6, 0.3])
    result = flatten.flatten_chart(chart, target=target)
    want = float(np.abs(flatten._signed_areas(target, tris)).sum())
    got = float(np.abs(flatten._signed_areas(result.uv, tris)).sum())
    assert got == pytest.approx(want, rel=1e-9)
    assert np.abs(result.uv.mean(axis=0) - target.mean(axis=0)).max() < 1e-9


def test_flatten_chart_accepts_a_duck_typed_chart(flatten):
    """``core/charts.py`` has its own Chart type; the two must interoperate."""

    class Foreign:
        """A chart-like object from another module."""

        def __init__(self, chart):
            self.verts = chart.verts
            self.tris = chart.tris
            self.name = "foreign"

    chart = grid_chart(flatten, 5)
    result = flatten.flatten_chart(Foreign(chart))
    assert result.ok
    assert result.uv.shape == (chart.n_verts, 2)


def test_flatten_chart_rejects_an_unusable_object_loudly(flatten):
    """A stranger without the required fields must raise, not misread whichever
    attributes happen to exist."""

    class Wrong:
        """Missing ``tris`` entirely."""

        verts = np.zeros((3, 3))

    with pytest.raises(TypeError):
        flatten.flatten_chart(Wrong())


def test_flatten_chart_is_deterministic(flatten):
    """Bit-identical output for bit-identical input."""
    chart = grid_chart(flatten, 6, warp=0.4, jitter=0.2, seed=9)
    assert np.array_equal(
        flatten.flatten_chart(chart).uv, flatten.flatten_chart(chart).uv
    )


# ---------------------------------------------------------------------------
# 5. Charts and welding
# ---------------------------------------------------------------------------


def test_chart_count_matches_the_evaluator_island_count(flatten):
    """This module's idea of a chart must be the metric's idea of an island.

    If the two drift apart, ``repair_folds`` repairs something the score is not
    measuring, and every acceptance number becomes meaningless.
    """
    from unwrap_variations.core.evaluator import evaluate

    for name in ("cube_project", "hardsurface", "sphere_smart"):
        sample = load_corpus(name)
        _, _, n_charts = flatten.uv_charts(sample)
        assert n_charts == evaluate(sample, compute_overlap=False).island_count, name


def test_chart_from_tris_welds_by_uv_by_default(flatten):
    """A seam splits a chart's vertices; a vertex weld would close it."""
    sample = load_corpus("cube_project")
    uv_vertex, tri_chart, _ = flatten.uv_charts(sample)
    group = np.flatnonzero(tri_chart == tri_chart[0])
    by_uv = flatten.chart_from_tris(sample, group, uv_vertex_of_loop=uv_vertex)
    by_vertex = flatten.chart_from_tris(sample, group, weld="vertex")
    assert by_uv.n_tris == by_vertex.n_tris == group.size
    assert by_uv.n_verts >= by_vertex.n_verts


def test_chart_from_tris_accepts_a_boolean_mask(flatten):
    """Callers pass masks as readily as index arrays."""
    sample = load_corpus("cube_project")
    mask = np.zeros(sample.n_tris, dtype=bool)
    mask[:4] = True
    assert flatten.chart_from_tris(sample, mask).n_tris == 4


def test_chart_from_tris_on_an_empty_selection(flatten):
    """Selecting nothing gives an empty chart, not an exception."""
    sample = load_corpus("cube_project")
    assert flatten.chart_from_tris(sample, np.zeros(0, dtype=np.int64)).n_tris == 0


# ---------------------------------------------------------------------------
# 6. Boundary crossings: the defect a flip test cannot see
# ---------------------------------------------------------------------------


def test_boundary_crossings_counts_a_self_intersecting_polygon(flatten):
    """A bow tie crosses itself exactly once."""
    bowtie = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert flatten._fallback_boundary_crossings(bowtie, [[0, 1, 2, 3]]) == 1


def test_boundary_crossings_ignores_a_simple_polygon(flatten):
    """A square does not cross itself, and adjacent segments sharing an endpoint
    are not crossings."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert flatten._fallback_boundary_crossings(square, [[0, 1, 2, 3]]) == 0


def test_boundary_crossings_treats_touching_as_not_crossing(flatten):
    """Exactly touching is not crossing -- the evaluator's overlap convention."""
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 0.0], [1.0, -1.0]])
    assert flatten._fallback_boundary_crossings(pts, [[0, 1, 2, 3]]) == 0
    # A T-junction: the spur's endpoint lands exactly on the square's edge.
    assert flatten._fallback_boundary_crossings(pts, [[0, 1, 2, 3], [4, 5, 4]]) == 0


def test_fallback_agrees_with_the_injectivity_module(flatten):
    """The runtime certificate and the local fallback must not disagree.

    ``core/injectivity.py`` is owned by another worker; if it is present, the two
    independent implementations are cross-checked, which is the only way either
    of them earns any trust.
    """
    module = flatten._injectivity_module()
    if module is None:
        pytest.skip("core/injectivity.py has not landed yet")
    sample = folded_strip_sample()
    uv_vertex, tri_chart, _ = flatten.uv_charts(sample)
    chart = flatten.chart_from_tris(
        sample, np.arange(sample.n_tris), uv_vertex_of_loop=uv_vertex
    )
    topo = flatten.chart_topology(chart)
    target = flatten._target_layout(chart, sample.uvs)
    loops = [list(loop) for loop in topo.loops]
    assert flatten._fallback_boundary_crossings(target, loops) == module.boundary_crossings(
        target, loops
    )


# ---------------------------------------------------------------------------
# 7. repair_folds -- the Suzanne defect, and Rule 2
# ---------------------------------------------------------------------------


def test_folded_strip_has_no_flips_but_is_not_injective(flatten):
    """The premise of the repair test, asserted rather than assumed.

    Every signed area positive **and** the boundary self-intersecting: this is
    exactly the shape of Suzanne's defect, and it is why the repair cannot be
    driven off ``flipped_tris``.
    """
    sample = folded_strip_sample()
    areas = tri_areas_uv(sample)
    assert (areas > 0.0).all(), "%d triangles are not positively oriented" % int(
        (areas <= 0.0).sum()
    )

    from unwrap_variations.core.evaluator import evaluate

    metrics = evaluate(sample)
    assert metrics.flipped_tris == 0
    assert metrics.overlap_area > 0.0

    uv_vertex, tri_chart, n_charts = flatten.uv_charts(sample)
    assert n_charts == 1
    chart = flatten.chart_from_tris(
        sample, np.arange(sample.n_tris), uv_vertex_of_loop=uv_vertex
    )
    topo = flatten.chart_topology(chart)
    assert topo.is_disk
    flipped, degenerate, crossings = flatten._certify_layout(
        flatten._target_layout(chart, sample.uvs), chart, topo
    )
    assert (flipped, degenerate) == (0, 0)
    assert crossings > 0, "the boundary of a coiled strip must self-intersect"


def test_repair_folds_fixes_the_folded_strip(flatten):
    """The headline: a fold with no flipped triangles is found and removed."""
    from unwrap_variations.core.evaluator import evaluate

    sample = folded_strip_sample()
    before = evaluate(sample)
    repaired, report = flatten.repair_folds(sample, sample.uvs)
    after = evaluate(sample.with_uvs(repaired))

    assert report.changed, report.messages
    assert report.charts_failed == 1
    assert report.charts_repaired == 1
    assert after.overlap_area == 0.0, "overlap %.10g remains" % after.overlap_area
    assert after.flipped_tris == 0
    assert before.overlap_area > 0.0


def test_repair_folds_leaves_a_clean_layout_bit_identical(flatten):
    """Rule 2: a stage that cannot improve a layout returns it untouched."""
    sample = make_flat_grid(8)
    repaired, report = flatten.repair_folds(sample, sample.uvs)
    assert not report.changed
    assert np.array_equal(repaired, sample.uvs)
    assert report.charts_repaired == 0


def test_repair_folds_leaves_clean_charts_bit_identical(flatten):
    """Only the failing chart's loops may move; every other UV is untouched.

    Not "close to identical" -- identical.  A repair that nudges a chart it had
    no reason to touch would change the layout's hash, invalidate a cached bake
    and make any regression impossible to attribute.
    """
    from unwrap_variations.core.evaluator import evaluate

    strip = folded_strip_sample(n=24)
    grid = make_flat_grid(4)
    combined = _combine(strip, grid, offset=(2.0, 0.0), uv_offset=(1.5, 0.0))
    before = evaluate(combined)
    assert before.overlap_area > 0.0

    repaired, report = flatten.repair_folds(combined, combined.uvs)
    if not report.changed:
        pytest.skip("repair declined on the combined sample: %s" % (report.messages,))
    grid_loops = np.arange(strip.n_loops, combined.n_loops)
    assert np.array_equal(repaired[grid_loops], combined.uvs[grid_loops])


def test_repair_folds_reverts_when_it_cannot_help(flatten):
    """A repair that does not raise the score is discarded, exactly.

    Simulated by making the flattener produce a certified but useless layout;
    the returned array must be the input's values, not merely close to them.
    """
    sample = folded_strip_sample(n=24)
    original = np.array(sample.uvs, copy=True)

    def useless(chart, *, target=None, iters=None):
        """Certified, but collapsed onto the target's own centroid."""
        return flatten.FlattenResult(
            uv=np.repeat(target.mean(axis=0)[None, :], chart.n_verts, axis=0),
            ok=True,
            stage="stub",
            flipped=0,
            degenerate=0,
            boundary_crossings=0,
            energy=4.0,
            iterations=0,
            seconds=0.0,
        )

    real = flatten.flatten_chart
    try:
        flatten.flatten_chart = useless
        repaired, report = flatten.repair_folds(sample, sample.uvs)
    finally:
        flatten.flatten_chart = real
    assert not report.changed
    assert np.array_equal(repaired, original)


def test_repair_folds_is_deterministic(flatten):
    """Bit-identical output for bit-identical input."""
    sample = folded_strip_sample(n=32)
    first, _ = flatten.repair_folds(sample, sample.uvs)
    second, _ = flatten.repair_folds(sample, sample.uvs)
    assert np.array_equal(first, second)


def test_repair_folds_rejects_a_mismatched_layout_without_raising(flatten):
    """A wrong-sized UV array comes back unchanged with the reason reported."""
    sample = folded_strip_sample(n=8)
    bogus = np.zeros((5, 2))
    out, report = flatten.repair_folds(sample, bogus)
    assert np.array_equal(out, bogus)
    assert not report.changed
    assert report.messages


def test_repair_folds_preserves_repaired_chart_area_when_it_can(flatten):
    """With no neighbours to collide with, the original UV area is kept exactly."""
    from unwrap_variations.core.evaluator import evaluate

    sample = folded_strip_sample()
    before = evaluate(sample)
    repaired, report = flatten.repair_folds(sample, sample.uvs)
    after = evaluate(sample.with_uvs(repaired))
    assert report.changed
    assert after.uv_coverage == pytest.approx(before.uv_coverage, rel=1e-9)


def _combine(first, second, *, offset, uv_offset):
    """Concatenate two samples into one, translated apart in 3D and in UV."""
    verts = np.concatenate(
        [first.verts, second.verts + np.array([offset[0], offset[1], 0.0])]
    )
    tri_verts = np.concatenate([first.tri_verts, second.tri_verts + first.n_verts])
    tri_loops = np.concatenate([first.tri_loops, second.tri_loops + first.n_loops])
    uvs = np.concatenate([first.uvs, second.uvs + np.asarray(uv_offset)])
    loop_verts = np.concatenate([first.loop_verts, second.loop_verts + first.n_verts])
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=loop_verts,
        name="combined",
    )


# ---------------------------------------------------------------------------
# 8. Containment: nothing here may raise into Blender
# ---------------------------------------------------------------------------


def test_repair_folds_survives_every_pathological_mesh(flatten):
    """Empty, collapsed, NaN-poisoned: report the damage, never raise."""
    for sample in all_pathological():
        out, report = flatten.repair_folds(sample, sample.uvs)
        assert out.shape == sample.uvs.shape
        assert isinstance(report.charts_checked, int)
        if out.size:
            assert np.isfinite(out).all() or not np.isfinite(sample.uvs).all()


def test_flatten_chart_survives_a_degenerate_chart(flatten):
    """A chart whose triangles have collapsed must report, not raise."""
    verts = np.zeros((4, 3))
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    chart = flatten.Chart(verts, tris, np.full(tris.shape, -1, np.int32), "collapsed")
    result = flatten.flatten_chart(chart)
    assert np.isfinite(result.uv).all()
    assert isinstance(result.ok, bool)


def test_chart_topology_survives_out_of_range_indices(flatten):
    """Corrupt indices are reported as problems, not raised as exceptions."""
    chart = flatten.Chart(
        np.zeros((3, 3)),
        np.array([[0, 1, 9]], dtype=np.int32),
        np.full((1, 3), -1, np.int32),
        "corrupt",
    )
    topo = flatten.chart_topology(chart)
    assert not topo.is_disk
    assert topo.problems


def test_flatten_chart_survives_out_of_range_indices(flatten):
    """``flatten_chart`` documents "Never raises"; a corrupt index must not.

    ``chart_topology`` already reports this chart as broken, so the flattener
    takes its not-a-disk branch -- but that branch still certifies the
    projection, and certification indexes ``uv`` with the corrupt triangles.
    A chart-like object arriving from ``core/charts.py`` with a stale index
    would take an ``IndexError`` straight out of a Blender operator.
    """
    chart = flatten.Chart(
        np.zeros((3, 3)),
        np.array([[0, 1, 9]], dtype=np.int32),
        np.full((1, 3), -1, np.int32),
        "corrupt",
    )
    result = flatten.flatten_chart(chart)
    assert result.ok is False
    assert np.isfinite(result.uv).all()
    assert result.uv.shape == (3, 2)


def test_certify_layout_refuses_a_chart_it_cannot_index(flatten):
    """An unindexable chart is never certified clean, whatever its layout."""
    chart = flatten.Chart(
        np.zeros((3, 3)),
        np.array([[0, 1, 7]], dtype=np.int32),
        np.full((1, 3), -1, np.int32),
        "corrupt",
    )
    topo = flatten.chart_topology(chart)
    verdict = flatten._certify_layout(np.zeros((3, 2)), chart, topo)
    assert verdict != (0, 0, 0)


def test_symmetric_dirichlet_energy_of_an_isometric_map_is_four(flatten):
    """The energy's normalisation is pinned: isometric reads exactly 4.0."""
    chart = grid_chart(flatten, 5)
    assert flatten.symmetric_dirichlet_energy(chart.verts[:, :2], chart) == pytest.approx(
        4.0, rel=1e-12
    )


def test_symmetric_dirichlet_energy_is_infinite_for_an_inverted_map(flatten):
    """The barrier: a collapsed or inverted triangle costs infinity, not a number."""
    chart = grid_chart(flatten, 4)
    mirrored = chart.verts[:, :2] * np.array([-1.0, 1.0])
    assert flatten.symmetric_dirichlet_energy(mirrored, chart) == math.inf


def test_max_feasible_step_stops_short_of_the_first_inversion(flatten):
    """The step cap is a closed-form root, so it can be checked exactly.

    A single triangle collapses when its apex reaches the opposite edge.  Moving
    the apex straight down at unit speed from height 1, that is ``t = 1``.
    """
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    direction = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
    assert flatten._max_feasible_step(uv, direction, tris) == pytest.approx(1.0)
    away = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    assert flatten._max_feasible_step(uv, away, tris) == math.inf
