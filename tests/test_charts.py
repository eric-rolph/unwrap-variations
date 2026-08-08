"""Tests for ``unwrap_variations.core.charts`` -- chart topology and merging.

Merging is the biggest single lever this add-on has on an organic mesh
(``docs/ORGANIC.md`` section 0.4: 100% of Suzanne's cut length is boundary
between two islands), which makes it the stage where a silent bug is most
expensive.  The properties that matter here:

* **Disk-ness is exact.**  ``chi = F - E + V == 1`` over the *triangulated*
  submesh -- diagonals included -- plus the conditions that make ``chi == 1``
  actually mean "disk".  The bowtie has ``chi == 1`` and is not a disk; that
  case is pinned below because the Euler number alone waves it through.
* **No merge is taken on trust.**  The stretch bound, the disk test and
  injectivity each have to be able to veto, and each is exercised separately.
* **Seam length never rises.**  A merge deletes mesh edges and creates none, so
  the cut length is monotonically non-increasing across the whole sequence.
* **The same input gives the same merges.**  A layout that changes between runs
  cannot be benchmarked.

The flattener is injected, so these tests supply their own: a best-fit-plane
projection, which is an *exact isometry* on a planar chart and therefore lets a
merge's measured stretch be predicted analytically instead of asserted loosely.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import (
    all_pathological,
    import_or_skip,
    make_cube,
    make_empty,
    make_flat_grid,
)
from unwrap_variations.core.meshdata import make_sample

charts = import_or_skip("unwrap_variations.core.charts")

is_disk = charts.is_disk
chart_graph = charts.chart_graph
merge_charts = charts.merge_charts
tri_islands = charts.tri_islands


# ---------------------------------------------------------------------------
# Flatteners the tests inject
# ---------------------------------------------------------------------------


def plane_flatten(chart) -> np.ndarray:
    """Project a chart onto its own best-fit plane.

    Exact and isometric for a planar chart, which is what makes the expected
    ``stretch_l2`` of the planar fixtures below exactly ``1.0`` rather than
    "small".  On a curved chart it is a genuinely bad parameterisation, which is
    equally useful: it exercises the rejection path with real geometry.
    """
    verts = np.asarray(chart.verts, dtype=np.float64)
    if verts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centred = verts - verts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def anisotropic_flatten(factor: float, *, only_above: int = 2):
    """Return a flattener that stretches any chart above ``only_above`` triangles.

    Scaling ``v`` by ``factor`` gives an analytically known Sander L2 stretch of
    ``sqrt((factor + 1 / factor) / 2)`` once the area normalisation is applied,
    so the ``max_stretch`` gate can be tested against an exact number instead of
    against whatever a real solver happens to produce.
    """

    def flatten(chart) -> np.ndarray:
        uv = plane_flatten(chart)
        if chart.n_tris > only_above:
            uv = uv * np.array([1.0, float(factor)])
        return uv

    return flatten


def expected_stretch(factor: float) -> float:
    """Return the L2 stretch of an anisotropic scale by ``factor`` in one axis."""
    return math.sqrt((factor + 1.0 / factor) / 2.0)


def failing_flatten(chart):
    """A flattener that never solves anything."""
    raise RuntimeError("no solution")


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------


def quad_grid(nx: int, ny: int, *, fold_from: int | None = None, angle_deg: float = 0.0):
    """Return an ``nx`` x ``ny`` grid of unit quads, each split into two triangles.

    Args:
        nx: Columns of quads.
        ny: Rows of quads.
        fold_from: When given, every vertex at ``x >= fold_from`` is rotated about
            the fold line by ``angle_deg``, producing a non-developable crease.
        angle_deg: Fold angle in degrees.

    Returns:
        A :class:`~unwrap_variations.core.meshdata.UVMeshSample` whose
        ``edge_verts`` covers **only the real quad edges** -- the tessellation
        diagonals are deliberately absent, exactly as Blender reports them, so
        anything that computes a Euler characteristic from ``sample.n_edges``
        gets the wrong answer and is caught.
    """
    xs = np.arange(nx + 1, dtype=np.float64)
    ys = np.arange(ny + 1, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx.reshape(-1), gy.reshape(-1), np.zeros(gx.size)], axis=1)

    if fold_from is not None and angle_deg:
        theta = math.radians(angle_deg)
        moved = verts[:, 0] >= float(fold_from)
        dx = verts[moved, 0] - float(fold_from)
        verts[moved, 0] = float(fold_from) + dx * math.cos(theta)
        verts[moved, 2] = dx * math.sin(theta)

    def vid(i: int, j: int) -> int:
        return i * (ny + 1) + j

    tris: list[list[int]] = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append([a, b, c])
            tris.append([a, c, d])

    edges: list[tuple[int, int]] = []
    for i in range(nx + 1):
        for j in range(ny):
            edges.append((vid(i, j), vid(i, j + 1)))
    for i in range(nx):
        for j in range(ny + 1):
            edges.append((vid(i, j), vid(i + 1, j)))

    return make_sample(verts, tris, edge_verts=edges, name="grid%dx%d" % (nx, ny))


def dome_grid(n: int = 4, curvature: float = 0.5):
    """Return an ``n`` x ``n`` quad grid lifted onto a paraboloid.

    Doubly curved, so -- unlike a crease, which is developable and flattens
    almost for free -- it has genuine Gaussian curvature and *cannot* be
    flattened without stretch.  Measured with the planar projection these tests
    inject, at ``curvature=0.5`` one half reads ``stretch_l2 = 1.082`` and the
    whole dome reads ``1.147``, which brackets the bound cleanly.
    """
    flat = quad_grid(n, n)
    verts = np.array(flat.verts, copy=True)
    verts[:, 0] -= n / 2.0
    verts[:, 1] -= n / 2.0
    verts[:, 2] = curvature * (verts[:, 0] ** 2 + verts[:, 1] ** 2)
    return make_sample(verts, flat.tri_verts, edge_verts=flat.edge_verts, name="dome")


def quad_grid_charts(nx: int, ny: int, cut_at: int) -> np.ndarray:
    """Return a chart label per triangle splitting a quad grid at column ``cut_at``."""
    labels = []
    for i in range(nx):
        for _ in range(ny):
            labels.extend([0 if i < cut_at else 1] * 2)
    return np.asarray(labels, dtype=np.int64)


def annulus(segments: int = 12, *, inner: float = 1.0, outer: float = 2.0):
    """Return a flat annulus: a closed ring of quads, each split into triangles."""
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    inner_ring = np.stack(
        [inner * np.cos(angles), inner * np.sin(angles), np.zeros(segments)], axis=1
    )
    outer_ring = np.stack(
        [outer * np.cos(angles), outer * np.sin(angles), np.zeros(segments)], axis=1
    )
    verts = np.concatenate([inner_ring, outer_ring], axis=0)

    tris: list[list[int]] = []
    edges: list[tuple[int, int]] = []
    for s in range(segments):
        t = (s + 1) % segments
        a, b = s, t
        c, d = segments + t, segments + s
        tris.append([a, b, c])
        tris.append([a, c, d])
        edges.append((a, b))
        edges.append((segments + s, segments + t))
        edges.append((a, d))
    return make_sample(verts, tris, edge_verts=edges, name="annulus")


def annulus_charts(segments: int = 12) -> np.ndarray:
    """Return chart ids splitting an annulus into two half-rings, each a disk."""
    half = segments // 2
    return np.asarray(
        [0 if s < half else 1 for s in range(segments) for _ in range(2)], dtype=np.int64
    )


def two_islands():
    """Return two coplanar quads that share no vertex, plus their chart labels."""
    verts = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [5.0, 1.0, 0.0],
    ]
    tris = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    sample = make_sample(verts, tris, name="two_islands")
    return sample, np.asarray([0, 0, 1, 1], dtype=np.int64)


def bowtie():
    """Return two triangles meeting at exactly one vertex.

    ``F=2, E=6, V=5`` gives ``chi = 1``: the Euler number on its own calls this a
    disk.  It is the counterexample that forces the connectedness and
    vertex-manifold conditions in :func:`is_disk`.
    """
    verts = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
    ]
    tris = [[0, 1, 2], [0, 3, 4]]
    return make_sample(verts, tris, name="bowtie")


# ---------------------------------------------------------------------------
# is_disk
# ---------------------------------------------------------------------------


class TestIsDisk:
    """The exact Euler test, and the three conditions it does not cover."""

    def test_single_triangle_is_a_disk(self):
        sample = quad_grid(1, 1)
        assert is_disk(sample, np.array([True, False]))

    def test_two_triangles_sharing_an_edge_are_a_disk(self):
        sample = quad_grid(1, 1)
        assert is_disk(sample, np.array([True, True]))

    def test_uses_the_triangulated_edge_set_not_the_quad_edge_set(self):
        """The 4 diagonals of a 2x2 quad grid must count towards ``E``.

        With them, ``chi = 8 - 16 + 9 = 1``.  Without them -- using the mesh's
        own 12 edges, which is what ``sample.n_edges`` reports -- ``chi`` comes
        out as 5 and a perfectly ordinary square patch is called a non-disk.
        ORGANIC.md section 0.1 warns about exactly this.
        """
        sample = quad_grid(2, 2)
        assert sample.n_edges == 12
        assert is_disk(sample, np.ones(sample.n_tris, dtype=bool))

    def test_annulus_is_not_a_disk(self):
        sample = annulus()
        assert not is_disk(sample, np.ones(sample.n_tris, dtype=bool))

    def test_closed_surface_is_not_a_disk(self):
        sample = make_cube()
        assert not is_disk(sample, np.ones(sample.n_tris, dtype=bool))

    def test_disconnected_union_is_not_a_disk(self):
        sample, _ = two_islands()
        assert not is_disk(sample, np.ones(sample.n_tris, dtype=bool))

    def test_bowtie_has_euler_one_but_is_not_a_disk(self):
        """The case the specified Euler test alone would accept."""
        sample = bowtie()
        topo = charts.chart_topology(sample.tri_verts, np.zeros(2, dtype=np.int64), 1)
        assert int(topo.chi[0]) == 1, "the bowtie's Euler characteristic really is 1"
        assert not is_disk(sample, np.ones(2, dtype=bool))

    def test_empty_selection_is_not_a_disk(self):
        sample = quad_grid(1, 1)
        assert not is_disk(sample, np.zeros(sample.n_tris, dtype=bool))
        assert not is_disk(make_empty(), np.zeros(0, dtype=bool))

    def test_accepts_an_index_array_as_well_as_a_mask(self):
        sample = quad_grid(2, 1)
        assert is_disk(sample, np.array([0, 1]))
        assert is_disk(sample, np.array([0, 1], dtype=np.int32))

    @pytest.mark.parametrize("sample", all_pathological(), ids=lambda s: s.name or "?")
    def test_never_raises(self, sample):
        assert isinstance(is_disk(sample, np.ones(max(sample.n_tris, 0), dtype=bool)), bool)


# ---------------------------------------------------------------------------
# chart_graph
# ---------------------------------------------------------------------------


class TestChartGraph:
    """Adjacency, shared boundary length and per-chart topology."""

    def test_two_charts_share_their_dividing_edge(self):
        sample = quad_grid(2, 1)
        graph = chart_graph(sample, quad_grid_charts(2, 1, 1))
        assert graph.n_charts == 2
        assert graph.pair_a.tolist() == [0]
        assert graph.pair_b.tolist() == [1]
        # The dividing edge is one unit long.
        assert graph.pair_length[0] == pytest.approx(1.0)
        assert graph.seam_length == pytest.approx(1.0)

    def test_shared_length_accumulates_over_a_longer_boundary(self):
        sample = quad_grid(2, 4)
        graph = chart_graph(sample, quad_grid_charts(2, 4, 1))
        assert graph.pair_length[0] == pytest.approx(4.0)

    def test_islands_that_touch_nowhere_are_not_adjacent(self):
        sample, labels = two_islands()
        graph = chart_graph(sample, labels)
        assert graph.n_charts == 2
        assert graph.pair_a.size == 0
        assert graph.seam_length == 0.0

    def test_areas_and_counts_are_per_chart(self):
        sample = quad_grid(2, 1)
        graph = chart_graph(sample, quad_grid_charts(2, 1, 1))
        assert graph.tri_count.tolist() == [2, 2]
        assert graph.area[0] == pytest.approx(1.0)
        assert graph.area[1] == pytest.approx(1.0)

    def test_reports_disk_ness_per_chart(self):
        sample = annulus()
        graph = chart_graph(sample, annulus_charts())
        assert graph.topology.disk.tolist() == [True, True]
        whole = chart_graph(sample, np.zeros(sample.n_tris, dtype=np.int64))
        assert whole.topology.disk.tolist() == [False]
        assert int(whole.topology.chi[0]) == 0

    def test_neighbours_are_sorted_and_symmetric(self):
        sample = quad_grid(3, 1)
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        graph = chart_graph(sample, labels)
        assert graph.neighbours(1) == [(0, pytest.approx(1.0)), (2, pytest.approx(1.0))]

    def test_summary_is_ascii(self):
        sample = quad_grid(2, 1)
        graph = chart_graph(sample, quad_grid_charts(2, 1, 1))
        assert graph.summary().isascii()

    @pytest.mark.parametrize("sample", all_pathological(), ids=lambda s: s.name or "?")
    def test_never_raises(self, sample):
        graph = chart_graph(sample, np.zeros(max(sample.n_tris, 0), dtype=np.int64))
        assert graph.n_charts >= 0


# ---------------------------------------------------------------------------
# tri_islands
# ---------------------------------------------------------------------------


class TestTriIslands:
    """Islands read back from a UV layout, as the seam metric defines them."""

    def test_a_continuous_layout_is_one_island(self):
        sample = make_flat_grid(4)
        assert int(tri_islands(sample).max()) == 0

    def test_a_split_layout_is_two_islands(self):
        sample = quad_grid(2, 1)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        # Give the two quads UVs that agree internally and disagree across the
        # dividing edge.
        uvs[sample.tri_loops[0]] = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]
        uvs[sample.tri_loops[1]] = [[0.0, 0.0], [0.1, 0.1], [0.0, 0.1]]
        uvs[sample.tri_loops[2]] = [[0.5, 0.0], [0.6, 0.0], [0.6, 0.1]]
        uvs[sample.tri_loops[3]] = [[0.5, 0.0], [0.6, 0.1], [0.5, 0.1]]
        labels = tri_islands(sample, uvs=uvs)
        assert labels.tolist() == [0, 0, 1, 1]

    def test_matches_the_chart_graph_seam_length(self):
        sample = quad_grid(2, 1)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        uvs[sample.tri_loops[0]] = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]
        uvs[sample.tri_loops[1]] = [[0.0, 0.0], [0.1, 0.1], [0.0, 0.1]]
        uvs[sample.tri_loops[2]] = [[0.5, 0.0], [0.6, 0.0], [0.6, 0.1]]
        uvs[sample.tri_loops[3]] = [[0.5, 0.0], [0.6, 0.1], [0.5, 0.1]]
        graph = chart_graph(sample, tri_islands(sample, uvs=uvs))
        assert graph.seam_length == pytest.approx(1.0)

    @pytest.mark.parametrize("sample", all_pathological(), ids=lambda s: s.name or "?")
    def test_never_raises(self, sample):
        labels = tri_islands(sample)
        assert labels.shape[0] == sample.n_tris


# ---------------------------------------------------------------------------
# merge_charts -- the hand-built cases from the contract
# ---------------------------------------------------------------------------


class TestMergeAcceptance:
    """Merges that must be taken."""

    def test_two_adjacent_quads_merge(self):
        sample = quad_grid(2, 1)
        labels, report = merge_charts(
            sample, quad_grid_charts(2, 1, 1), flatten=plane_flatten
        )
        assert report.changed
        assert report.chart_count_before == 2
        assert report.chart_count_after == 1
        assert labels.tolist() == [0, 0, 0, 0]
        assert report.seam_length_after == pytest.approx(0.0)
        assert len(report.steps) == 1
        # A planar chart projected onto its own plane is an exact isometry.
        assert report.steps[0].stretch_after == pytest.approx(1.0, abs=1e-9)
        assert report.message.isascii()

    def test_a_row_of_charts_collapses_to_one(self):
        sample = quad_grid(4, 2)
        labels = np.repeat(np.arange(4, dtype=np.int64), 4)
        out, report = merge_charts(sample, labels, flatten=plane_flatten)
        assert report.chart_count_after == 1
        assert int(np.unique(out).size) == 1
        assert report.seam_length_after == pytest.approx(0.0)

    def test_the_merged_chart_is_still_a_disk(self):
        sample = quad_grid(3, 3)
        labels = np.repeat(np.arange(3, dtype=np.int64), 6)
        out, _ = merge_charts(sample, labels, flatten=plane_flatten)
        for chart in np.unique(out):
            assert is_disk(sample, out == chart)


class TestMergeRejection:
    """Merges that must be refused, one gate at a time."""

    def test_an_annulus_is_never_formed(self):
        sample = annulus()
        labels, report = merge_charts(sample, annulus_charts(), flatten=plane_flatten)
        assert not report.changed
        assert report.rejected_not_disk >= 1
        assert labels.tolist() == annulus_charts().tolist()
        assert report.seam_length_after == report.seam_length_before

    def test_charts_with_a_disconnected_union_never_merge(self):
        sample, labels = two_islands()
        out, report = merge_charts(sample, labels, flatten=plane_flatten)
        assert not report.changed
        assert out.tolist() == labels.tolist()
        # They are not even adjacent, so there is nothing to consider.
        assert "nothing adjacent" in report.message

    def test_a_merge_past_the_stretch_bound_is_refused(self):
        """The gate is tested against an analytically known stretch."""
        factor = 3.0
        assert expected_stretch(factor) == pytest.approx(1.29099, abs=1e-5)
        sample = quad_grid(2, 1)
        labels = quad_grid_charts(2, 1, 1)

        _, tight = merge_charts(
            sample, labels, max_stretch=1.25, flatten=anisotropic_flatten(factor)
        )
        assert not tight.changed
        assert tight.rejected_stretch == 1

        _, loose = merge_charts(
            sample, labels, max_stretch=1.35, flatten=anisotropic_flatten(factor)
        )
        assert loose.changed
        assert loose.steps[0].stretch_after == pytest.approx(expected_stretch(factor), abs=1e-6)

    def test_real_curvature_is_refused_at_a_tight_bound(self):
        """A doubly curved pair genuinely cannot be flattened cheaply.

        A crease is the wrong test here: a folded pair of quads is still
        developable, and a best-fit-plane projection of a 90-degree fold measures
        only 1.030.  Gaussian curvature is what actually costs stretch, so the
        fixture is a paraboloid.
        """
        sample = dome_grid(4, curvature=0.5)
        labels = quad_grid_charts(4, 4, 2)
        _, tight = merge_charts(sample, labels, max_stretch=1.10, flatten=plane_flatten)
        assert not tight.changed
        assert tight.rejected_stretch == 1
        _, loose = merge_charts(sample, labels, max_stretch=1.20, flatten=plane_flatten)
        assert loose.changed
        assert 1.10 < loose.steps[0].stretch_after <= 1.20
        assert loose.steps[0].stretch_after > loose.steps[0].stretch_before

    def test_a_failing_flattener_merges_nothing(self):
        sample = quad_grid(2, 1)
        labels = quad_grid_charts(2, 1, 1)
        out, report = merge_charts(sample, labels, flatten=failing_flatten)
        assert not report.changed
        assert report.rejected_flatten >= 1
        assert out.tolist() == labels.tolist()

    def test_a_non_injective_flattening_is_refused(self):
        """A flattener that folds the chart over itself must not be trusted.

        ``flipped_tris`` alone cannot see this class of defect (ORGANIC.md
        section 0.3), so the certifier has to test the boundary curve.
        """

        def folded(chart):
            uv = plane_flatten(chart)
            if chart.n_tris > 2:
                # Reflect one half onto the other: locally injective everywhere
                # it is not degenerate, globally a mess.
                out = uv.copy()
                far = out[:, 0] > 0.0
                out[far, 0] = -out[far, 0] - 0.15
                return out
            return uv

        sample = quad_grid(2, 1)
        _, report = merge_charts(
            sample, quad_grid_charts(2, 1, 1), max_stretch=99.0, flatten=folded
        )
        assert not report.changed
        assert report.rejected_injectivity + report.rejected_flatten >= 1

    def test_min_gain_holds_back_a_short_boundary(self):
        sample = quad_grid(2, 1)
        labels = quad_grid_charts(2, 1, 1)
        _, report = merge_charts(sample, labels, min_gain=2.0, flatten=plane_flatten)
        assert not report.changed


class TestMergeDisabled:
    """The escape hatches: no flattener, and merging switched off."""

    def test_max_stretch_below_one_disables_merging(self):
        """``stretch_l2 >= 1.0`` is a theorem, so a bound below it can never pass.

        This is what keeps the *unmerged* layout reachable from the parameter
        grid, which ORGANIC.md section 5 requires so a benchmark can attribute
        the gain to merging rather than to everything at once.
        """
        sample = quad_grid(2, 1)
        labels = quad_grid_charts(2, 1, 1)
        out, report = merge_charts(sample, labels, max_stretch=0.5, flatten=plane_flatten)
        assert not report.changed
        assert report.flatten_calls == 0
        assert out.tolist() == labels.tolist()
        assert "disabled" in report.message

    def test_no_flattener_returns_the_input_untouched(self, monkeypatch):
        monkeypatch.setattr(charts, "_resolve_flatten", lambda _: None)
        sample = quad_grid(2, 1)
        labels = quad_grid_charts(2, 1, 1)
        out, report = merge_charts(sample, labels)
        assert not report.changed
        assert out.tolist() == labels.tolist()
        assert "no flattener" in report.message

    def test_a_single_chart_has_nothing_to_do(self):
        sample = quad_grid(2, 1)
        labels = np.zeros(sample.n_tris, dtype=np.int64)
        out, report = merge_charts(sample, labels, flatten=plane_flatten)
        assert not report.changed
        assert out.tolist() == labels.tolist()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def random_partitions():
    """Yield ``(sample, labels)`` partitions of a grid into disk-shaped charts.

    Charts are built from a seeded region-growing pass over the quad grid, so
    every input chart is connected and simply connected, which is the
    precondition for asserting that every *output* chart is a disk.
    """
    out = []
    for seed in (1, 7, 42, 1234):
        nx = ny = 4
        sample = quad_grid(nx, ny)
        rng = np.random.default_rng(seed)
        # Assign each quad column-major to one of a few vertical strips, then
        # perturb the strip boundaries: still contiguous, still simply connected.
        cuts = np.sort(rng.choice(np.arange(1, nx), size=2, replace=False))
        labels = []
        for i in range(nx):
            band = int(np.searchsorted(cuts, i, side="right"))
            labels.extend([band] * (2 * ny))
        out.append((sample, np.asarray(labels, dtype=np.int64)))
    return out


class TestProperties:
    """Invariants that must hold for every input, not just the fixtures."""

    @pytest.mark.parametrize("case", random_partitions(), ids=lambda c: c[0].name)
    def test_seam_length_is_monotonically_non_increasing(self, case):
        sample, labels = case
        _, report = merge_charts(sample, labels, flatten=plane_flatten)
        assert report.seam_length_after <= report.seam_length_before + 1e-9
        running = report.seam_length_before
        for step in report.steps:
            assert step.seam_length_after <= running + 1e-9
            running = step.seam_length_after
        assert report.seam_length_after == pytest.approx(running)

    @pytest.mark.parametrize("case", random_partitions(), ids=lambda c: c[0].name)
    def test_chart_count_is_non_increasing(self, case):
        sample, labels = case
        out, report = merge_charts(sample, labels, flatten=plane_flatten)
        assert report.chart_count_after <= report.chart_count_before
        assert int(np.unique(out).size) == report.chart_count_after

    @pytest.mark.parametrize("case", random_partitions(), ids=lambda c: c[0].name)
    def test_every_output_chart_is_a_disk(self, case):
        sample, labels = case
        out, _ = merge_charts(sample, labels, flatten=plane_flatten)
        for chart in np.unique(out):
            assert is_disk(sample, out == chart)

    @pytest.mark.parametrize("case", random_partitions(), ids=lambda c: c[0].name)
    def test_the_labelling_partitions_every_triangle(self, case):
        sample, labels = case
        out, _ = merge_charts(sample, labels, flatten=plane_flatten)
        assert out.shape[0] == sample.n_tris
        assert int(out.min()) == 0
        assert set(np.unique(out).tolist()) == set(range(int(np.unique(out).size)))

    def test_no_merge_ever_raises_the_measured_seam_length(self):
        """Cross-check the report against a fresh graph over the merged labels."""
        sample = quad_grid(4, 3)
        labels = np.repeat(np.arange(4, dtype=np.int64), 6)
        out, report = merge_charts(sample, labels, flatten=plane_flatten)
        after = chart_graph(sample, out).seam_length
        assert after == pytest.approx(report.seam_length_after, abs=1e-9)
        assert after <= chart_graph(sample, labels).seam_length + 1e-9

    @pytest.mark.parametrize("sample", all_pathological(), ids=lambda s: s.name or "?")
    def test_never_raises(self, sample):
        labels = np.zeros(max(sample.n_tris, 0), dtype=np.int64)
        out, report = merge_charts(sample, labels, flatten=plane_flatten)
        assert out.shape[0] == max(sample.n_tris, 0)
        assert report.message.isascii()


class TestScreening:
    """The cheap screening tier, which must never widen the gate."""

    def test_the_screening_budget_reaches_the_flattener(self):
        seen = []

        def recording(chart, *, iters=None):
            seen.append(iters)
            return plane_flatten(chart)

        sample = quad_grid(3, 2)
        labels = np.repeat(np.arange(3, dtype=np.int64), 4)
        _, report = merge_charts(
            sample, labels, flatten=recording, screen_iters=5, uvs=None
        )
        assert report.changed
        assert 5 in seen, "screening must ask for the cheap budget"
        assert None in seen, "an accepted merge must be re-flattened at full quality"

    def test_the_default_budget_tracks_the_flatteners_own(self, monkeypatch):
        """A pinned constant already went stale here once, silently.

        The screening budget was 8 against a flattener defaulting to 32.  When
        that default moved to 256, 8 iterations left every candidate over the
        distortion bound and Suzanne merged nothing at all -- while still paying
        for the screening that found out.  The budget is a ratio for that reason.
        """
        flatten_mod = import_or_skip("unwrap_variations.core.flatten")
        monkeypatch.setattr(flatten_mod, "_DEFAULT_ITERS", 256, raising=False)
        assert charts._screen_iters() == 32
        monkeypatch.setattr(flatten_mod, "_DEFAULT_ITERS", 32, raising=False)
        assert charts._screen_iters() == 16
        monkeypatch.setattr(flatten_mod, "_DEFAULT_ITERS", "nonsense", raising=False)
        assert charts._screen_iters() == charts._SCREEN_ITERS_FALLBACK

    def test_a_flattener_without_an_iters_argument_still_works(self):
        """An injected stub must not have to grow an argument it does not use."""
        sample = quad_grid(2, 1)
        _, report = merge_charts(
            sample, quad_grid_charts(2, 1, 1), flatten=plane_flatten, screen_iters=4
        )
        assert report.changed

    def test_screening_off_is_still_a_valid_run(self):
        sample = quad_grid(3, 2)
        labels = np.repeat(np.arange(3, dtype=np.int64), 4)
        _, report = merge_charts(
            sample, labels, flatten=plane_flatten, screen_iters=None
        )
        assert report.chart_count_after == 1

    def test_a_screened_reject_stays_rejected(self):
        """The cheap tier may only ever reject; it must not admit anything."""

        def coarse(chart, *, iters=None):
            uv = plane_flatten(chart)
            if chart.n_tris > 2:
                # The cheap pass is deliberately the worse one, as a real
                # truncated descent is.
                return uv * np.array([1.0, 3.0 if iters else 1.05])
            return uv

        sample = quad_grid(2, 1)
        labels = quad_grid_charts(2, 1, 1)
        _, report = merge_charts(
            sample, labels, max_stretch=1.25, flatten=coarse, screen_iters=8
        )
        assert not report.changed
        assert report.rejected_stretch == 1


class TestTimeBudget:
    """The work cap, which may cost seam length but never a gate."""

    def test_a_zero_budget_screens_nothing_and_merges_nothing(self):
        sample = quad_grid(4, 2)
        labels = np.repeat(np.arange(4, dtype=np.int64), 4)
        out, report = merge_charts(
            sample, labels, flatten=plane_flatten, time_budget=1e-9
        )
        assert not report.changed
        assert report.out_of_time > 0
        assert report.flatten_calls == 0
        assert out.tolist() == labels.tolist()

    def test_an_ample_budget_changes_nothing(self):
        sample = quad_grid(4, 2)
        labels = np.repeat(np.arange(4, dtype=np.int64), 4)
        free, _ = merge_charts(sample, labels, flatten=plane_flatten)
        capped, report = merge_charts(
            sample, labels, flatten=plane_flatten, time_budget=600.0
        )
        assert free.tolist() == capped.tolist()
        assert report.out_of_time == 0

    def test_past_the_deadline_nothing_new_is_flattened(self):
        """Not even the full-quality polish on an accepted merge.

        A budget that only stopped screening would still pay one full solve per
        accepted merge, which is what made a 3 s budget cost 8 s on Suzanne.
        """
        calls = []

        def recording(chart, *, iters=None):
            calls.append(iters)
            return plane_flatten(chart)

        sample = quad_grid(4, 2)
        labels = np.repeat(np.arange(4, dtype=np.int64), 4)
        _, free = merge_charts(sample, labels, flatten=recording, screen_iters=5)
        assert None in calls, "unbudgeted runs do polish accepted merges"

        calls.clear()
        # Long enough to screen, far too short to have any time left over.
        _, capped = merge_charts(
            sample, labels, flatten=recording, screen_iters=5, time_budget=1e-9
        )
        assert None not in calls
        assert capped.out_of_time > 0
        del free

    def test_a_budgeted_run_is_a_prefix_of_the_unbudgeted_one(self):
        """Whatever it does take must be what greedy would have taken first."""
        sample = quad_grid(4, 2)
        labels = np.repeat(np.arange(4, dtype=np.int64), 4)
        _, full = merge_charts(sample, labels, flatten=plane_flatten)
        _, capped = merge_charts(
            sample, labels, flatten=plane_flatten, time_budget=600.0, max_merges=1
        )
        assert len(capped.steps) <= len(full.steps)
        taken = [s.members for s in capped.steps]
        assert taken == [s.members for s in full.steps[: len(taken)]]
        assert capped.seam_length_after >= full.seam_length_after - 1e-9


class TestLayoutBaseline:
    """The free benefit-ratio denominator read off the incoming layout."""

    def test_it_reproduces_the_evaluator_on_a_single_chart(self):
        """The accumulators must agree with the frozen metric they stand in for."""
        evaluator = import_or_skip("unwrap_variations.core.evaluator")
        sample = dome_grid(4, curvature=0.5)
        rng = np.random.default_rng(3)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        # A perturbed projection: real distortion, nothing degenerate.
        flat = sample.verts[:, :2] + 0.05 * rng.normal(size=(sample.n_verts, 2))
        for tri in range(sample.n_tris):
            uvs[sample.tri_loops[tri]] = flat[sample.tri_verts[tri]]
        labels = np.zeros(sample.n_tris, dtype=np.int64)
        accum = charts._LayoutStretch(sample, uvs, labels, 1)
        expected = evaluator.evaluate(sample.with_uvs(uvs), compute_overlap=False)
        assert accum.of([0]) == pytest.approx(expected.stretch_l2, rel=1e-12)

    def test_it_is_additive_over_charts(self):
        """A union's stretch must equal the whole-layout figure, exactly."""
        evaluator = import_or_skip("unwrap_variations.core.evaluator")
        sample = dome_grid(4, curvature=0.5)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        for tri in range(sample.n_tris):
            uvs[sample.tri_loops[tri]] = sample.verts[sample.tri_verts[tri]][:, :2]
        labels = quad_grid_charts(4, 4, 2)
        accum = charts._LayoutStretch(sample, uvs, labels, 2)
        whole = charts._LayoutStretch(sample, uvs, np.zeros(sample.n_tris, np.int64), 1)
        assert accum.of([0, 1]) == pytest.approx(whole.of([0]), rel=1e-12)
        expected = evaluator.evaluate(sample.with_uvs(uvs), compute_overlap=False)
        assert accum.of([0, 1]) == pytest.approx(expected.stretch_l2, rel=1e-12)

    def test_it_never_reports_below_the_isometric_floor(self):
        sample = quad_grid(2, 1)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        accum = charts._LayoutStretch(sample, uvs, np.zeros(sample.n_tris, np.int64), 1)
        assert accum.of([0]) >= 1.0
        assert accum.of([]) >= 1.0

    def test_supplying_a_layout_removes_the_baseline_flattens(self):
        """With a layout to read, no chart is flattened just to be a denominator."""
        sample = quad_grid(4, 2)
        labels = np.repeat(np.arange(4, dtype=np.int64), 4)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        for tri in range(sample.n_tris):
            uvs[sample.tri_loops[tri]] = sample.verts[sample.tri_verts[tri]][:, :2] * 0.1
        _, without = merge_charts(sample, labels, flatten=plane_flatten)
        _, with_layout = merge_charts(sample, labels, flatten=plane_flatten, uvs=uvs)
        assert with_layout.flatten_calls < without.flatten_calls


class TestDeterminism:
    """Same input, same merges -- otherwise nothing can be benchmarked."""

    def test_the_merge_sequence_repeats_exactly(self):
        sample = quad_grid(4, 3)
        labels = np.repeat(np.arange(6, dtype=np.int64), 4)
        first_labels, first = merge_charts(sample, labels, flatten=plane_flatten)
        second_labels, second = merge_charts(sample, labels, flatten=plane_flatten)
        assert first_labels.tolist() == second_labels.tolist()
        assert [s.members for s in first.steps] == [s.members for s in second.steps]
        assert [s.kept for s in first.steps] == [s.kept for s in second.steps]
        assert first.seam_length_after == second.seam_length_after

    def test_relabelling_the_input_does_not_change_the_partition(self):
        """Chart ids are names, not data: renaming them must not move a seam."""
        sample = quad_grid(4, 3)
        labels = np.repeat(np.arange(4, dtype=np.int64), 6)
        renamed = labels * 10 + 5
        out_a, report_a = merge_charts(sample, labels, flatten=plane_flatten)
        out_b, report_b = merge_charts(sample, renamed, flatten=plane_flatten)
        assert out_a.tolist() == out_b.tolist()
        assert report_a.seam_length_after == pytest.approx(report_b.seam_length_after)


# ---------------------------------------------------------------------------
# The rebuilt layout
# ---------------------------------------------------------------------------


class TestRebuiltLayout:
    """``report.uvs`` -- what the repack and the evaluator actually see."""

    def build(self):
        sample = quad_grid(2, 1)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        uvs[sample.tri_loops[0]] = [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4]]
        uvs[sample.tri_loops[1]] = [[0.0, 0.0], [0.4, 0.4], [0.0, 0.4]]
        uvs[sample.tri_loops[2]] = [[0.6, 0.0], [1.0, 0.0], [1.0, 0.4]]
        uvs[sample.tri_loops[3]] = [[0.6, 0.0], [1.0, 0.4], [0.6, 0.4]]
        return sample, uvs

    def test_merging_removes_the_seam_from_the_layout(self):
        sample, uvs = self.build()
        labels = tri_islands(sample, uvs=uvs)
        assert int(labels.max()) == 1
        _, report = merge_charts(sample, labels, flatten=plane_flatten, uvs=uvs)
        assert report.uvs is not None
        after = tri_islands(sample, uvs=report.uvs)
        assert int(after.max()) == 0
        assert chart_graph(sample, after).seam_length == pytest.approx(0.0)

    def test_the_rebuilt_layout_preserves_uv_area(self):
        sample, uvs = self.build()
        labels = tri_islands(sample, uvs=uvs)
        _, report = merge_charts(sample, labels, flatten=plane_flatten, uvs=uvs)
        before = float(np.abs(_signed_uv_area(sample, uvs)).sum())
        after = float(np.abs(_signed_uv_area(sample, report.uvs)).sum())
        assert after == pytest.approx(before, rel=1e-9)

    def test_uv_area_survives_a_quad_mesh_where_loops_are_shared(self):
        """``loop_triangles`` reuses loops within an n-gon; the rebuild must cope.

        Blender tessellates a quad into two triangles that share two of their
        four loops.  Anything that maps a loop back to "the" triangle owning it
        therefore loses half a quad mesh's area -- and Suzanne is a quad mesh.
        """
        sample = quad_grid(2, 1)
        # One loop per *vertex*, so both triangles of a quad share their loops.
        loops = sample.tri_verts.astype(np.int32)
        shared = make_sample(
            sample.verts,
            sample.tri_verts,
            tri_loops=loops,
            uvs=sample.verts[:, :2] * 0.25,
            edge_verts=sample.edge_verts,
            name="shared_loops",
        )
        before = float(np.abs(_signed_uv_area(shared, shared.uvs)).sum())
        _, report = merge_charts(
            shared, quad_grid_charts(2, 1, 1), flatten=plane_flatten, uvs=shared.uvs
        )
        assert report.changed
        after = float(np.abs(_signed_uv_area(shared, report.uvs)).sum())
        assert after == pytest.approx(before, rel=1e-9)

    def test_the_rebuilt_layout_has_no_flipped_triangles(self):
        sample, uvs = self.build()
        labels = tri_islands(sample, uvs=uvs)
        _, report = merge_charts(sample, labels, flatten=plane_flatten, uvs=uvs)
        assert float(_signed_uv_area(sample, report.uvs).min()) > 0.0

    def test_untouched_charts_stay_bit_identical(self):
        sample = quad_grid(3, 1)
        uvs = np.zeros((sample.n_loops, 2), dtype=np.float64)
        for tri in range(sample.n_tris):
            uvs[sample.tri_loops[tri]] = [[0.1 * tri, 0.0], [0.1 * tri + 0.05, 0.0],
                                          [0.1 * tri, 0.05]]
        labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        _, report = merge_charts(
            sample, labels, flatten=plane_flatten, uvs=uvs, max_merges=1
        )
        assert len(report.steps) == 1
        untouched = set(range(sample.n_tris)) - {
            int(t) for step in report.steps for t in np.flatnonzero(
                np.isin(labels, list(step.members))
            )
        }
        for tri in untouched:
            loops = sample.tri_loops[tri]
            assert np.array_equal(report.uvs[loops], uvs[loops])


def _signed_uv_area(sample, uvs) -> np.ndarray:
    """Return the ``(T,)`` signed UV areas of a layout, for the assertions above."""
    q = np.asarray(uvs, dtype=np.float64)[np.asarray(sample.tri_loops, dtype=np.int64)]
    e1 = q[:, 1] - q[:, 0]
    e2 = q[:, 2] - q[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
