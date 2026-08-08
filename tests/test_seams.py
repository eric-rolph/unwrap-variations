"""Tests for ``unwrap_variations.backends.seams`` -- where to cut.

Seams decide what the unwrap can possibly look like, and they are computed from
geometry alone, so the heuristics are pure numpy and fully testable here.

The properties that matter:

* **Scale invariance.**  A model imported in centimetres and the same model in
  metres are the same model.  A curvature threshold that fires on one and not
  the other is measuring units, not shape.
* **Boundary edges are never seams.**  An edge with one incident face is
  already a chart boundary; marking it as a seam is meaningless and confuses
  every downstream solver.
* **Non-manifold input must not raise.**  Real production meshes have edges with
  three incident faces.  Blender lets you make them, so the add-on has to
  survive them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.conftest import (
    CORPUS_NAMES,
    import_or_skip,
    load_corpus,
    make_all_identical,
    make_cube,
    make_empty,
    make_flat_grid,
    make_nan_uvs,
    make_random_mesh,
    make_unit_triangle,
)

seams = import_or_skip("unwrap_variations.backends.seams")

compute_seams = seams.compute_seams
SEAM_STRATEGIES = seams.SEAM_STRATEGIES

#: Strategies that derive cuts from geometry rather than echoing the input.
COMPUTED = ("SHARP", "ANGLE", "CURVATURE", "DUAL_GRAPH")


def boundary_mask(sample) -> np.ndarray:
    """Return ``(E,)`` bool marking edges with fewer than two incident triangles.

    Computed here independently of the module under test, so a bug in the
    module's own topology pass cannot make the boundary assertions vacuous.
    """
    if sample.n_edges == 0:
        return np.zeros(0, dtype=bool)
    tri_pairs = np.concatenate(
        [
            sample.tri_verts[:, [0, 1]],
            sample.tri_verts[:, [1, 2]],
            sample.tri_verts[:, [2, 0]],
        ],
        axis=0,
    ).astype(np.int64)
    base = int(max(sample.n_verts, 1))
    tri_keys = np.sort(tri_pairs, axis=1)
    tri_keys = tri_keys[:, 0] * base + tri_keys[:, 1]
    edge_keys = np.sort(sample.edge_verts.astype(np.int64), axis=1)
    edge_keys = edge_keys[:, 0] * base + edge_keys[:, 1]
    counts = np.zeros(sample.n_edges, dtype=np.int64)
    order = np.argsort(edge_keys)
    sorted_keys = edge_keys[order]
    pos = np.searchsorted(sorted_keys, tri_keys)
    safe = np.minimum(pos, sorted_keys.size - 1)
    valid = (pos < sorted_keys.size) & (sorted_keys[safe] == tri_keys)
    np.add.at(counts, order[pos[valid]], 1)
    return counts < 2


# ---------------------------------------------------------------------------
# Contract shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", SEAM_STRATEGIES)
@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_returns_a_bool_mask_of_edge_length(strategy, name):
    """Every strategy returns a fresh ``(E,)`` bool array on every real mesh."""
    sample = load_corpus(name)
    mask = compute_seams(sample, strategy, angle_deg=30.0, curvature_threshold=0.35)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (sample.n_edges,)
    assert mask is not sample.edge_seam


def test_strategy_list_matches_the_params_module():
    """The two copies of the strategy list must not drift apart."""
    from unwrap_variations.core.params import SEAM_STRATEGIES as PARAM_STRATEGIES

    assert tuple(SEAM_STRATEGIES) == tuple(PARAM_STRATEGIES)


def test_none_marks_nothing():
    """``NONE`` is the "unwrap it as one sheet" control: no cuts at all."""
    sample = load_corpus("hardsurface")
    assert not compute_seams(sample, "NONE").any()


def test_existing_passes_the_input_through():
    """``EXISTING`` respects whatever the artist already marked."""
    sample = load_corpus("hardsurface")
    wanted = np.zeros(sample.n_edges, dtype=bool)
    interior = ~boundary_mask(sample)
    wanted[np.nonzero(interior)[0][::5]] = True
    seamed = sample.with_seams(wanted)
    np.testing.assert_array_equal(compute_seams(seamed, "EXISTING"), wanted)


def test_existing_prefers_the_explicit_argument():
    """The ``existing`` argument overrides the mesh's own seam flags."""
    sample = load_corpus("hardsurface")
    override = np.zeros(sample.n_edges, dtype=bool)
    override[np.nonzero(~boundary_mask(sample))[0][:3]] = True
    got = compute_seams(sample, "EXISTING", existing=override)
    np.testing.assert_array_equal(got, override)


def test_unknown_strategy_falls_back_without_raising():
    """A typo must not clear the artist's seams or invent new ones."""
    sample = load_corpus("hardsurface")
    wanted = np.zeros(sample.n_edges, dtype=bool)
    wanted[np.nonzero(~boundary_mask(sample))[0][:4]] = True
    seamed = sample.with_seams(wanted)
    got = compute_seams(seamed, "NOT_A_STRATEGY")
    np.testing.assert_array_equal(got, wanted)


@pytest.mark.parametrize("strategy", ["angle", "Angle", "  CURVATURE  "])
def test_strategy_names_are_case_and_space_insensitive(strategy):
    """A preset string from a ``.blend`` may arrive in any case."""
    sample = load_corpus("hardsurface")
    mask = compute_seams(sample, strategy, angle_deg=30.0)
    assert mask.shape == (sample.n_edges,)


# ---------------------------------------------------------------------------
# Boundary edges.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", COMPUTED)
def test_boundary_edges_are_never_seams_on_a_grid(strategy):
    """An open plane's rim is already a chart boundary; it is never cut."""
    sample = make_flat_grid(8)
    rim = boundary_mask(sample)
    assert rim.any(), "the grid fixture should have a boundary"
    mask = compute_seams(sample, strategy, angle_deg=5.0, curvature_threshold=0.02)
    assert not (mask & rim).any(), (
        "%s marked %d boundary edges as seams" % (strategy, int((mask & rim).sum()))
    )


@pytest.mark.parametrize("strategy", COMPUTED)
def test_boundary_edges_are_never_seams_on_a_single_triangle(strategy):
    """Every edge of a lone triangle is a boundary, so nothing can be cut."""
    mask = compute_seams(make_unit_triangle(), strategy, angle_deg=1.0)
    assert not mask.any()


@pytest.mark.parametrize("strategy", COMPUTED)
def test_boundary_edges_are_never_seams_on_the_corpus(strategy):
    """The same guarantee, on real Blender meshes with real open boundaries."""
    for name in ("hardsurface", "nonmanifold"):
        sample = load_corpus(name)
        rim = boundary_mask(sample)
        mask = compute_seams(sample, strategy, angle_deg=20.0, curvature_threshold=0.1)
        assert not (mask & rim).any(), (
            "%s on %s marked %d boundary edges" % (strategy, name, int((mask & rim).sum()))
        )


# ---------------------------------------------------------------------------
# Scale invariance -- the property that catches unit bugs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.01, 100.0, 1e4])
@pytest.mark.parametrize("name", ["hardsurface", "sphere_smart", "cube_project"])
def test_curvature_is_scale_invariant(factor, name):
    """Scaling the mesh by 100 must give the byte-identical seam set.

    Discrete mean curvature has units of 1/length, so an implementation that
    compares a raw curvature against a dimensionless threshold silently changes
    its mind when the same model is imported in centimetres instead of metres.
    """
    sample = load_corpus(name)
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    base = compute_seams(sample, "CURVATURE", curvature_threshold=0.35)
    got = compute_seams(scaled, "CURVATURE", curvature_threshold=0.35)
    np.testing.assert_array_equal(
        got,
        base,
        err_msg=(
            "%s scaled by %g changed the CURVATURE seam set (%d -> %d edges); "
            "the threshold is being compared against a value that carries units"
            % (name, factor, int(base.sum()), int(got.sum()))
        ),
    )


@pytest.mark.parametrize("strategy", ["SHARP", "ANGLE", "CURVATURE"])
@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_threshold_strategies_are_bit_identical_under_scale(strategy, factor):
    """Cuts depend on shape, never on the unit the model happens to be in.

    The threshold strategies compare a dimensionless quantity against a
    dimensionless constant, so their answer must be identical bit for bit.
    """
    sample = load_corpus("hardsurface")
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    base = compute_seams(sample, strategy, angle_deg=30.0, curvature_threshold=0.35)
    got = compute_seams(scaled, strategy, angle_deg=30.0, curvature_threshold=0.35)
    np.testing.assert_array_equal(
        got, base, err_msg="%s changed under a %g scale" % (strategy, factor)
    )


@pytest.mark.parametrize("factor", [1e-3, 0.01, 100.0, 1e3, 1e6])
def test_dual_graph_is_scale_invariant_up_to_tie_breaking(factor):
    """The dual-graph cut *count* is exactly scale invariant.

    ``DUAL_GRAPH`` takes a maximum spanning tree over dual-edge weights derived
    from face normals.  Normalising the cross products of scaled coordinates
    perturbs those weights in the last bit (measured: 4.4e-16), and a mesh full
    of coplanar faces has many exactly-tied weights, so which of two equally
    good cuts the tree picks can flip.  That is a different tree of the same
    size, not a different answer: the cut count is identical at every scale, and
    only a handful of edges out of hundreds ever move.
    """
    sample = load_corpus("hardsurface")
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    base = compute_seams(sample, "DUAL_GRAPH")
    got = compute_seams(scaled, "DUAL_GRAPH")
    assert int(got.sum()) == int(base.sum()), (
        "DUAL_GRAPH cut %d edges at scale %g against %d at scale 1 -- that is a "
        "units bug, not a tie-break" % (int(got.sum()), factor, int(base.sum()))
    )
    moved = int((got != base).sum())
    assert moved <= max(4, sample.n_edges // 100), (
        "DUAL_GRAPH moved %d of %d cuts under a %g scale, far more than float "
        "tie-breaking explains" % (moved, sample.n_edges, factor)
    )


def test_seams_are_independent_of_uvs():
    """Seam heuristics read geometry only; a different UV layer changes nothing."""
    sample = load_corpus("hardsurface")
    rotated = sample.with_uvs(sample.uvs[:, ::-1] * 3.0 + 0.7)
    for strategy in COMPUTED:
        np.testing.assert_array_equal(
            compute_seams(sample, strategy),
            compute_seams(rotated, strategy),
            err_msg="%s reacted to a UV change" % strategy,
        )


# ---------------------------------------------------------------------------
# ANGLE.
# ---------------------------------------------------------------------------


def test_angle_cuts_a_cube_at_its_hard_edges():
    """A cube folds by exactly 90 degrees at its 12 real edges and nowhere else.

    The triangulated cube also has 6 face diagonals, which are perfectly flat.
    A 45-degree threshold must cut the 12 folds and leave the 6 diagonals alone;
    a 91-degree threshold must cut nothing.  There is no ambiguity in either
    number, which is what makes the cube the right shape for this test.
    """
    cube = make_cube()
    interior = ~boundary_mask(cube)
    assert int(interior.sum()) == 18  # 12 cube edges + 6 tessellation diagonals

    sharp = compute_seams(cube, "ANGLE", angle_deg=45.0)
    assert int(sharp.sum()) == 12, (
        "expected the 12 folded edges of a cube to be cut at 45 degrees, got %d"
        % int(sharp.sum())
    )
    # Every cut edge is axis aligned: a real cube edge, never a face diagonal.
    for a, b in cube.edge_verts[sharp]:
        delta = np.abs(cube.verts[a] - cube.verts[b])
        assert int((delta > 1e-12).sum()) == 1, (
            "ANGLE cut a face diagonal, which is flat, not a fold"
        )

    blunt = compute_seams(cube, "ANGLE", angle_deg=91.0)
    assert not blunt.any(), "a 91-degree threshold cut a 90-degree fold"


def test_angle_threshold_is_monotone():
    """Raising the angle threshold can only remove cuts, never add them."""
    sample = load_corpus("hardsurface")
    previous = None
    for angle in (5.0, 20.0, 45.0, 80.0, 120.0):
        mask = compute_seams(sample, "ANGLE", angle_deg=angle)
        if previous is not None:
            assert not (mask & ~previous).any(), (
                "raising the threshold to %g degrees added a seam" % angle
            )
        previous = mask


def test_angle_marks_nothing_on_a_flat_sheet():
    """A perfectly flat grid has no folds, so no angle-based cuts."""
    assert not compute_seams(make_flat_grid(8), "ANGLE", angle_deg=1.0).any()


def test_curvature_threshold_is_monotone():
    """Raising the curvature threshold can only remove cuts."""
    sample = load_corpus("sphere_smart")
    previous = None
    for threshold in (0.02, 0.1, 0.35, 0.7, 0.95):
        mask = compute_seams(sample, "CURVATURE", curvature_threshold=threshold)
        if previous is not None:
            assert not (mask & ~previous).any(), (
                "raising the curvature threshold to %g added a seam" % threshold
            )
        previous = mask


# ---------------------------------------------------------------------------
# DUAL_GRAPH.
# ---------------------------------------------------------------------------


def test_dual_graph_leaves_a_flat_disk_uncut():
    """A grid is already a disk.  It needs no cuts, so it must get none.

    Cutting a shape that is already flattenable is pure damage: every cut is a
    texture discontinuity the artist then has to paint across.
    """
    mask = compute_seams(make_flat_grid(10), "DUAL_GRAPH", curvature_threshold=0.35)
    assert int(mask.sum()) == 0, (
        "DUAL_GRAPH cut %d edges of a flat grid, which unwraps perfectly with "
        "none" % int(mask.sum())
    )


def test_dual_graph_opens_a_closed_surface():
    """A sphere is closed; without cuts it cannot be flattened at all."""
    mask = compute_seams(load_corpus("sphere_smart"), "DUAL_GRAPH")
    assert mask.any(), "DUAL_GRAPH left a closed sphere with no cuts"


@pytest.mark.parametrize("name", ["cube_project", "sphere_smart"])
def test_dual_graph_cut_set_is_a_spanning_tree(name):
    """On a closed genus-0 surface the cuts form a spanning tree of the vertices.

    This is the textbook result and it pins the strategy exactly: the complement
    of a spanning tree of the *dual* graph is a spanning tree of the *primal*
    graph.  So the cut set must have exactly ``V - 1`` edges, contain no cycle,
    and touch every vertex.  Too few edges and the surface does not open; a
    cycle means a whole island was cut adrift for nothing.
    """
    from tests.reference import UnionFind

    sample = load_corpus(name)
    mask = compute_seams(sample, "DUAL_GRAPH")
    cut_edges = sample.edge_verts[mask]

    assert len(cut_edges) == sample.n_verts - 1, (
        "%s: DUAL_GRAPH cut %d edges; a spanning tree of %d vertices has %d"
        % (name, len(cut_edges), sample.n_verts, sample.n_verts - 1)
    )
    uf = UnionFind(sample.n_verts)
    cycles = 0
    for a, b in cut_edges:
        if uf.find(int(a)) == uf.find(int(b)):
            cycles += 1
        uf.union(int(a), int(b))
    assert cycles == 0, "%s: the cut set contains %d redundant edges" % (name, cycles)
    assert len({uf.find(i) for i in range(sample.n_verts)}) == 1, (
        "%s: the cut set does not span every vertex" % name
    )


def test_dual_graph_is_deterministic():
    """Same mesh, same seed, same cuts -- twice."""
    sample = load_corpus("sphere_smart")
    a = compute_seams(sample, "DUAL_GRAPH", seed=4)
    b = compute_seams(sample, "DUAL_GRAPH", seed=4)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("strategy", SEAM_STRATEGIES)
def test_every_strategy_is_deterministic(strategy):
    """Determinism across the board: a rerun of a seed reproduces a run."""
    sample = load_corpus("hardsurface")
    kwargs = {"angle_deg": 33.0, "curvature_threshold": 0.4, "seed": 9}
    np.testing.assert_array_equal(
        compute_seams(sample, strategy, **kwargs),
        compute_seams(sample, strategy, **kwargs),
    )


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", SEAM_STRATEGIES)
def test_non_manifold_input_does_not_raise(strategy):
    """An edge with three incident faces is legal in Blender, so it is legal here."""
    sample = load_corpus("nonmanifold")
    mask = compute_seams(sample, strategy, angle_deg=30.0, curvature_threshold=0.35)
    assert mask.shape == (sample.n_edges,)
    assert mask.dtype == bool


@pytest.mark.parametrize("strategy", SEAM_STRATEGIES)
def test_pathological_meshes_do_not_raise(strategy):
    """Empty, collapsed, NaN-bearing and random meshes all produce a valid mask."""
    for sample in (
        make_empty(),
        make_all_identical(4),
        make_nan_uvs(),
        make_unit_triangle(),
        make_random_mesh(5, mode="chaotic"),
        make_random_mesh(6, mode="spiky"),
    ):
        mask = compute_seams(sample, strategy, angle_deg=30.0, curvature_threshold=0.35)
        assert mask.dtype == bool
        assert mask.shape == (sample.n_edges,)


@pytest.mark.parametrize("strategy", SEAM_STRATEGIES)
def test_empty_mesh_returns_an_empty_mask(strategy):
    """No edges in, no edges out."""
    assert compute_seams(make_empty(), strategy).shape == (0,)


@pytest.mark.parametrize("strategy", SEAM_STRATEGIES)
def test_hostile_arguments_do_not_raise(strategy):
    """NaN thresholds and absurd angles are clamped, not propagated."""
    sample = load_corpus("hardsurface")
    for angle, threshold in (
        (float("nan"), 0.35),
        (float("inf"), 0.35),
        (-90.0, 0.35),
        (30.0, float("nan")),
        (30.0, -5.0),
        (30.0, 1e9),
    ):
        mask = compute_seams(
            sample, strategy, angle_deg=angle, curvature_threshold=threshold
        )
        assert mask.shape == (sample.n_edges,)


def test_mismatched_existing_mask_is_ignored():
    """An ``existing`` array of the wrong length is ignored, not fatal."""
    sample = load_corpus("hardsurface")
    mask = compute_seams(sample, "EXISTING", existing=np.ones(3, dtype=bool))
    assert mask.shape == (sample.n_edges,)


def test_out_of_range_indices_do_not_raise():
    """A malformed sample degrades rather than throwing an IndexError mid-run."""
    sample = make_flat_grid(4)
    broken = dataclasses.replace(
        sample, edge_verts=sample.edge_verts + 10_000
    )
    for strategy in SEAM_STRATEGIES:
        mask = compute_seams(broken, strategy)
        assert mask.shape == (broken.n_edges,)


def test_seam_mask_never_aliases_the_input():
    """The returned array is owned by the caller and safe to mutate."""
    sample = load_corpus("hardsurface")
    original = sample.edge_seam.copy()
    mask = compute_seams(sample, "EXISTING")
    mask[:] = True
    np.testing.assert_array_equal(sample.edge_seam, original)
