"""Tests for ``unwrap_variations.core.meshdata`` -- the bpy-free interchange.

:class:`UVMeshSample` is the only structure that crosses from Blender into the
pure core, so everything downstream inherits its correctness.  The geometric
helpers here (``tri_areas_3d``, ``tri_areas_uv``, ``degenerate_mask``) are the
primitives the evaluator builds on, and they are checked against the naive
oracle rather than against themselves.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests import reference as ref
from tests.conftest import (
    CORPUS_NAMES,
    load_corpus,
    make_all_identical,
    make_cube,
    make_empty,
    make_flat_grid,
    make_nan_uvs,
    make_unit_triangle,
    make_zero_area_mix,
)
from unwrap_variations.core.meshdata import (
    UVMeshSample,
    degenerate_mask,
    make_sample,
    tri_areas_3d,
    tri_areas_uv,
)

# ---------------------------------------------------------------------------
# Shape and size accounting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_corpus_samples_are_internally_consistent(name):
    """Every committed corpus mesh validates clean.

    If the corpus itself were malformed, every downstream comparison would be
    measuring the wrong thing.
    """
    sample = load_corpus(name)
    assert sample.validate() == [], "%s: %s" % (name, sample.validate())
    assert sample.n_tris > 0
    assert sample.n_loops > 0
    assert sample.tri_verts.shape == sample.tri_loops.shape
    assert sample.uvs.shape == (sample.n_loops, 2)
    assert sample.loop_verts.shape == (sample.n_loops,)
    assert sample.edge_seam.shape == (sample.n_edges,)


def test_sizes_match_the_arrays():
    """The size properties simply report array shapes."""
    sample = make_flat_grid(4)
    assert sample.n_verts == sample.verts.shape[0]
    assert sample.n_tris == sample.tri_verts.shape[0]
    assert sample.n_loops == sample.uvs.shape[0]
    assert sample.n_edges == sample.edge_verts.shape[0]
    assert sample.n_tris == 32


def test_tri_positions_and_tri_uvs_shapes():
    """The derived views are ``(T, 3, 3)`` and ``(T, 3, 2)``."""
    sample = make_cube()
    assert sample.tri_positions().shape == (sample.n_tris, 3, 3)
    assert sample.tri_uvs().shape == (sample.n_tris, 3, 2)
    np.testing.assert_allclose(
        sample.tri_positions()[0, 0], sample.verts[sample.tri_verts[0, 0]]
    )
    np.testing.assert_allclose(sample.tri_uvs()[0, 0], sample.uvs[sample.tri_loops[0, 0]])


def test_empty_mesh_views_are_empty_not_broken():
    """An empty mesh yields correctly-shaped empty arrays, not an exception."""
    sample = make_empty()
    assert sample.n_tris == 0
    assert sample.tri_positions().shape == (0, 3, 3)
    assert sample.tri_uvs().shape == (0, 3, 2)
    assert tri_areas_3d(sample).shape == (0,)
    assert tri_areas_uv(sample).shape == (0,)
    assert degenerate_mask(sample).shape == (0,)


# ---------------------------------------------------------------------------
# Immutability and copy-with helpers.
# ---------------------------------------------------------------------------


def test_sample_is_frozen():
    """The sample is immutable, so nothing can mutate a shared mesh mid-run."""
    sample = make_flat_grid(2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.name = "nope"  # type: ignore[misc]


def test_with_uvs_returns_a_new_sample_sharing_topology():
    """``with_uvs`` swaps the UV layer and leaves every other array alone."""
    base = make_flat_grid(3)
    swapped = base.with_uvs(base.uvs * 2.0)
    assert swapped is not base
    np.testing.assert_allclose(swapped.uvs, base.uvs * 2.0)
    assert swapped.tri_verts is base.tri_verts
    assert swapped.tri_loops is base.tri_loops
    # The original is untouched.
    np.testing.assert_allclose(base.uvs, make_flat_grid(3).uvs)


def test_with_uvs_rejects_a_wrong_length():
    """A UV array of the wrong length is a bug at the call site, and raises.

    This is the one place a raise is right: silently accepting a mismatched
    array would blit the wrong UVs onto the mesh, which is far worse than an
    exception in the caller.
    """
    base = make_flat_grid(2)
    with pytest.raises(ValueError):
        base.with_uvs(np.zeros((base.n_loops + 3, 2)))


def test_with_uvs_accepts_flat_and_float32():
    """A flat ``(2L,)`` float32 buffer -- what ``foreach_get`` gives -- is accepted."""
    base = make_flat_grid(2)
    flat = np.asarray(base.uvs, dtype=np.float32).reshape(-1)
    swapped = base.with_uvs(flat)
    assert swapped.uvs.shape == (base.n_loops, 2)
    np.testing.assert_allclose(swapped.uvs, base.uvs, rtol=1e-6)


def test_with_seams_swaps_the_mask():
    """``with_seams`` replaces the seam mask and validates its length."""
    base = make_flat_grid(2)
    mask = np.zeros(base.n_edges, dtype=bool)
    mask[::3] = True
    seamed = base.with_seams(mask)
    assert int(seamed.edge_seam.sum()) == int(mask.sum())
    with pytest.raises(ValueError):
        base.with_seams(np.zeros(base.n_edges + 1, dtype=bool))


# ---------------------------------------------------------------------------
# validate() -- reports, never raises.
# ---------------------------------------------------------------------------


def test_validate_is_clean_for_a_good_mesh():
    """A well-formed mesh reports no problems."""
    assert make_cube().validate() == []


def test_validate_reports_an_empty_mesh():
    """A mesh with no triangles is reported, not raised on."""
    problems = make_empty().validate()
    assert any("no triangles" in p for p in problems)


def test_validate_reports_out_of_range_indices():
    """A triangle referencing a nonexistent vertex is caught."""
    sample = UVMeshSample(
        verts=np.zeros((3, 3)),
        tri_verts=np.array([[0, 1, 99]], dtype=np.int32),
        tri_loops=np.array([[0, 1, 2]], dtype=np.int32),
        uvs=np.zeros((3, 2)),
        edge_verts=np.zeros((0, 2), dtype=np.int32),
        edge_seam=np.zeros(0, dtype=bool),
        loop_verts=np.zeros(3, dtype=np.int32),
    )
    problems = sample.validate()
    assert any("vertex index out of range" in p for p in problems)


def test_validate_reports_non_finite_coordinates():
    """NaN in either the positions or the UVs is reported."""
    problems = make_nan_uvs().validate()
    assert any("NaN or Inf" in p for p in problems)


def test_validate_never_raises_on_anything():
    """Validation is diagnostic: it must survive input it cannot understand."""
    broken = UVMeshSample(
        verts=np.zeros((2, 3)),
        tri_verts=np.zeros((1, 3), dtype=np.int32),
        tri_loops=np.zeros((2, 3), dtype=np.int32),  # deliberately mismatched
        uvs=np.zeros((1, 2)),
        edge_verts=np.zeros((3, 2), dtype=np.int32),
        edge_seam=np.zeros(1, dtype=bool),  # deliberately mismatched
        loop_verts=np.zeros(1, dtype=np.int32),
    )
    problems = broken.validate()
    assert isinstance(problems, list)
    assert problems


def test_summary_is_ascii_and_informative():
    """``summary`` is for logs and a cp1252 console, so it must be pure ASCII."""
    text = load_corpus("hardsurface").summary()
    text.encode("ascii")
    assert "T=508" in text


# ---------------------------------------------------------------------------
# make_sample -- the loose constructor.
# ---------------------------------------------------------------------------


def test_make_sample_derives_edges_loops_and_uvs():
    """Given only vertices and triangles, everything else is derived sanely."""
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    sample = make_sample(verts, [[0, 1, 2], [1, 3, 2]])
    assert sample.n_tris == 2
    assert sample.n_loops == 6
    np.testing.assert_allclose(sample.uvs, 0.0)
    # 5 unique undirected edges: 3 per triangle minus the shared one.
    assert sample.n_edges == 5
    assert sample.edge_seam.shape == (5,)
    assert not sample.edge_seam.any()
    # Derived edges are sorted pairs and unique.
    pairs = {tuple(e) for e in sample.edge_verts}
    assert len(pairs) == 5
    assert all(a < b for a, b in pairs)


def test_make_sample_maps_loops_to_vertices():
    """The derived ``loop_verts`` agrees with ``tri_verts`` corner for corner."""
    sample = make_sample([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    for tri in range(sample.n_tris):
        for corner in range(3):
            loop = sample.tri_loops[tri, corner]
            assert sample.loop_verts[loop] == sample.tri_verts[tri, corner]


def test_make_sample_with_no_triangles():
    """An empty triangle list yields an empty, valid sample."""
    sample = make_sample([[0, 0, 0]], np.zeros((0, 3), dtype=np.int32))
    assert sample.n_tris == 0
    assert sample.n_edges == 0
    assert sample.n_loops == 0


# ---------------------------------------------------------------------------
# Geometric helpers, checked against the oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_tri_areas_3d_match_the_oracle(name):
    """Vectorised 3D areas match the naive cross-product formula exactly."""
    sample = load_corpus(name)
    fast = tri_areas_3d(sample)
    naive = np.array(
        [
            ref.tri_area_3d(*[tuple(sample.verts[v]) for v in sample.tri_verts[i]])
            for i in range(sample.n_tris)
        ]
    )
    np.testing.assert_allclose(fast, naive, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_tri_areas_uv_are_signed_and_match_the_oracle(name):
    """UV areas are signed -- the sign is how flipped triangles are found."""
    sample = load_corpus(name)
    fast = tri_areas_uv(sample)
    naive = np.array(
        [
            ref.tri_area_uv_signed(*[tuple(sample.uvs[loop]) for loop in sample.tri_loops[i]])
            for i in range(sample.n_tris)
        ]
    )
    np.testing.assert_allclose(fast, naive, rtol=1e-12, atol=1e-18)


def test_tri_areas_uv_sign_flips_with_winding():
    """Reversing the winding negates the area and nothing else."""
    forward = make_unit_triangle(uv=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    mirrored = make_unit_triangle(uv=((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    assert tri_areas_uv(forward)[0] == pytest.approx(0.5)
    assert tri_areas_uv(mirrored)[0] == pytest.approx(-0.5)
    assert tri_areas_3d(forward)[0] == pytest.approx(0.5)


def test_sphere_planar_has_exactly_half_its_triangles_flipped():
    """The bad-unwrap control flips 1104 of its 2208 triangles, by construction.

    A planar projection of a sphere maps the far hemisphere onto the near one
    with reversed winding.  Anything other than exactly half means the sign
    convention has drifted.
    """
    sample = load_corpus("sphere_planar")
    flipped = int((tri_areas_uv(sample) < 0).sum())
    assert flipped == 1104
    assert flipped * 2 == sample.n_tris


# ---------------------------------------------------------------------------
# degenerate_mask.
# ---------------------------------------------------------------------------


def test_degenerate_mask_flags_zero_area_and_nan():
    """Zero 3D area, zero UV area and non-finite coordinates are all degenerate."""
    mask = degenerate_mask(make_zero_area_mix())
    assert mask.tolist() == [False, True, True]


def test_degenerate_mask_flags_nan_uvs():
    """Triangles touching a NaN UV are excluded from the distortion means."""
    sample = make_nan_uvs()
    mask = degenerate_mask(sample)
    assert mask.any()
    # Only the triangles that actually touch a poisoned loop.
    bad_loops = np.nonzero(~np.isfinite(sample.uvs).all(axis=1))[0]
    for tri in np.nonzero(mask)[0]:
        assert set(sample.tri_loops[tri]) & set(bad_loops.tolist())


def test_degenerate_mask_flags_everything_on_a_collapsed_mesh():
    """A mesh of coincident points is entirely degenerate."""
    sample = make_all_identical(5)
    assert degenerate_mask(sample).all()


def test_degenerate_mask_is_clean_on_a_good_mesh():
    """A healthy unwrap has no degenerate triangles at all."""
    assert not degenerate_mask(load_corpus("sphere_smart")).any()
    assert not degenerate_mask(make_flat_grid(6)).any()


def test_degenerate_mask_honours_epsilon():
    """A larger epsilon marks more triangles; the threshold is on *area*."""
    sample = make_flat_grid(4)  # cell area 1/16, triangle area 1/32
    assert not degenerate_mask(sample, eps=1e-12).any()
    assert degenerate_mask(sample, eps=1.0).all()


def test_degenerate_mask_matches_the_oracle_count(corpus_sample):
    """The mask agrees with the oracle's own degenerate tally."""
    want = ref.reference_metrics(corpus_sample, do_overlap=False)["degenerate_tris"]
    assert int(degenerate_mask(corpus_sample).sum()) == want
