"""Degeneracy must be scale invariant, and must agree with the shared helper.

Every other metric this suite measures is scale free by construction: the same
mesh imported in millimetres and in metres scores identically.  Degeneracy was
the one place that was not, because :func:`evaluate` compared triangle areas
against an *absolute* 1e-12.  That is not a small number for a small model -- a
0.1 mm part, entirely ordinary for a CAD or photogrammetry import, has triangles
of roughly 5e-15, so every one of them was flagged degenerate and a flawless
unwrap scored near zero while still reporting a perfect ``stretch_l2`` of 1.0.

The sweep below spans model scales from 1e+3 down to 1e-11.  A narrower range
does not reproduce the bug: an absolute 1e-12 floor only starts biting on a flat
unit grid somewhere around a model scale of 1e-5, so a sweep that stops at 1e-3
passes against the broken code.
"""

from __future__ import annotations

import numpy as np
import pytest

from unwrap_variations.core.meshdata import UVMeshSample, degenerate_mask

from .conftest import make_flat_grid

#: Model scales the sweep covers.  ``1e-5`` and below are where the old absolute
#: floor misclassified an entirely healthy mesh.
MODEL_SCALES = (1e3, 1e1, 1.0, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-9, 1e-11)

#: UV scales.  UV coordinates can legitimately be tiny on their own account, so
#: the UV-area floor needs the same relative treatment as the 3D one.
UV_SCALES = (1e3, 1.0, 1e-3, 1e-6, 1e-9, 1e-11)


@pytest.mark.parametrize("size", MODEL_SCALES)
def test_isometric_grid_is_never_degenerate_at_any_model_scale(evaluator, size):
    """One mesh, one unwrap, many scales: the reading must not move."""
    sample = make_flat_grid(11, size=size)
    m = evaluator.evaluate(sample, compute_overlap=False)

    assert m.n_tris == 242
    assert m.degenerate_tris == 0, (
        "model scale %g: %d of %d triangles called degenerate. The mesh is a "
        "healthy flat grid; only the scale changed. An absolute area epsilon "
        "misclassifies small models -- the floor must be relative to the mesh's "
        "own mean triangle area." % (size, m.degenerate_tris, m.n_tris)
    )
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-12)
    assert m.stretch_linf == pytest.approx(1.0, abs=1e-12)
    assert m.flipped_tris == 0


@pytest.mark.parametrize("uv_scale", UV_SCALES)
def test_isometric_grid_is_never_degenerate_at_any_uv_scale(evaluator, uv_scale):
    """The UV-area floor is relative to the mean UV area, not to 1e-12."""
    sample = make_flat_grid(11, uv_scale=uv_scale)
    m = evaluator.evaluate(sample, compute_overlap=False)

    assert m.degenerate_tris == 0, (
        "uv scale %g: %d of %d triangles called degenerate on a uniformly "
        "scaled isometric layout" % (uv_scale, m.degenerate_tris, m.n_tris)
    )
    assert m.stretch_l2 == pytest.approx(1.0, abs=1e-12)


def test_scoring_of_a_small_model_matches_the_same_model_scaled_up(evaluator, metrics_mod):
    """The end-to-end consequence: a sub-millimetre part must score the same."""
    big = evaluator.evaluate(make_flat_grid(11, size=1.0), compute_overlap=False)
    tiny = evaluator.evaluate(make_flat_grid(11, size=1e-9), compute_overlap=False)
    weights = metrics_mod.ScoreWeights()
    assert metrics_mod.score(tiny, weights) == pytest.approx(
        metrics_mod.score(big, weights), rel=1e-9
    ), "a 1e-9-scale copy of the same mesh scored differently from the original"


@pytest.mark.parametrize("size", MODEL_SCALES)
def test_a_genuine_collapse_is_still_caught_at_every_scale(evaluator, size):
    """A relative floor must not become a floor of zero."""
    sample = make_flat_grid(8, size=size)
    verts = sample.verts.copy()
    # Weld two corners of triangle 0 onto the third: a real, deliberate collapse.
    verts[sample.tri_verts[0, 1]] = verts[sample.tri_verts[0, 0]]
    verts[sample.tri_verts[0, 2]] = verts[sample.tri_verts[0, 0]]
    collapsed = UVMeshSample(
        verts=verts,
        tri_verts=sample.tri_verts,
        tri_loops=sample.tri_loops,
        uvs=sample.uvs,
        edge_verts=sample.edge_verts,
        edge_seam=sample.edge_seam,
        loop_verts=sample.loop_verts,
    )
    m = evaluator.evaluate(collapsed, compute_overlap=False)
    assert m.degenerate_tris > 0, (
        "model scale %g: a collapsed triangle went unreported, so the relative "
        "floor has degenerated into no floor at all" % size
    )
    assert m.degenerate_tris == int(degenerate_mask(collapsed).sum())


def test_a_fully_collapsed_mesh_is_entirely_degenerate(evaluator):
    """With no usable scale to work from, everything is degenerate, not nothing."""
    sample = make_flat_grid(5, size=0.0)
    m = evaluator.evaluate(sample, compute_overlap=False)
    assert m.degenerate_tris == m.n_tris
    assert m.n_tris > 0


@pytest.mark.parametrize("size", (1e3, 1.0, 1e-6, 1e-9))
@pytest.mark.parametrize("uv_scale", (1e2, 1.0, 1e-8))
def test_evaluator_and_degenerate_mask_agree(evaluator, size, uv_scale):
    """The two implementations of one rule must not drift apart.

    ``evaluate`` deliberately does *not* call
    :func:`~unwrap_variations.core.meshdata.degenerate_mask`: the helper works
    from the ``(T, 3, 3)`` and ``(T, 3, 2)`` corner blocks and costs 137 ms at
    522k triangles against 194 ms for the whole evaluation, which would put it
    30% over its stated budget.  It reimplements the identical rule on the
    ``(T,)`` fundamental-form scalars instead, so this test is what keeps the
    two honest.
    """
    sample = make_flat_grid(9, size=size, uv_scale=uv_scale)
    m = evaluator.evaluate(sample, compute_overlap=False)
    assert m.degenerate_tris == int(degenerate_mask(sample).sum())


def test_evaluator_and_degenerate_mask_agree_on_random_meshes(evaluator):
    """The same agreement, over 20 decades of scale and mixed-in collapses."""
    rng = np.random.default_rng(20260807)
    for i in range(250):
        n_tris = int(rng.integers(1, 30))
        n_verts = max(3, n_tris)
        scale = 10.0 ** float(rng.integers(-12, 8))
        verts = rng.normal(0.0, 1.0, (n_verts, 3)) * scale
        tri_verts = rng.integers(0, n_verts, (n_tris, 3)).astype(np.int32)
        tri_loops = np.arange(n_tris * 3, dtype=np.int32).reshape(-1, 3)
        uvs = rng.uniform(0.0, 1.0, (n_tris * 3, 2)) * 10.0 ** float(rng.integers(-12, 8))
        if i % 3 == 0:
            k = int(rng.integers(0, n_tris))
            uvs[tri_loops[k, 1]] = uvs[tri_loops[k, 0]]
            uvs[tri_loops[k, 2]] = uvs[tri_loops[k, 0]]
        loop_verts = np.zeros(n_tris * 3, dtype=np.int32)
        loop_verts[tri_loops.reshape(-1)] = tri_verts.reshape(-1)
        sample = UVMeshSample(
            verts=verts,
            tri_verts=tri_verts,
            tri_loops=tri_loops,
            uvs=uvs,
            edge_verts=np.zeros((0, 2), np.int32),
            edge_seam=np.zeros(0, dtype=bool),
            loop_verts=loop_verts,
        )
        got = evaluator.evaluate(sample, compute_overlap=False).degenerate_tris
        want = int(degenerate_mask(sample).sum())
        assert got == want, (
            "random mesh %d at scale %g: evaluate() says %d degenerate, "
            "degenerate_mask() says %d" % (i, scale, got, want)
        )
