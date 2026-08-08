"""The organic corpus and its oracles -- docs/ORGANIC.md section 8.

Three separate claims are checked here, and they fail for different reasons:

1. **The corpus is what it says it is.**  Every mesh matches its pinned
   invariants, and the two independent topology implementations agree shell for
   shell.  A drift in numpy, in the builders or in the polygoniser shows up
   here rather than as a mysteriously moved benchmark cell three scripts later.
2. **The oracles catch the deliberate defects.**  ``folded_ribbon`` ships a UV
   layout that is locally injective everywhere and globally non-injective -- the
   exact defect class ORGANIC.md section 0.3 found on Suzanne, where
   ``flipped_tris`` is 0 and the overlap is real.  A test suite that cannot
   distinguish those two failures cannot certify a repair for either.
3. **Nothing crashes on ``pinched``.**  Zero-area faces, near-degenerate
   slivers and coincident vertices go through every metric path without an
   exception and without a NaN.

The two Suzannes need Blender and are skipped here; they are covered by
``tools/diag_organic.py``, which runs headless against both supported builds.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tests.reference import (
    reference_euler_characteristic,
    reference_layout_metrics,
    reference_metrics,
    reference_shell_topology,
    reference_union_rect_area,
)
from unwrap_variations.core.evaluator import evaluate
from unwrap_variations.core.meshdata import (
    degenerate_mask,
    make_sample,
    tri_areas_3d,
    tri_areas_uv,
)

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_corpus():
    """Import ``tools/organic_corpus.py`` by path.

    ``tools/`` is deliberately not a package -- its scripts are run by Blender
    with a bare path -- so the corpus is loaded through the import machinery
    rather than by adding another entry to ``sys.path``.
    """
    name = "unwrap_variations_organic_corpus"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / "organic_corpus.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


corpus = _load_corpus()


def _load_render():
    """Import ``tools/render_uv_report.py`` behind a bare ``bpy`` stub.

    The report needs Blender to *render*, but the parts that decide what the
    picture means -- island labelling, the fill palette, the caption -- are
    plain numpy and are exactly the parts that can silently lie.  A stub module
    is enough to import them, and testing them here rather than only inside
    Blender is what makes a regression on them cheap to catch.
    """
    name = "unwrap_variations_render_uv_report"
    if name in sys.modules:
        return sys.modules[name]
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location(name, _TOOLS / "render_uv_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = _load_render()


@pytest.fixture(scope="module")
def meshes():
    """Build every bpy-free corpus mesh once for the whole module."""
    return {m.name: m for m in corpus.build_all(skip_blender=True)}


# ---------------------------------------------------------------------------
# 1. The corpus is what it says it is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", corpus.NUMPY_NAMES)
def test_invariants_match_the_pinned_table(meshes, name):
    """Each mesh reproduces its pinned vertex/triangle/shell/chi fingerprint."""
    assert meshes[name].invariants == corpus.EXPECTED[name]


@pytest.mark.parametrize("name", corpus.NUMPY_NAMES)
def test_vectorised_topology_agrees_with_the_naive_oracle(meshes, name):
    """The fast shell/chi computation matches the loop-based reference exactly.

    This is the check that makes every ``chi`` in ORGANIC.md section 0 worth
    quoting: two implementations that share no code produce the same answer.
    """
    mesh = meshes[name]
    fast = corpus.mesh_topology(mesh.tris, mesh.n_verts)
    slow = reference_shell_topology(mesh.tris, mesh.n_verts)
    assert len(fast) == len(slow)
    for got, want in zip(fast, slow, strict=True):
        assert got.faces == want["faces"]
        assert got.edges == want["edges"]
        assert got.verts == want["verts"]
        assert got.chi == want["chi"]
        assert got.boundary_edges == want["boundary_edges"]
        assert got.nonmanifold_edges == want["nonmanifold_edges"]


def test_closed_meshes_are_watertight_and_manifold(meshes):
    """The polygoniser and the sweeps produce no boundary and no T-junction.

    A marching polygoniser that cracks produces boundary edges; a
    non-conforming one produces edges with three incident faces.  Both would
    silently invalidate every unwrap measured on the mesh, so both are checked
    rather than assumed.
    """
    for name in ("blob_limbs", "hand", "displaced_sphere", "torus_knot", "pinched"):
        inv = meshes[name].invariants
        assert inv.shells == 1, name
        assert inv.boundary_edges == (0,), name
        assert inv.nonmanifold_edges == 0, name


def test_the_corpus_spans_the_topologies_it_claims_to(meshes):
    """Genus 0 closed, genus 1 closed and a disk with boundary are all present."""
    chis = {name: meshes[name].invariants.chi[0] for name in corpus.NUMPY_NAMES}
    assert chis["displaced_sphere"] == 2, "closed genus 0"
    assert chis["torus_knot"] == 0, "closed genus 1"
    assert chis["folded_ribbon"] == 1, "disk with a boundary"
    assert meshes["folded_ribbon"].invariants.boundary_edges[0] > 0


def test_euler_characteristic_oracle_sums_over_shells(meshes):
    """The whole-soup Euler characteristic is the sum of the per-shell ones."""
    mesh = meshes["displaced_sphere"]
    assert reference_euler_characteristic(mesh.tris) == 2


def test_the_builders_are_deterministic(meshes):
    """Building a mesh twice gives bit-identical arrays.

    Determinism is not merely a nice property here.  ``docs/BENCHMARKS.md``
    section 3 makes medians-over-repeats mandatory *because* Blender's
    operators are non-deterministic; if the corpus were non-deterministic too
    there would be no fixed point left to measure anything against.
    """
    for name in corpus.NUMPY_NAMES:
        again = corpus.build(name)
        assert np.array_equal(again.verts, meshes[name].verts), name
        assert np.array_equal(again.tris, meshes[name].tris), name
        assert again.digest() == meshes[name].digest(), name


def test_an_altered_mesh_is_rejected():
    """A corpus regression raises at build time rather than propagating."""
    saved = corpus.EXPECTED["pinched"]
    corpus.EXPECTED["pinched"] = corpus.Invariants(1, 1, 1, (1,), (0,), 0)
    try:
        with pytest.raises(RuntimeError, match="pinched"):
            corpus.build("pinched")
    finally:
        corpus.EXPECTED["pinched"] = saved


# ---------------------------------------------------------------------------
# 2. The oracles catch the deliberate defects
# ---------------------------------------------------------------------------


def test_folded_ribbon_is_locally_injective(meshes):
    """Every signed UV area is positive: no triangle is flipped.

    Half the point of this entry.  ORGANIC.md section 0.3 records that on
    Suzanne ``flipped_tris`` is 0 while the overlap is real, so a defect that
    announced itself as a flip would not be reproducing the bug at all.
    """
    sample = corpus.to_sample(meshes["folded_ribbon"])
    signed = tri_areas_uv(sample)
    assert (signed > 0.0).all()
    assert int(evaluate(sample, compute_overlap=False).flipped_tris) == 0


def test_folded_ribbon_is_globally_non_injective(meshes):
    """The independent all-pairs oracle finds real overlap area anyway.

    Local injectivity plus global non-injectivity is precisely the combination
    a flip-only check cannot see, and it is the combination the repair chain has
    to be certified against.
    """
    sample = corpus.to_sample(meshes["folded_ribbon"])
    ref = reference_metrics(sample, do_overlap=True, overlap_limit=4000)
    assert ref["flipped_tris"] == 0
    assert ref["overlap_pairs"] > 0
    assert ref["overlap_area"] > 1e-6


def test_evaluator_agrees_with_the_oracle_on_the_ribbon_overlap(meshes):
    """The shipped evaluator and the naive oracle report the same overlap area.

    ORGANIC.md section 0.3 established this agreement on Suzanne to ten
    significant digits and concluded the evaluator must not be "fixed".  This
    is the same check on a mesh whose defect is known by construction rather
    than by measurement.
    """
    sample = corpus.to_sample(meshes["folded_ribbon"])
    ref = reference_metrics(sample, do_overlap=True, overlap_limit=4000)
    got = evaluate(sample, compute_overlap=True)
    assert got.overlap_area == pytest.approx(ref["overlap_area"], rel=1e-6)


def test_a_clean_layout_is_not_flagged(meshes):
    """The oracle does not cry wolf: the same mesh unrolled flat reports zero.

    This is the direction that actually bites in production.  A certifier that
    answered "broken" unconditionally would pass every defect test above and
    then reject every good layout a user has, so the corpus ships the control
    alongside the defect and both are checked.
    """
    sample = corpus.to_sample(meshes["folded_ribbon"])
    clean = sample.with_uvs(corpus.folded_ribbon_flat_uvs().reshape(-1, 2))
    ref = reference_metrics(clean, do_overlap=True, overlap_limit=4000)
    assert ref["flipped_tris"] == 0
    assert ref["overlap_pairs"] == 0
    assert ref["overlap_area"] == 0.0
    assert evaluate(clean, compute_overlap=True).overlap_area == 0.0


def _certifier():
    """Return ``core.injectivity.certify`` if it has landed, else ``None``.

    Feature-detected rather than imported at module scope on purpose.  The
    corpus and its oracles are a deliverable in their own right (ORGANIC.md
    section 8) and must keep working while the modules in sections 4 and 5 are
    still being written in parallel.
    """
    try:
        from unwrap_variations.core.injectivity import certify  # noqa: PLC0415
    except ImportError:
        return None
    return certify


def _naive_certifier():
    """Return ``tests.reference.naive_certify`` if it has landed, else ``None``."""
    from tests import reference  # noqa: PLC0415

    return getattr(reference, "naive_certify", None)


def test_runtime_certifier_flags_the_ribbon(meshes):
    """``core.injectivity.certify`` must call the ribbon globally non-injective.

    Skipped until that module lands.  The corpus does not depend on it; it is
    the other way round.
    """
    certify = _certifier()
    if certify is None:
        pytest.skip("core.injectivity has not landed yet")
    sample = corpus.to_sample(meshes["folded_ribbon"])
    broken = certify(sample)
    assert broken.flipped == 0, "the defect must not be a flip"
    assert broken.locally_injective is True
    assert broken.globally_injective is False
    assert broken.boundary_crossings > 0

    clean = certify(sample.with_uvs(corpus.folded_ribbon_flat_uvs().reshape(-1, 2)))
    assert clean.globally_injective is True
    assert clean.boundary_crossings == 0


@pytest.mark.parametrize("name", ["folded_ribbon", "pinched", "displaced_sphere"])
def test_runtime_certifier_agrees_with_its_oracle(meshes, name):
    """The fast certifier and the naive oracle agree on every corpus mesh.

    ORGANIC.md section 4.1: "Last time an independent reference caught more
    than all the self-testing combined; this is not optional."
    """
    certify = _certifier()
    naive = _naive_certifier()
    if certify is None or naive is None:
        pytest.skip("injectivity module or its oracle has not landed yet")
    mesh = meshes[name]
    sample = corpus.to_sample(mesh)
    if mesh.uvs is None:
        flat = sample.verts[sample.tri_verts][:, :, :2].reshape(-1, 2)
        lo, hi = flat.min(axis=0), flat.max(axis=0)
        sample = sample.with_uvs(0.02 + 0.96 * (flat - lo) / np.maximum(hi - lo, 1e-12))
    got = certify(sample)
    want = naive(sample)
    assert got.flipped == want["flipped"]
    assert got.degenerate == want["degenerate"]
    assert got.boundary_crossings == want["boundary_crossings"]
    assert got.locally_injective == want["locally_injective"]
    assert got.globally_injective == want["globally_injective"]


# ---------------------------------------------------------------------------
# 3. pinched must not crash anything
# ---------------------------------------------------------------------------


def test_pinched_really_is_degenerate(meshes):
    """The damage is present: exact zero-area faces and coincident vertices."""
    mesh = meshes["pinched"]
    sample = corpus.to_sample(mesh)
    areas = tri_areas_3d(sample)
    assert (areas == 0.0).any(), "expected at least one exactly zero-area triangle"
    positions = {tuple(v) for v in mesh.verts.tolist()}
    assert len(positions) < mesh.n_verts, "expected coincident vertices"


def test_pinched_survives_every_metric_path(meshes):
    """Evaluation of the damaged mesh returns finite numbers, and does not raise."""
    sample = corpus.to_sample(meshes["pinched"])
    for overlap in (False, True):
        metrics = evaluate(sample, compute_overlap=overlap)
        for field, value in metrics.to_dict().items():
            if isinstance(value, (int, float)):
                assert math.isfinite(float(value)), field
    assert int(degenerate_mask(sample).sum()) > 0


def test_pinched_survives_the_oracles(meshes):
    """Both oracles handle the damaged mesh without raising or returning NaN."""
    sample = corpus.to_sample(meshes["pinched"])
    ref = reference_metrics(sample, do_overlap=True, overlap_limit=4000)
    layout = reference_layout_metrics(sample)
    for source in (ref, layout):
        for key, value in source.items():
            if isinstance(value, float):
                assert math.isfinite(value), key


# ---------------------------------------------------------------------------
# The layout-metric oracle itself
# ---------------------------------------------------------------------------


def test_exact_clipper_rejects_a_collapsed_triangle():
    """A zero-area triangle overlaps nothing, however the clipper is written.

    This is a regression, not a hypothetical.  Sutherland-Hodgman clips against
    each edge of the clip polygon by a half-plane test, and a zero-length edge's
    half-plane test is vacuously true -- so a collapsed clip triangle clips
    *nothing* and the routine hands back the subject triangle's entire area for
    a pair that does not overlap at all.  ``tests/reference.py`` carried exactly
    that bug.  It matters most on ``pinched``, whose whole purpose is to carry
    exactly zero-area faces, and where it would paint most of the mesh as a
    defect in the visual report.
    """
    flat = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    collapsed = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    degenerate_edge = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert corpus.tri_clip_area_uv(flat, collapsed) == 0.0
    assert corpus.tri_clip_area_uv(collapsed, flat) == 0.0
    assert corpus.tri_clip_area_uv(flat, degenerate_edge) == 0.0
    assert corpus.tri_clip_area_uv(flat, flat) == pytest.approx(0.5)
    # Sharing an edge is touching, not overlapping: measure zero.
    neighbour = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert corpus.tri_clip_area_uv(flat, neighbour) == pytest.approx(0.0, abs=1e-15)


def test_overlap_mask_flags_the_ribbon_and_clears_the_control(meshes):
    """The grid-accelerated mask agrees with the layouts it is given.

    The defective ribbon has overlapping triangles; the same ribbon unrolled
    flat has none. The mask is what paints red in the visual report, so a false
    positive here becomes a picture that cries wolf.
    """
    mesh = meshes["folded_ribbon"]
    broken = corpus.overlapping_triangle_mask(mesh.uvs)
    clean = corpus.overlapping_triangle_mask(corpus.folded_ribbon_flat_uvs())
    assert int(broken.sum()) > 0
    assert int(clean.sum()) == 0


def test_overlap_mask_is_silent_on_a_clean_projection(meshes):
    """A non-overlapping projection of a corpus mesh flags nothing.

    ``pinched`` is used because its zero-area faces are what the collapsed-clip
    bug turns into false positives.
    """
    sample = corpus.to_sample(meshes["pinched"])
    flat = sample.verts[sample.tri_verts][:, :, :2]
    lo = flat.reshape(-1, 2).min(axis=0)
    hi = flat.reshape(-1, 2).max(axis=0)
    # Spread the triangles onto a grid so none of them can genuinely overlap.
    n = flat.shape[0]
    side = int(math.ceil(math.sqrt(n)))
    cell = 1.0 / side
    normalised = (flat - lo) / np.maximum(hi - lo, 1e-12) * (cell * 0.8)
    idx = np.arange(n)
    offset = np.stack([(idx % side) * cell, (idx // side) * cell], axis=1)
    spread = normalised + offset[:, None, :]
    assert int(corpus.overlapping_triangle_mask(spread).sum()) == 0


def test_union_rect_area_on_known_cases():
    """The rectangle-union sweep gets the textbook answers right."""
    assert reference_union_rect_area([]) == 0.0
    assert reference_union_rect_area([(0.0, 0.0, 2.0, 3.0)]) == pytest.approx(6.0)
    # Two unit squares overlapping in a 0.5 x 0.5 corner.
    got = reference_union_rect_area([(0.0, 0.0, 1.0, 1.0), (0.5, 0.5, 1.5, 1.5)])
    assert got == pytest.approx(2.0 - 0.25)
    # Fully nested.
    nested = reference_union_rect_area([(0.0, 0.0, 4.0, 4.0), (1.0, 1.0, 2.0, 2.0)])
    assert nested == pytest.approx(16.0)
    # Degenerate rectangles contribute nothing.
    assert reference_union_rect_area([(0.0, 0.0, 0.0, 5.0)]) == 0.0


# ---------------------------------------------------------------------------
# The visual report has to tell the truth
# ---------------------------------------------------------------------------


def _laid_out(mesh):
    """Return a sample carrying a layout: the mesh's own, or a flat projection."""
    sample = corpus.to_sample(mesh)
    if mesh.uvs is not None:
        return sample
    flat = sample.verts[sample.tri_verts][:, :, :2].reshape(-1, 2)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    return sample.with_uvs(0.02 + 0.96 * (flat - lo) / np.maximum(hi - lo, 1e-12))


def _scattered_islands(count):
    """Return a sample of ``count`` triangles laid out with no two touching."""
    tri = np.array([[0.0, 0.0], [0.16, 0.0], [0.0, 0.16]])
    tri_uv = np.asarray(
        [tri + np.array([0.02 + 0.2 * (i % 5), 0.02 + 0.2 * (i // 5)]) for i in range(count)]
    )
    verts = np.zeros((count * 3, 3))
    verts[:, :2] = tri_uv.reshape(-1, 2)
    return make_sample(
        verts, np.arange(count * 3).reshape(count, 3), uvs=tri_uv.reshape(-1, 2), name="grid"
    )


def test_every_island_gets_its_own_colour():
    """Distinct islands get distinct fills, however many there are.

    Regression.  The palette was a 12-entry cycle indexed by ``label % 12``,
    while the corpus routinely produces layouts of 90 to 330 islands
    (``blob_limbs`` measures 328 under ``smart_project``, ``hand`` 97 after
    merging).  At 30 islands, 18 of them shared a fill with another one, so two
    adjacent charts painted the same colour read as a single chart -- in the one
    picture whose entire purpose, per ``docs/BENCHMARKS.md`` section 4.1, is to
    show that a good-scoring layout is really confetti.
    """
    for count in (12, 30, 400):
        colours = [render.island_colour(i) for i in range(count)]
        assert len(set(colours)) == count, "%d islands share a colour at n=%d" % (
            count - len(set(colours)), count
        )
    # The first twelve are unchanged, so older renders stay comparable.
    assert [render.island_colour(i) for i in range(12)] == list(render.PALETTE)
    # Deterministic, and independent of how many are asked for.
    assert render.island_colour(97) == render.island_colour(97)
    for r, g, b in (render.island_colour(i) for i in range(200)):
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0


def test_island_labels_weld_on_the_evaluators_tolerance(meshes):
    """The picture's island count matches the number printed beside it.

    Regression.  ``island_labels`` welded on ``round(uv, 7)``, an *absolute*
    lattice, while ``core/evaluator.py`` welds on ``1e-6`` of the layout's own
    span.  Measured inside Blender on a ``smart_project`` layout of
    ``blob_limbs``: the evaluator reported **328** islands and the picture's
    caption said **329**, on the same page, for the same layout.
    """
    for name in ("folded_ribbon", "pinched", "displaced_sphere"):
        sample = _laid_out(meshes[name])
        drawn = int(render.island_labels(sample).max()) + 1
        assert drawn == evaluate(sample, compute_overlap=False).island_count, name

    # The tolerance is relative, so a layout and the same layout at a millionth
    # of the size have the same islands.  An absolute round-to-7 lattice fuses
    # the small one into a handful of cells and reports far fewer.
    scattered = _scattered_islands(24)
    full = int(render.island_labels(scattered).max()) + 1
    tiny = scattered.with_uvs(scattered.uvs * 1e-6)
    assert full == 24
    assert int(render.island_labels(tiny).max()) + 1 == full
    assert evaluate(tiny, compute_overlap=False).island_count == full


def test_island_labels_survive_a_broken_layout(meshes):
    """A NaN UV must not crash the report that exists to show broken layouts."""
    sample = corpus.to_sample(meshes["folded_ribbon"])
    uvs = sample.uvs.copy()
    uvs[0] = np.nan
    uvs[1] = np.inf
    labels = render.island_labels(sample.with_uvs(uvs))
    assert labels.shape[0] == sample.n_tris
    assert int(labels.min()) >= 0


def test_the_caption_names_a_disagreement_instead_of_hiding_it():
    """When the two island counts differ the caption says so."""
    agree = render.uv_caption({"islands": 14, "flipped": 0, "overlapping": 3}, 14)
    assert agree == "UV layout -- 14 islands, 3 red"
    differ = render.uv_caption({"islands": 329, "flipped": 0, "overlapping": 0}, 328)
    assert "MISMATCH" in differ
    assert "328" in differ and "329" in differ


@pytest.mark.parametrize("name", ["folded_ribbon", "displaced_sphere", "pinched"])
def test_layout_oracle_agrees_with_the_evaluator(meshes, name):
    """Layout metrics from the oracle match the shipped evaluator.

    ``packing_efficiency`` is compared loosely on purpose: the evaluator
    rasterises the union of island bounding boxes onto a bounded lattice while
    the oracle sweeps it exactly, so a small disagreement is expected and is
    the lattice resolution, not a bug.  Everything else must agree closely.
    """
    mesh = meshes[name]
    sample = corpus.to_sample(mesh)
    if mesh.uvs is None:
        # Give the mesh a real layout to measure: a plain XY projection,
        # normalised into the unit square.
        flat = sample.verts[sample.tri_verts][:, :, :2].reshape(-1, 2)
        lo, hi = flat.min(axis=0), flat.max(axis=0)
        sample = sample.with_uvs(0.02 + 0.96 * (flat - lo) / np.maximum(hi - lo, 1e-12))
    got = evaluate(sample, compute_overlap=False)
    want = reference_layout_metrics(sample)
    assert got.island_count == want["island_count"]
    assert got.uv_coverage == pytest.approx(want["uv_coverage"], rel=1e-9)
    assert got.seam_ratio == pytest.approx(want["seam_ratio"], rel=1e-9)
    assert got.texel_density_cv == pytest.approx(want["texel_density_cv"], rel=1e-6)
    assert got.out_of_bounds_tris == want["out_of_bounds_tris"]
    assert got.uv_bounds == pytest.approx(want["uv_bounds"], abs=1e-12)
    assert got.packing_efficiency == pytest.approx(want["packing_efficiency"], abs=0.02)
