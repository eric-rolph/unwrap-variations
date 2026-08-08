"""Shared fixtures and mesh builders for the bpy-free test suite.

Nothing in ``tests/`` may import :mod:`bpy`.  The whole suite runs under plain
pytest with only stdlib, numpy and pytest available, which is the point: the
core of the add-on is testable without launching Blender at all.  Anything that
genuinely needs Blender lives in ``tools/run_blender_tests.py``.

Two kinds of mesh feed the tests:

* **Corpus meshes** -- real Blender output exported to ``tests/data/*.npz`` as
  the exact flat arrays :class:`UVMeshSample` consumes.  Only a small,
  representative subset is committed; the trivial shapes are generated in code
  instead so the repository stays light.
* **Procedural meshes** -- built here, on demand, including the deliberately
  awful ones (zero-area triangles, NaN UVs, a mesh with no triangles at all)
  that a real corpus cannot contain.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"

from unwrap_variations.core.meshdata import UVMeshSample, make_sample  # noqa: E402

#: Corpus meshes committed to ``tests/data``.  Each is a real Blender mesh with
#: a real unwrap, chosen to cover one distinct failure mode:
#:
#: * ``cube_project``   - isometric projection; L2 and Linf must be exactly 1.0.
#: * ``nonmanifold``    - an edge with three incident faces; must never raise.
#: * ``hardsurface``    - a realistic mid-size mechanical part, 6 islands.
#: * ``sphere_smart``   - a good curved unwrap: real, moderate distortion.
#: * ``sphere_planar``  - the deliberate bad-unwrap control: half its triangles
#:                        flipped and heavy self-overlap.  Any metric suite that
#:                        does not flag this loudly is broken.
CORPUS_NAMES = (
    "cube_project",
    "nonmanifold",
    "hardsurface",
    "sphere_smart",
    "sphere_planar",
)

#: Corpus meshes small enough for the O(T^2) reference overlap pass.
SMALL_CORPUS_NAMES = ("cube_project", "nonmanifold", "hardsurface")


# ---------------------------------------------------------------------------
# Module availability.  Other modules are written in parallel with this suite.
# ---------------------------------------------------------------------------


def import_or_skip(dotted: str) -> ModuleType:
    """Import ``dotted``, skipping the test only if the file does not exist yet.

    This distinguishes the two cases that matter while the repository is being
    written in parallel:

    * the module's source file is **absent** -- another worker has not landed it
      yet, so the test is skipped with a clear reason;
    * the module's source file is **present but broken** -- that is a real bug,
      so the import error propagates and the test fails loudly.

    A blanket ``pytest.importorskip`` would silently hide the second case, which
    is exactly the case this suite exists to catch.
    """
    parts = dotted.split(".")
    path = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    pkg_init = REPO_ROOT.joinpath(*parts, "__init__.py")
    if not path.is_file() and not pkg_init.is_file():
        pytest.skip("%s has not been written yet (%s missing)" % (dotted, path))
    return importlib.import_module(dotted)


@pytest.fixture(scope="session")
def evaluator():
    """The vectorised evaluator module under test."""
    return import_or_skip("unwrap_variations.core.evaluator")


@pytest.fixture(scope="session")
def metrics_mod():
    """The metrics/scoring module under test."""
    return import_or_skip("unwrap_variations.core.metrics")


@pytest.fixture(scope="session")
def seams_mod():
    """The seam-heuristics module under test."""
    return import_or_skip("unwrap_variations.backends.seams")


@pytest.fixture(scope="session")
def candidate_mod():
    """The candidate/store module under test."""
    return import_or_skip("unwrap_variations.core.candidate")


# ---------------------------------------------------------------------------
# Corpus loading.
# ---------------------------------------------------------------------------


def load_corpus(name: str) -> UVMeshSample:
    """Load one committed corpus mesh as a :class:`UVMeshSample`.

    Args:
        name: Base name without the ``.npz`` suffix, e.g. ``"suzanne"``.

    Returns:
        The sample, with float64 positions and UVs.
    """
    path = DATA_DIR / ("%s.npz" % name)
    if not path.is_file():
        raise FileNotFoundError("missing corpus mesh %s" % path)
    data = np.load(path, allow_pickle=False)
    return UVMeshSample(
        verts=np.asarray(data["verts"], dtype=np.float64),
        tri_verts=np.asarray(data["tri_verts"], dtype=np.int32),
        tri_loops=np.asarray(data["tri_loops"], dtype=np.int32),
        uvs=np.asarray(data["uvs"], dtype=np.float64),
        edge_verts=np.asarray(data["edge_verts"], dtype=np.int32),
        edge_seam=np.asarray(data["edge_seam"], dtype=bool),
        loop_verts=np.asarray(data["loop_verts"], dtype=np.int32),
        name=name,
    )


@pytest.fixture(scope="session", params=CORPUS_NAMES)
def corpus_sample(request) -> UVMeshSample:
    """Every committed corpus mesh, one per test invocation."""
    return load_corpus(request.param)


@pytest.fixture(scope="session", params=SMALL_CORPUS_NAMES)
def small_corpus_sample(request) -> UVMeshSample:
    """Corpus meshes small enough for the quadratic reference overlap pass."""
    return load_corpus(request.param)


# ---------------------------------------------------------------------------
# Procedural meshes.
# ---------------------------------------------------------------------------


def make_flat_grid(n: int = 32, *, size: float = 1.0, uv_scale: float = 1.0) -> UVMeshSample:
    """Build an ``n x n`` quad grid triangulated into ``2 n^2`` triangles.

    The UV layout is the grid's own XY, so the map is **isometric** up to the
    global scale ``uv_scale``: normalised L2 and Linf stretch must both come out
    at exactly 1.0, and self-overlap must be exactly zero.  This is the case a
    separating-axis overlap test gets catastrophically wrong -- every one of the
    thousands of edge-adjacent triangle pairs is reported as overlapping.

    Args:
        n: Quads per side.
        size: World-space side length.
        uv_scale: Uniform scale applied to the UVs.  Stretch metrics are
            normalised and must be blind to it.

    Returns:
        A fully populated sample with per-corner (split) loops.
    """
    xs = np.linspace(0.0, size, n + 1)
    ys = np.linspace(0.0, size, n + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    def vid(i, j):
        return i * (n + 1) + j

    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))
    tri_verts = np.asarray(tris, dtype=np.int32)

    # One loop per triangle corner: the same layout Blender's loop_triangles
    # produce once a mesh has any UV seam at all.
    tri_loops = np.arange(tri_verts.size, dtype=np.int32).reshape(-1, 3)
    loop_verts = tri_verts.reshape(-1).astype(np.int32)
    uvs = (verts[loop_verts][:, :2] / max(size, 1e-12)) * uv_scale

    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=loop_verts,
        name="flat_grid_%d" % n,
    )


def make_unit_triangle(uv=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))) -> UVMeshSample:
    """Build the single right triangle ``(0,0,0) (1,0,0) (0,1,0)`` with given UVs."""
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return make_sample(
        verts,
        [[0, 1, 2]],
        uvs=np.asarray(uv, dtype=np.float64),
        name="unit_triangle",
    )


def make_rect_to_square() -> UVMeshSample:
    """Build the analytic 2x1-rectangle-into-a-unit-UV-square case.

    A 2 x 1 rectangle in 3D mapped onto a 1 x 1 UV square is uniformly squashed
    by 2 along one axis.  After the area normalisation that makes an isometric
    map score 1.0, the singular values are ``Gamma = sqrt(2)`` and
    ``gamma = 1/sqrt(2)`` (their product is 1, i.e. area is preserved), so the
    mesh-wide L2 stretch is ``sqrt((2 + 0.5) / 2) = 1.118034``.  Every number in
    that sentence is hand-computable, which is why this is the pinned case.
    """
    verts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    tri_verts = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    tri_loops = np.arange(6, dtype=np.int32).reshape(-1, 3)
    loop_verts = tri_verts.reshape(-1).astype(np.int32)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    uvs = square[loop_verts]
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=loop_verts,
        name="rect_to_square",
    )


def make_cube(*, uv_scale: float = 1.0) -> UVMeshSample:
    """Build a unit cube with a per-face planar (isometric) UV layout."""
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    quads = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    tri_verts = []
    uvs = []
    tri_loops = []
    loop_verts = []
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) * uv_scale
    for f, quad in enumerate(quads):
        offset_u = (f % 3) * 1.2 * uv_scale
        offset_v = (f // 3) * 1.2 * uv_scale
        quad_uv = unit + np.array([offset_u, offset_v])
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            base = len(loop_verts)
            tri_verts.append([quad[a], quad[b], quad[c]])
            tri_loops.append([base, base + 1, base + 2])
            for corner in (a, b, c):
                loop_verts.append(quad[corner])
                uvs.append(quad_uv[corner])
    return make_sample(
        corners,
        np.asarray(tri_verts, dtype=np.int32),
        tri_loops=np.asarray(tri_loops, dtype=np.int32),
        uvs=np.asarray(uvs, dtype=np.float64),
        loop_verts=np.asarray(loop_verts, dtype=np.int32),
        name="cube",
    )


def make_empty() -> UVMeshSample:
    """Build a mesh with no vertices, triangles, loops or edges."""
    return UVMeshSample(
        verts=np.zeros((0, 3), dtype=np.float64),
        tri_verts=np.zeros((0, 3), dtype=np.int32),
        tri_loops=np.zeros((0, 3), dtype=np.int32),
        uvs=np.zeros((0, 2), dtype=np.float64),
        edge_verts=np.zeros((0, 2), dtype=np.int32),
        edge_seam=np.zeros(0, dtype=bool),
        loop_verts=np.zeros(0, dtype=np.int32),
        name="empty",
    )


def make_all_identical(n_tris: int = 4) -> UVMeshSample:
    """Build a mesh whose vertices are all at the origin -- every triangle is a point."""
    verts = np.zeros((3, 3), dtype=np.float64)
    tri_verts = np.tile(np.array([[0, 1, 2]], dtype=np.int32), (n_tris, 1))
    tri_loops = np.arange(n_tris * 3, dtype=np.int32).reshape(-1, 3)
    uvs = np.zeros((n_tris * 3, 2), dtype=np.float64)
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=tri_verts.reshape(-1),
        name="all_identical",
    )


def make_zero_area_mix() -> UVMeshSample:
    """Build a mesh mixing valid triangles with 3D-degenerate and UV-degenerate ones."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear with 0 and 1 -> zero 3D area
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )
    tri_verts = np.array(
        [
            [0, 1, 2],  # good
            [0, 1, 3],  # zero 3D area (collinear)
            [4, 5, 6],  # good 3D, but zero UV area below
        ],
        dtype=np.int32,
    )
    tri_loops = np.arange(9, dtype=np.int32).reshape(-1, 3)
    uvs = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.2, 0.1],
            [0.5, 0.5],
            [0.5, 0.5],  # collapsed UV triangle
            [0.5, 0.5],
        ]
    )
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=tri_verts.reshape(-1),
        name="zero_area_mix",
    )


def make_nan_uvs() -> UVMeshSample:
    """Build a valid grid whose UV layer is poisoned with NaN and Inf."""
    sample = make_flat_grid(4)
    uvs = np.array(sample.uvs, dtype=np.float64)
    uvs[0] = (np.nan, 0.0)
    uvs[1] = (np.inf, -np.inf)
    uvs[7] = (0.0, np.nan)
    return sample.with_uvs(uvs)


def make_overlapping_pair(kind: str = "partial") -> UVMeshSample:
    """Build two coplanar triangles whose UVs relate in a controlled way.

    Args:
        kind: One of

            * ``"adjacent"`` -- the two halves of a unit square, sharing the
              diagonal.  True overlap area is exactly ``0``.  A separating-axis
              boolean gets this wrong.
            * ``"partial"`` -- two right triangles overlapping in the axis-
              aligned rectangle ``[0.5, 1] x [0, 0.5]``, area ``0.25`` exactly.
            * ``"contained"`` -- a small triangle entirely inside a large one;
              the overlap is the small triangle's own area, ``0.02``.
            * ``"identical"`` -- two copies of the same triangle; the overlap is
              its full area, ``0.5``.
    """
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ]
    )
    tri_verts = np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32)
    tri_loops = np.arange(6, dtype=np.int32).reshape(-1, 3)

    first = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    if kind == "adjacent":
        second = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    elif kind == "partial":
        # Intersection is exactly the rectangle [0.5, 1] x [0, 0.5]: area 0.25.
        second = [(0.5, 0.5), (1.5, 0.5), (0.5, -0.5)]
    elif kind == "contained":
        second = [(0.6, 0.2), (0.8, 0.2), (0.8, 0.4)]
    elif kind == "identical":
        second = list(first)
    else:
        raise ValueError("unknown kind %r" % kind)

    uvs = np.asarray(first + second, dtype=np.float64)
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=tri_verts.reshape(-1),
        name="overlap_%s" % kind,
    )


def make_random_mesh(seed: int, *, n_tris: int = 60, mode: str = "chaotic") -> UVMeshSample:
    """Build a random mesh for property testing.

    Args:
        seed: RNG seed; the mesh is fully determined by it.
        n_tris: Triangle count.
        mode: ``"chaotic"`` gives independent random 3D corners and random UVs
            (a nonsense unwrap, which is exactly what an invariant proof has to
            survive).  ``"jittered"`` perturbs an isometric grid's UVs, giving a
            plausible but imperfect unwrap.  ``"spiky"`` mixes in exactly-zero
            areas and huge magnitude differences.

    Returns:
        A sample.  No guarantee of manifoldness, winding or sanity -- that is
        the point.
    """
    rng = np.random.default_rng(seed)
    if mode == "jittered":
        side = max(2, int(math.sqrt(max(n_tris, 2) / 2)))
        base = make_flat_grid(side)
        uvs = np.array(base.uvs, dtype=np.float64)
        # Jitter scaled to a fraction of the cell size, so the result is a
        # plausible imperfect unwrap rather than confetti.
        uvs += rng.normal(0.0, 0.12 / side, size=uvs.shape)
        return base.with_uvs(uvs).with_seams(
            rng.random(base.n_edges) < 0.15
        )

    n_verts = max(3, n_tris)
    verts = rng.normal(0.0, 1.0, size=(n_verts, 3))
    tri_verts = rng.integers(0, n_verts, size=(n_tris, 3)).astype(np.int32)
    tri_loops = np.arange(n_tris * 3, dtype=np.int32).reshape(-1, 3)
    uvs = rng.random((n_tris * 3, 2))
    if mode == "spiky":
        scale = 10.0 ** rng.integers(-4, 5, size=(n_verts, 1))
        verts = verts * scale
        # Collapse a tenth of the triangles to a point in UV space.
        collapse = rng.random(n_tris) < 0.1
        for i in np.nonzero(collapse)[0]:
            uvs[tri_loops[i]] = uvs[tri_loops[i][0]]
        # And a tenth to a point in 3D.
        collapse3 = rng.random(n_tris) < 0.1
        for i in np.nonzero(collapse3)[0]:
            tri_verts[i] = tri_verts[i][0]
    return make_sample(
        verts,
        tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        loop_verts=tri_verts.reshape(-1),
        name="random_%s_%d" % (mode, seed),
    )


#: Every procedural mesh a "must not raise" test should sweep.
def all_pathological() -> list[UVMeshSample]:
    """Return the full set of deliberately broken meshes."""
    return [
        make_empty(),
        make_all_identical(),
        make_zero_area_mix(),
        make_nan_uvs(),
        make_unit_triangle(),
        make_unit_triangle(uv=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
        make_random_mesh(11, mode="spiky"),
    ]


@pytest.fixture
def flat_grid() -> UVMeshSample:
    """A 32x32 isometric grid: 2048 triangles, L2 == 1.0, zero overlap."""
    return make_flat_grid(32)


@pytest.fixture
def rect_to_square() -> UVMeshSample:
    """The analytic 2x1-rectangle case with a known L2 of 1.118034."""
    return make_rect_to_square()


@pytest.fixture
def cube() -> UVMeshSample:
    """A unit cube with a per-face isometric UV layout."""
    return make_cube()
