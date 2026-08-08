#!/usr/bin/env python3
"""The organic-mesh corpus: eight hard surfaces, built in-script, no downloads.

``docs/ORGANIC.md`` section 8.1 requires a corpus of *organic* meshes.  Blender's
primitives are not organic -- a sphere, a torus and a cube are exactly the shapes
every unwrapper is already tuned for -- and Suzanne on her own is one data point,
not a corpus.  This module builds eight surfaces that between them cover the
failure modes an organic mesh actually presents:

======================  =====================================================
mesh                    why it is hard
======================  =====================================================
``suzanne``             creases, ears, two boundary loops, three shells; the
                        reference case that every number in ORGANIC.md
                        section 0 was measured against
``suzanne_sub2``        the same shape at ~15.7k triangles; tests scaling of
                        everything that is quadratic or worse
``blob_limbs``          metaball creature with five protruding limbs; tests
                        chart-per-limb decomposition, where a single chart
                        cannot be flattened without enormous stretch
``displaced_sphere``    noise-displaced closed genus 0 with **no sharp edge
                        anywhere**; the ``SHARP`` seam strategy selects
                        nothing at all here, so a cut has to come from
                        somewhere else
``torus_knot``          genus 1 with an organic varying cross-section; needs
                        two cuts before any parameterisation exists, and the
                        curvature hides where they should go
``hand``                five protrusions *close together*; the fingers'
                        charts collide in UV space even when each one is
                        individually fine -- the cross-island collision case
``folded_ribbon``       ships a UV layout that is locally injective
                        everywhere and globally non-injective: every signed
                        area is positive and the chart boundary still crosses
                        itself.  This is Suzanne's defect (ORGANIC.md section
                        0.3) reduced to its minimal reproducible form, and it
                        is the case a flip-only check cannot see
``pinched``             exact zero-area faces, near-degenerate slivers and
                        coincident vertices; robustness only -- nothing may
                        crash, and no metric may return NaN
======================  =====================================================

Determinism
-----------

Every mesh except the two Suzannes is built from explicit vertex arrays by
straight-line numpy: no ``random``, no Blender operator output, no asset
download.  The implicit surfaces (``blob_limbs``, ``hand``) go through a
marching-tetrahedra polygoniser written here, whose field kernels use only
``+ - * /`` -- all IEEE-exact, and deliberately not ``np.dot``, which goes to
BLAS and sums in a vendor-dependent order.  Measured, not assumed:

* Blender 4.5.4 (numpy 1.26.4, Python 3.11) and Blender 5.1.2 (numpy 2.3.4,
  Python 3.13) produce **bit-identical** vertex and triangle arrays for all
  eight meshes -- Suzanne and her subdivided form included.
* Against unrelated numpy builds (2.2.6 and 2.5.1 on pip's scipy-openblas),
  connectivity is bit-identical everywhere and five of the six numpy meshes are
  bit-identical outright.  Only ``torus_knot`` still drifts, by **1 ULP**
  (2.2e-16), and the reason is measured rather than guessed: isolating the
  pipeline step by step, ``np.linalg.norm`` over an axis, ``np.cos``, ``np.sin``
  and ``np.einsum`` all hash identically on both builds, and every divergence
  traces to a **BLAS call** -- ``np.dot`` on the 3-vectors inside
  :func:`_rotation_minimising_frame`, and ``np.linalg.norm`` on a single vector,
  which goes to ``dnrm2``.  ``displaced_sphere`` and ``pinched`` used to drift
  for the same reason and no longer do; see :func:`_band_limited_bumps`.  An
  earlier version of this note blamed ``np.linalg.norm`` and libm's ``cos``.
  That was wrong, and it was wrong in the direction that would have stopped
  anybody fixing it.

Each entry carries its **expected invariants** -- vertex and triangle count,
shell count, Euler characteristic and boundary-edge count per shell -- plus a
content digest.  :func:`build` verifies the invariants, so a corpus regression
is caught the moment the corpus is built, not three scripts later when a
benchmark cell moves.

Run standalone to print the corpus table::

    python tools/organic_corpus.py                 # the six numpy meshes
    blender -b --factory-startup --python tools/organic_corpus.py   # all eight
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

__all__ = [
    "CorpusMesh",
    "Invariants",
    "ShellTopology",
    "NAMES",
    "NUMPY_NAMES",
    "BLENDER_NAMES",
    "WHY_HARD",
    "EXPECTED",
    "mesh_topology",
    "invariants_of",
    "build",
    "build_all",
    "to_sample",
    "to_object",
    "folded_ribbon_flat_uvs",
    "RIBBON_GRID",
    "tri_clip_area_uv",
    "tri_signed_area_uv",
    "overlapping_triangle_mask",
]

#: Every mesh in the corpus, in the order ORGANIC.md section 8.1 lists them.
NAMES = (
    "suzanne",
    "suzanne_sub2",
    "blob_limbs",
    "displaced_sphere",
    "torus_knot",
    "hand",
    "folded_ribbon",
    "pinched",
)

#: The subset buildable with numpy alone, i.e. outside Blender.
NUMPY_NAMES = (
    "blob_limbs",
    "displaced_sphere",
    "torus_knot",
    "hand",
    "folded_ribbon",
    "pinched",
)

#: The subset that needs :mod:`bpy` (Suzanne is a Blender asset).
BLENDER_NAMES = ("suzanne", "suzanne_sub2")

#: One line per mesh saying what it is testing.  Printed by every report so a
#: reader never has to guess why a mesh is in the corpus.
WHY_HARD = {
    "suzanne": "creases, ears, two boundary loops, three shells -- the reference case",
    "suzanne_sub2": "same shape at 15.7k tris -- tests scaling",
    "blob_limbs": "metaball creature, 5 protrusions -- chart-per-limb decomposition",
    "displaced_sphere": "noise-displaced closed genus 0 -- no sharp edge anywhere",
    "torus_knot": "genus 1, organic cross-section -- topology plus curvature",
    "hand": "5 close-together protrusions -- cross-island collision",
    "folded_ribbon": "ships a locally injective, globally non-injective UV layout",
    "pinched": "zero-area and near-degenerate faces -- robustness, must not crash",
}


# ---------------------------------------------------------------------------
# Invariants and the topology computation that produces them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellTopology:
    """Euler data for one connected component of a triangle soup.

    Attributes:
        faces: Triangle count.
        edges: Distinct undirected edge count.
        verts: Distinct vertex count.
        chi: ``V - E + F``.  ``2`` is a closed sphere, ``1`` a disk, ``0`` a
            torus *or* a genus-0 surface with two boundary loops -- which is
            why ``boundary_edges`` is reported alongside it and never folded in.
        boundary_edges: Edges with exactly one incident triangle.
        nonmanifold_edges: Edges with three or more incident triangles.
    """

    faces: int
    edges: int
    verts: int
    chi: int
    boundary_edges: int
    nonmanifold_edges: int


@dataclass(frozen=True)
class Invariants:
    """The build-time fingerprint of a corpus mesh.

    Attributes:
        n_verts: Vertex count.
        n_tris: Triangle count.
        shells: Number of connected components.
        chi: Euler characteristic per shell, ordered by descending face count.
        boundary_edges: Boundary-edge count per shell, in the same order.
        nonmanifold_edges: Total non-manifold edges across all shells.
    """

    n_verts: int
    n_tris: int
    shells: int
    chi: tuple[int, ...]
    boundary_edges: tuple[int, ...]
    nonmanifold_edges: int

    def describe(self) -> str:
        """Return a one-line ASCII summary."""
        return "V=%d T=%d shells=%d chi=%s bnd=%s nonmanifold=%d" % (
            self.n_verts,
            self.n_tris,
            self.shells,
            ",".join(str(c) for c in self.chi),
            ",".join(str(b) for b in self.boundary_edges),
            self.nonmanifold_edges,
        )


def mesh_topology(tris: np.ndarray, n_verts: int) -> list[ShellTopology]:
    """Return the per-shell Euler data of a triangle soup.

    Vectorised counterpart of the deliberately naive
    ``tests.reference.reference_shell_topology``; the two must agree on every
    corpus mesh, and a test asserts that they do.  Working from the triangle
    soup rather than from a half-edge structure is on purpose: it cannot
    inherit a bug from the connectivity code it is supposed to be checking.

    Args:
        tris: ``(T, 3)`` vertex indices.
        n_verts: Number of vertex slots.  Unreferenced vertices are ignored --
            they belong to no shell and would otherwise invent one each.

    Returns:
        One :class:`ShellTopology` per connected component, ordered by
        descending face count then ascending lowest vertex index, so the order
        is deterministic and does not depend on the union-find's internals.
    """
    tv = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if tv.shape[0] == 0:
        return []
    n = max(int(n_verts), int(tv.max()) + 1)

    # Connected components of the vertex graph, by label propagation along
    # triangle corners.  No union-find: propagating "smallest label seen by any
    # neighbour" and then pointer-jumping (``label[label]``) converges in
    # O(log diameter) rounds and is obviously correct at a glance, which matters
    # for a routine whose job is to catch other code being wrong.
    ea = np.concatenate([tv[:, 0], tv[:, 1], tv[:, 2]])
    eb = np.concatenate([tv[:, 1], tv[:, 2], tv[:, 0]])
    src = np.concatenate([ea, eb])
    dst = np.concatenate([eb, ea])
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    dst_s = dst[order]
    starts = np.flatnonzero(np.concatenate([[True], src_s[1:] != src_s[:-1]]))
    groups = src_s[starts]

    label = np.arange(n, dtype=np.int64)
    while True:
        nxt = label.copy()
        nxt[groups] = np.minimum(nxt[groups], np.minimum.reduceat(label[dst_s], starts))
        nxt = nxt[nxt]
        if np.array_equal(nxt, label):
            break
        label = nxt

    pairs = np.sort(np.stack([ea, eb], axis=1), axis=1)
    keys = pairs[:, 0] * n + pairs[:, 1]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    edge_v0 = uniq // n
    edge_shell = label[edge_v0]

    tri_shell = label[tv[:, 0]]
    shells = np.unique(tri_shell)

    out: list[ShellTopology] = []
    for s in shells:
        face_mask = tri_shell == s
        edge_mask = edge_shell == s
        vert_ids = np.unique(tv[face_mask].reshape(-1))
        ecounts = counts[edge_mask]
        out.append(
            ShellTopology(
                faces=int(face_mask.sum()),
                edges=int(edge_mask.sum()),
                verts=int(vert_ids.size),
                chi=int(vert_ids.size) - int(edge_mask.sum()) + int(face_mask.sum()),
                boundary_edges=int((ecounts == 1).sum()),
                nonmanifold_edges=int((ecounts >= 3).sum()),
            )
        )
    out.sort(key=lambda s: (-s.faces, -s.verts))
    return out


def invariants_of(verts: np.ndarray, tris: np.ndarray) -> Invariants:
    """Return the :class:`Invariants` of a mesh given as loose arrays."""
    shells = mesh_topology(tris, int(np.asarray(verts).shape[0]))
    return Invariants(
        n_verts=int(np.asarray(verts).shape[0]),
        n_tris=int(np.asarray(tris).reshape(-1, 3).shape[0]),
        shells=len(shells),
        chi=tuple(s.chi for s in shells),
        boundary_edges=tuple(s.boundary_edges for s in shells),
        nonmanifold_edges=sum(s.nonmanifold_edges for s in shells),
    )


@dataclass(frozen=True)
class CorpusMesh:
    """One built corpus mesh plus everything a harness needs to use it.

    Attributes:
        name: Corpus key, one of :data:`NAMES`.
        why_hard: The one-line rationale from :data:`WHY_HARD`.
        verts: ``(V, 3)`` float64 positions.
        tris: ``(T, 3)`` int32 vertex indices.
        uvs: ``(T, 3, 2)`` float64 per-corner UVs, or ``None`` when the mesh
            ships no layout of its own.  Only ``folded_ribbon`` sets this, and
            it does so to carry a deliberately non-injective layout.
        invariants: Measured at build time and checked against
            :data:`EXPECTED`.
        source: ``"numpy"`` or ``"blender"``.
        faces: The mesh's **original** polygons as a list of vertex-index
            tuples, when it has any that are not already triangles.  Suzanne is
            a quad mesh; rebuilding her from ``tris`` would hand Blender a
            968-face triangle soup with 1473 edges in place of a 500-face quad
            mesh with 1005, and every seam, chart and packing decision would
            differ.  The resulting ``suzanne`` row would then quietly mean
            something other than every other Suzanne number in the repository,
            including all of ORGANIC.md section 0.  ``None`` when the mesh is
            genuinely triangles, which is every procedural entry.
    """

    name: str
    why_hard: str
    verts: np.ndarray
    tris: np.ndarray
    uvs: np.ndarray | None
    invariants: Invariants
    source: str
    faces: list | None = None

    @property
    def n_tris(self) -> int:
        """Triangle count."""
        return int(self.tris.shape[0])

    @property
    def n_verts(self) -> int:
        """Vertex count."""
        return int(self.verts.shape[0])

    def digest(self) -> str:
        """Return a 12-hex-digit content hash of the geometry.

        Invariants prove the *topology* did not move; this proves the
        *coordinates* did not either.  Two Blender versions printing the same
        digest is the strongest determinism claim this corpus can make, and it
        is checked rather than asserted: 4.5.4 ships numpy 1.26 on Python 3.11
        and 5.1.2 ships numpy 2.3 on Python 3.13.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.verts, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.tris, dtype=np.int32).tobytes())
        if self.uvs is not None:
            h.update(np.ascontiguousarray(self.uvs, dtype=np.float64).tobytes())
        return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Marching tetrahedra -- the implicit-surface polygoniser
# ---------------------------------------------------------------------------

#: Kuhn decomposition of a cube into six tetrahedra, all sharing the main
#: diagonal 0-6.  It is *conforming*: two cubes sharing a face agree on how that
#: face is split, so the extracted surface is watertight with no crack repair.
#: Corner order is the usual (i,j,k) walk 000,100,110,010,001,101,111,011.
_CUBE_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))

#: The six edges of a tetrahedron, as local corner pairs.
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _build_tet_table() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tri_count[16], tri_table[16, 2, 3])`` for marching tetrahedra.

    Generated here rather than typed out.  There are only three topological
    cases inside a tetrahedron -- one corner in, two in, three in -- so the
    table is four lines of logic, and a generated table cannot contain the
    transcription error that a 256-entry marching-cubes table invariably does.
    """
    edge_id = {}
    for e, (a, b) in enumerate(_TET_EDGES):
        edge_id[(a, b)] = e
        edge_id[(b, a)] = e

    count = np.zeros(16, dtype=np.int64)
    table = np.full((16, 2, 3), -1, dtype=np.int64)
    for code in range(16):
        inside = [i for i in range(4) if code & (1 << i)]
        outside = [i for i in range(4) if not code & (1 << i)]
        if not inside or not outside:
            continue
        if len(inside) == 1:
            i = inside[0]
            tris = [[edge_id[(i, o)] for o in outside]]
        elif len(inside) == 3:
            o = outside[0]
            tris = [[edge_id[(o, i)] for i in inside]]
        else:
            i0, i1 = inside
            o0, o1 = outside
            quad = [edge_id[(i0, o0)], edge_id[(i0, o1)], edge_id[(i1, o1)], edge_id[(i1, o0)]]
            tris = [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
        count[code] = len(tris)
        for t, tri in enumerate(tris):
            table[code, t] = tri
    return count, table


_TET_TRI_COUNT, _TET_TRI_TABLE = _build_tet_table()

#: Crossing parameters are clamped away from the grid nodes.  Without this an
#: edge whose far value sits exactly on the isolevel emits a vertex *at* a grid
#: node, which two different edges can then both produce -- a zero-area triangle
#: and, worse, a non-manifold vertex.  Clamping is applied per edge, so both
#: tetrahedra sharing that edge still compute the identical position and the
#: surface stays watertight.
_T_CLAMP = 1e-6


def _marching_tets(
    field,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
    res,
    iso: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Polygonise ``{field == iso}`` over a regular grid, watertight and manifold.

    Args:
        field: Callable taking ``(N, 3)`` float64 points and returning ``(N,)``
            field values.  Must be built from IEEE-exact operations only if the
            result is to be bit-identical across numpy versions.
        lo: Lower grid corner; must strictly enclose the surface.
        hi: Upper grid corner.
        res: Node count per axis, either one int or a ``(nx, ny, nz)`` triple.
            Per-axis counts exist so a flat shape can be resolved finely where
            its detail is without paying for it in the thin direction.
        iso: Isolevel.  A point is *inside* when ``field(p) > iso``, tested
            strictly, so a node exactly on the level counts as outside and no
            edge can have two equal endpoint values while still crossing.

    Returns:
        ``(verts, tris)`` with ``verts`` ``(V, 3)`` float64 and ``tris``
        ``(T, 3)`` int32, oriented outward (normals along ``-grad field``).
    """
    nx, ny, nz = (int(res),) * 3 if isinstance(res, int) else tuple(int(v) for v in res)
    xs = np.linspace(lo[0], hi[0], nx, dtype=np.float64)
    ys = np.linspace(lo[1], hi[1], ny, dtype=np.float64)
    zs = np.linspace(lo[2], hi[2], nz, dtype=np.float64)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=1)
    values = np.asarray(field(nodes), dtype=np.float64).reshape(-1)
    n_nodes = nodes.shape[0]

    ci, cj, ck = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    ci = ci.reshape(-1)
    cj = cj.reshape(-1)
    ck = ck.reshape(-1)
    corner_lin = np.empty((ci.size, 8), dtype=np.int64)
    for c, (di, dj, dk) in enumerate(_CUBE_CORNERS):
        corner_lin[:, c] = ((ci + di) * ny + (cj + dj)) * nz + (ck + dk)

    tet_nodes = np.concatenate([corner_lin[:, list(t)] for t in _TETS], axis=0)
    inside = values[tet_nodes] > iso
    code = (
        inside[:, 0].astype(np.int64)
        + 2 * inside[:, 1].astype(np.int64)
        + 4 * inside[:, 2].astype(np.int64)
        + 8 * inside[:, 3].astype(np.int64)
    )
    live = np.flatnonzero((code > 0) & (code < 15))
    if live.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    tet_nodes = tet_nodes[live]
    code = code[live]
    counts = _TET_TRI_COUNT[code]

    edge_a = np.asarray([e[0] for e in _TET_EDGES], dtype=np.int64)
    edge_b = np.asarray([e[1] for e in _TET_EDGES], dtype=np.int64)

    key_rows = []
    for slot in (0, 1):
        sel = np.flatnonzero(counts > slot)
        if sel.size == 0:
            continue
        eids = _TET_TRI_TABLE[code[sel], slot]  # (n, 3)
        pa = np.take_along_axis(tet_nodes[sel], edge_a[eids], axis=1)
        pb = np.take_along_axis(tet_nodes[sel], edge_b[eids], axis=1)
        key_rows.append(np.minimum(pa, pb) * n_nodes + np.maximum(pa, pb))
    keys = np.concatenate(key_rows, axis=0)

    uniq, inv = np.unique(keys.reshape(-1), return_inverse=True)
    tris = inv.reshape(-1, 3).astype(np.int64)

    p = uniq // n_nodes
    q = uniq % n_nodes
    vp = values[p]
    vq = values[q]
    denom = vq - vp
    t = np.where(np.abs(denom) > 0.0, (iso - vp) / np.where(denom == 0.0, 1.0, denom), 0.5)
    t = np.clip(t, _T_CLAMP, 1.0 - _T_CLAMP)
    verts = nodes[p] + t[:, None] * (nodes[q] - nodes[p])

    cell = min((hi[0] - lo[0]) / (nx - 1), (hi[1] - lo[1]) / (ny - 1), (hi[2] - lo[2]) / (nz - 1))
    tris = _orient_outward(verts, tris, field, cell)
    return verts, tris.astype(np.int32)


def _orient_outward(verts: np.ndarray, tris: np.ndarray, field, h: float) -> np.ndarray:
    """Flip triangles so every normal points outward, consistently.

    Two passes, and the second is what makes it correct.

    **Locally, from the field gradient.**  ``field`` decreases outward by
    convention (inside is ``field > iso``), so the outward normal is ``-grad``.
    On a well-resolved surface away from features this is exact.

    **Then globally, by propagation.**  The gradient test alone is *not* enough,
    and the failure is not hypothetical: where two parts of the surface nearly
    touch, the field has a saddle and its gradient is small and badly
    conditioned, so the sign comes out wrong on a handful of triangles.
    Measured on the ``hand`` mesh, whose fingers nearly meet: **22 triangles
    were wound against the rest**, giving 30 manifold edges traversed
    identically by both incident faces.  The consequence landed far downstream
    -- the flattener reported **737 flipped triangles and 18 boundary
    crossings** on that mesh's charts, which reads as a solver defect and is
    actually a corpus defect.  Correcting the winding gives **0 flips and 0
    crossings** on the same 17 charts.

    So a breadth-first pass over the face adjacency makes every shell
    internally consistent -- a shared manifold edge must be traversed in
    *opposite* directions by its two faces -- and the gradient is then used only
    to pick each shell's global sign, by majority vote.  A majority cannot be
    swung by a few bad gradients, which is exactly the failure mode above.
    """
    if tris.shape[0] == 0:
        return tris

    out = np.array(tris, dtype=tris.dtype, copy=True)
    n_tris = out.shape[0]

    # -- pass 1: make each shell internally consistent -----------------------
    # Map each undirected edge to the faces on it. A consistently wound pair
    # traverses their shared edge in opposite directions.
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for t in range(n_tris):
        a, b, c = int(out[t, 0]), int(out[t, 1]), int(out[t, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((u, v) if u < v else (v, u), []).append(t)

    seen = np.zeros(n_tris, dtype=bool)
    for seed in range(n_tris):
        if seen[seed]:
            continue
        seen[seed] = True
        stack = [seed]
        while stack:
            t = stack.pop()
            a, b, c = int(out[t, 0]), int(out[t, 1]), int(out[t, 2])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                for other in edge_faces.get(key, ()):
                    if other == t or seen[other]:
                        continue
                    # Non-manifold edges (3+ faces) carry no orientation
                    # information; skip rather than guess.
                    if len(edge_faces[key]) != 2:
                        continue
                    oa, ob, oc = int(out[other, 0]), int(out[other, 1]), int(out[other, 2])
                    same = ((oa, ob) == (u, v) or (ob, oc) == (u, v) or (oc, oa) == (u, v))
                    if same:
                        out[other] = out[other][[0, 2, 1]]
                    seen[other] = True
                    stack.append(other)

    # -- pass 2: choose each shell's global sign from the field --------------
    step = max(float(h) * 0.25, 1e-9)
    centroid = verts[out].mean(axis=1)
    grad = np.empty_like(centroid)
    for axis in range(3):
        d = np.zeros(3, dtype=np.float64)
        d[axis] = step
        grad[:, axis] = (
            np.asarray(field(centroid + d)) - np.asarray(field(centroid - d))
        ) / (2.0 * step)
    normal = np.cross(verts[out[:, 1]] - verts[out[:, 0]], verts[out[:, 2]] - verts[out[:, 0]])
    agrees = np.einsum("ij,ij->i", normal, -grad) >= 0.0

    shell = _face_shell_labels(out, edge_faces, n_tris)
    for label in np.unique(shell):
        member = shell == label
        # Majority vote, so a few badly-conditioned gradients cannot flip a
        # whole shell. Ties keep the current winding.
        if int(np.count_nonzero(agrees[member])) * 2 < int(np.count_nonzero(member)):
            out[member] = out[member][:, [0, 2, 1]]
    return out


def _face_shell_labels(tris: np.ndarray, edge_faces: dict, n_tris: int) -> np.ndarray:
    """Return a connected-component label per face, joined across shared edges."""
    parent = list(range(n_tris))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for faces in edge_faces.values():
        first = faces[0]
        for other in faces[1:]:
            ra, rb = find(first), find(other)
            if ra != rb:
                parent[rb] = ra
    labels = np.array([find(t) for t in range(n_tris)], dtype=np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    return labels


# ---------------------------------------------------------------------------
# Implicit field kernels -- IEEE-exact, so bit-identical across numpy versions
# ---------------------------------------------------------------------------


def _segment_distance_sq(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the squared distance from each point to the segment ``a-b``.

    The dot products are written out as three multiplies and two adds rather
    than handed to ``np.dot``.  That is not style: ``np.dot`` on a float64 array
    goes to BLAS, and two numpy builds with different BLAS vendors sum in a
    different order, which moves the answer by an ULP.  An ULP is enough to
    change which side of the isolevel a grid node falls on, and therefore enough
    to change the polygonised connectivity.  Written out, the field is
    reproducible on any numpy build in existence.
    """
    ab = b - a
    denom = float(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
    ap = points - a
    if denom <= 0.0:
        return ap[:, 0] * ap[:, 0] + ap[:, 1] * ap[:, 1] + ap[:, 2] * ap[:, 2]
    proj = ap[:, 0] * ab[0] + ap[:, 1] * ab[1] + ap[:, 2] * ab[2]
    t = np.clip(proj / denom, 0.0, 1.0)
    d = ap - t[:, None] * ab
    return d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]


def _capsule_kernel(points: np.ndarray, a, b, radius: float, stiffness: float) -> np.ndarray:
    """Blender's metaball falloff around a line segment.

    ``(1 - r^2/R^2)^3 * stiffness`` inside the radius and zero outside -- the
    same Wyvill-style kernel Blender's own metaballs use, which is why the
    result looks like a metaball creature rather than a union of capsules.  It
    is a cubic in ``r^2``, so it never calls a transcendental function and the
    field is reproducible to the last bit.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    r2 = _segment_distance_sq(points, a, b) / (float(radius) * float(radius))
    u = np.clip(1.0 - r2, 0.0, None)
    return float(stiffness) * u * u * u


def _blob_field(balls) -> "callable":
    """Return a summed metaball field callable for a list of capsule kernels."""

    def field(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        total = np.zeros(pts.shape[0], dtype=np.float64)
        for a, b, radius, stiffness in balls:
            total += _capsule_kernel(pts, a, b, radius, stiffness)
        return total

    return field


# ---------------------------------------------------------------------------
# Analytic builders
# ---------------------------------------------------------------------------

#: Golden angle, used to spread directions over a sphere without an RNG.
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return the unit icosahedron as ``(verts, tris)``."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            raw.append((0.0, s1 * 1.0, s2 * phi))
            raw.append((s1 * 1.0, s2 * phi, 0.0))
            raw.append((s2 * phi, 0.0, s1 * 1.0))
    verts = np.asarray(raw, dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    # Every pair at the minimum separation is an edge; every triangle of mutually
    # adjacent vertices is a face.  Derived rather than typed, so there is no
    # index list to get wrong.
    d = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    near = d < d.min() * 1.05
    tris = []
    for i in range(12):
        for j in range(i + 1, 12):
            if not near[i, j]:
                continue
            for k in range(j + 1, 12):
                if near[i, k] and near[j, k]:
                    tris.append((i, j, k))
    tv = np.asarray(sorted(tris), dtype=np.int64)
    normal = np.cross(verts[tv[:, 1]] - verts[tv[:, 0]], verts[tv[:, 2]] - verts[tv[:, 0]])
    centre = verts[tv].mean(axis=1)
    flip = np.einsum("ij,ij->i", normal, centre) < 0.0
    tv[flip] = tv[flip][:, [0, 2, 1]]
    return verts, tv


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit icosphere by repeated 1-to-4 midpoint subdivision.

    Deterministic: midpoints are deduplicated on the sorted endpoint pair via
    ``np.unique``, so the vertex order depends only on the topology.
    """
    verts, tris = _icosahedron()
    for _ in range(int(subdivisions)):
        n = verts.shape[0]
        pairs = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
        pairs = np.sort(pairs, axis=1)
        keys = pairs[:, 0] * n + pairs[:, 1]
        uniq, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        mid = 0.5 * (verts[uniq // n] + verts[uniq % n])
        mid /= np.linalg.norm(mid, axis=1)[:, None]
        verts = np.concatenate([verts, mid], axis=0)
        m = tris.shape[0]
        a = n + inv[:m]
        b = n + inv[m: 2 * m]
        c = n + inv[2 * m:]
        tris = np.concatenate(
            [
                np.stack([tris[:, 0], a, c], axis=1),
                np.stack([a, tris[:, 1], b], axis=1),
                np.stack([c, b, tris[:, 2]], axis=1),
                np.stack([a, b, c], axis=1),
            ],
            axis=0,
        )
    return verts, tris.astype(np.int64)


def _band_limited_bumps(directions: np.ndarray, n_terms: int) -> "callable":
    """Return a smooth, single-valued displacement function on the sphere.

    ``sum_k w_k * cos(f_k * <d_k, n>)`` is band-limited and infinitely
    differentiable, so the displaced sphere has **no sharp edge anywhere** --
    which is the whole point of that corpus entry.  The directions come from a
    golden-angle spiral and the frequencies from a fixed arithmetic sequence, so
    there is no RNG to seed.

    The projection is written out as three multiplies and two adds for the same
    reason :func:`_segment_distance_sq` is, and it was found the same way.  It
    used to read ``normals @ directions.T``, which is a BLAS ``dgemm``: measured,
    that single expression is the **entire** reason ``displaced_sphere`` and
    ``pinched`` were not bit-identical between Blender's numpy and a pip wheel.
    Isolated step by step, the icosphere vertices (``np.linalg.norm``), the
    golden-angle directions (``np.cos``) and the knot's curve (``np.cos``,
    ``np.sin``) all hash identically across both builds; only the matmul's
    output diverges, by 1 ULP (2.2e-16).  Written out, the projection is
    bit-identical on both -- and identical to what the matmul produced under
    Blender, so no corpus digest moves on a supported build.
    """
    freqs = np.asarray([3.0 + 1.7 * k for k in range(n_terms)], dtype=np.float64)
    weights = np.asarray([1.0 / (1.0 + 0.85 * k) for k in range(n_terms)], dtype=np.float64)
    weights = weights / float(np.abs(weights).sum())

    def displace(normals: np.ndarray) -> np.ndarray:
        proj = (
            normals[:, 0:1] * directions[None, :, 0]
            + normals[:, 1:2] * directions[None, :, 1]
            + normals[:, 2:3] * directions[None, :, 2]
        )  # (V, K)
        return (np.cos(proj * freqs[None, :]) * weights[None, :]).sum(axis=1)

    return displace


def _fibonacci_directions(count: int) -> np.ndarray:
    """Return ``count`` well-spread unit vectors from a golden-angle spiral."""
    idx = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / float(count)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = _GOLDEN * idx
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _grid_tris(n_u: int, n_v: int, wrap_u: bool, wrap_v: bool) -> np.ndarray:
    """Triangulate a ``n_u x n_v`` vertex grid, optionally wrapping either axis.

    Vertex ``(i, j)`` is at linear index ``i * n_v + j``.  Both diagonals of a
    quad are equivalent for topology; the ``(0, 1, 2) / (0, 2, 3)`` split is
    used consistently so the result is deterministic.
    """
    iu = np.arange(n_u if wrap_u else n_u - 1)
    iv = np.arange(n_v if wrap_v else n_v - 1)
    ii, jj = np.meshgrid(iu, iv, indexing="ij")
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    i1 = (ii + 1) % n_u
    j1 = (jj + 1) % n_v
    a = ii * n_v + jj
    b = i1 * n_v + jj
    c = i1 * n_v + j1
    d = ii * n_v + j1
    return np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
    ).astype(np.int64)


def _rotation_minimising_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tangent, normal, binormal)`` along a closed curve.

    Uses the double-reflection method (Wang et al. 2008), which is stable where
    a Frenet frame flips at an inflection, then removes the residual holonomy by
    spreading it evenly.  Without the correction a swept tube closes with a
    visible twist discontinuity, which would put an artificial crease on a mesh
    whose entire purpose is to have none.
    """
    n = points.shape[0]
    nxt = np.roll(points, -1, axis=0)
    prv = np.roll(points, 1, axis=0)
    tangent = nxt - prv
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]

    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, tangent[0]))) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    normal = np.empty_like(points)
    normal[0] = seed - float(np.dot(seed, tangent[0])) * tangent[0]
    normal[0] /= np.linalg.norm(normal[0])
    for i in range(n - 1):
        v1 = points[i + 1] - points[i]
        c1 = float(np.dot(v1, v1))
        if c1 <= 0.0:
            normal[i + 1] = normal[i]
            continue
        r_l = normal[i] - (2.0 / c1) * float(np.dot(v1, normal[i])) * v1
        t_l = tangent[i] - (2.0 / c1) * float(np.dot(v1, tangent[i])) * v1
        v2 = tangent[i + 1] - t_l
        c2 = float(np.dot(v2, v2))
        nxt_n = r_l if c2 <= 0.0 else r_l - (2.0 / c2) * float(np.dot(v2, r_l)) * v2
        nxt_n = nxt_n - float(np.dot(nxt_n, tangent[i + 1])) * tangent[i + 1]
        normal[i + 1] = nxt_n / np.linalg.norm(nxt_n)
    binormal = np.cross(tangent, normal)

    # Holonomy: how far the transported frame has rotated by the time it comes
    # back round.  Unwind it linearly so ring n-1 lines up with ring 0 exactly.
    closing = normal[0] - float(np.dot(normal[0], tangent[0])) * tangent[0]
    closing /= np.linalg.norm(closing)
    residual = math.atan2(float(np.dot(closing, binormal[-1])), float(np.dot(closing, normal[-1])))
    twist = residual * np.arange(n, dtype=np.float64) / float(n)
    cos_t = np.cos(twist)[:, None]
    sin_t = np.sin(twist)[:, None]
    return tangent, normal * cos_t + binormal * sin_t, -normal * sin_t + binormal * cos_t


# ---------------------------------------------------------------------------
# The eight meshes
# ---------------------------------------------------------------------------


def _build_blob_limbs() -> tuple:
    """Metaball creature: a body with five clearly separated protruding limbs.

    Each limb is a capsule metaball whose axis runs outward from inside the
    body, so the limbs are real protrusions rather than surface bumps.  A single
    chart cannot cover a limb and the body without severe stretch, which is
    exactly the chart-per-limb decomposition this entry is here to exercise.
    """
    balls = [
        ((0.0, 0.0, 0.0), (0.0, 0.10, 0.0), 1.00, 1.00),  # body
        ((0.0, 0.28, 0.10), (0.0, 0.95, 0.42), 0.52, 0.95),  # head / neck
        ((-0.22, 0.10, 0.02), (-1.05, 0.22, 0.10), 0.44, 0.90),  # arm L
        ((0.22, 0.10, 0.02), (1.05, 0.22, 0.10), 0.44, 0.90),  # arm R
        ((-0.20, -0.18, 0.0), (-0.58, -1.02, 0.05), 0.42, 0.90),  # leg L
        ((0.20, -0.18, 0.0), (0.58, -1.02, 0.05), 0.42, 0.90),  # leg R
    ]
    field = _blob_field(balls)
    verts, tris = _marching_tets(field, (-1.7, -1.7, -1.1), (1.7, 1.7, 1.1), 26, 0.14)
    return verts, tris, None, None


def _build_hand() -> tuple:
    """Palm plus five fingers whose charts collide in UV even when each is fine.

    The fingers sit 0.30 apart with an effective surface radius of 0.116, so the
    gap between two fingers is 0.067 -- about half a finger.  That spacing was
    chosen by measurement, not by eye: at radius 0.185 they stay six separate
    protrusions (``chi = 2``) at every resolution tried up to 56 nodes, while at
    0.185 with the fingers 0.02 closer they *fuse* into a handle and the mesh
    becomes genus 1.  A corpus mesh whose topology depends on the polygoniser's
    resolution would be worthless, so the parameters sit on the safe side of
    that boundary and the invariant check pins it.

    A per-finger chart is small and well-shaped in isolation; the difficulty is
    that the charts land on top of each other in UV, and a flip-only injectivity
    check cannot see that at all.
    """
    balls = [((-0.62, -0.30, 0.0), (0.62, -0.30, 0.0), 0.62, 1.00)]  # palm
    spacing = 0.30
    finger_len = (0.72, 0.88, 0.94, 0.86, 0.66)
    for i, length in enumerate(finger_len):
        x = (i - 2) * spacing
        balls.append(((x, 0.02, 0.0), (x * 1.12, 0.02 + length, 0.0), 0.185, 1.00))
    balls.append(((-0.70, -0.34, 0.0), (-1.05, 0.16, 0.0), 0.21, 1.00))  # thumb
    field = _blob_field(balls)
    verts, tris = _marching_tets(
        field, (-1.45, -1.15, -0.80), (1.20, 1.30, 0.80), (32, 32, 16), 0.22
    )
    return verts, tris, None, None


def _build_displaced_sphere() -> tuple:
    """Closed genus 0 with a smooth radial displacement and no sharp edge.

    ``SHARP`` seam selection at 30 degrees finds **nothing** here, which is the
    failure mode ``docs/BENCHMARKS.md`` section 4.3 records for the sphere and
    the icosphere -- but on a shape whose curvature actually varies, so a
    curvature-driven strategy has something to find.
    """
    unit, tris = _icosphere(3)
    displace = _band_limited_bumps(_fibonacci_directions(7), 7)
    radius = 1.0 + 0.24 * displace(unit)
    return unit * radius[:, None], tris, None, None


def _build_torus_knot() -> tuple:
    """A (2, 3) torus knot swept with an organic, varying cross-section.

    A tube around any closed space curve is topologically a torus: genus 1,
    ``chi = 0``, no boundary.  Two cuts are needed before a parameterisation
    exists at all, and the knot's curvature makes it non-obvious where they go.
    """
    n_ring = 132
    n_side = 22
    s = np.arange(n_ring, dtype=np.float64) * (2.0 * math.pi / n_ring)
    p, q = 2.0, 3.0
    r = 2.0 + np.cos(q * s)
    curve = np.stack([r * np.cos(p * s), r * np.sin(p * s), -np.sin(q * s)], axis=1)
    _, normal, binormal = _rotation_minimising_frame(curve)
    # An organic cross-section: elliptical, and thicker on the outer bends.
    thickness = 0.34 + 0.11 * np.cos(3.0 * s) + 0.05 * np.cos(5.0 * s)
    phi = np.arange(n_side, dtype=np.float64) * (2.0 * math.pi / n_side)
    ring = (
        np.cos(phi)[None, :, None] * normal[:, None, :]
        + 0.72 * np.sin(phi)[None, :, None] * binormal[:, None, :]
    )
    verts = (curve[:, None, :] + thickness[:, None, None] * ring).reshape(-1, 3)
    return verts, _grid_tris(n_ring, n_side, wrap_u=True, wrap_v=True), None, None


#: The ``folded_ribbon`` grid, as ``(along, across)`` vertex counts.  Exposed
#: because :func:`folded_ribbon_flat_uvs` needs it and because a harness that
#: wants to walk the ribbon by row and column should not have to guess.
RIBBON_GRID = (220, 5)


def folded_ribbon_flat_uvs() -> np.ndarray:
    """Return the *injective* layout of ``folded_ribbon``, as ``(T, 3, 2)``.

    The corpus ships ``folded_ribbon`` with a deliberately non-injective UV
    layout so an oracle can be shown to catch a real defect.  An oracle that
    only ever sees defective input is untested in the direction that matters
    most in practice, though: a certifier which returns "broken" unconditionally
    would pass every such check and then reject every good layout a user has.

    So the corpus ships the control as well.  This is the same ribbon unrolled
    flat -- the grid parameterisation itself, which is injective by construction
    because it is an affine map of a rectangle -- and any correct injectivity
    test must report it clean.

    Returns:
        ``(T, 3, 2)`` per-corner UVs matching the triangle order of the built
        ``folded_ribbon`` mesh.
    """
    n_u, n_v = RIBBON_GRID
    ii, jj = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="ij")
    uv = np.stack(
        [
            0.02 + 0.96 * ii.reshape(-1) / float(n_u - 1),
            0.35 + 0.30 * jj.reshape(-1) / float(n_v - 1),
        ],
        axis=1,
    )
    return uv[_grid_tris(n_u, n_v, wrap_u=False, wrap_v=False)]


def _build_folded_ribbon() -> tuple:
    """A ribbon shipping a locally injective, globally non-injective UV layout.

    ORGANIC.md section 0.3 is the most consequential fact in the contract:
    Suzanne's overlap is **not** a flip.  All 968 signed UV areas are positive,
    the map is locally injective everywhere, and the chart boundary still laps
    over a distant part of itself.  A flip-only check cannot see that defect,
    and a corpus that cannot reproduce it cannot prove an injectivity oracle
    works.

    So this entry ships the defect deliberately, in its minimal form.  In UV the
    ribbon is laid along an Archimedean spiral ``r = 0.30 + 0.020 * theta``.  Its
    radial pitch, ``2*pi*0.020 = 0.1257``, is smaller than the ribbon's width
    ``2 * 0.090 = 0.18``, so successive turns *must* overlap; and the half-width
    0.090 is more than three times below the smallest radius of curvature (0.30),
    so no triangle can invert.  Both properties are asserted at build time by
    :func:`_assert_ribbon_defect` -- a corpus entry that quietly stops
    reproducing its own defect would make every oracle pass for the wrong reason.

    Returns:
        ``(verts, tris, uvs)`` with ``uvs`` shaped ``(T, 3, 2)``.
    """
    n_u, n_v = RIBBON_GRID
    u = np.linspace(0.0, 1.0, n_u, dtype=np.float64)
    v = np.linspace(-1.0, 1.0, n_v, dtype=np.float64)

    # 3D: a ribbon curling through space, twisting as it goes.  Organic in the
    # sense that matters here -- smooth, no sharp edges, no symmetry to exploit.
    theta = u * (2.6 * math.pi)
    radius = 0.55 + 0.42 * u
    spine = np.stack([radius * np.cos(theta), radius * np.sin(theta), 0.85 * u - 0.42], axis=1)
    twist = u * (1.15 * math.pi)
    tangent = np.stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=1)
    up = np.zeros_like(tangent)
    up[:, 2] = 1.0
    side = np.cross(tangent, up)
    side /= np.linalg.norm(side, axis=1)[:, None]
    across = np.cos(twist)[:, None] * side + np.sin(twist)[:, None] * up
    half_width_3d = 0.20
    verts = (spine[:, None, :] + half_width_3d * v[None, :, None] * across[:, None, :]).reshape(
        -1, 3
    )
    tris = _grid_tris(n_u, n_v, wrap_u=False, wrap_v=False)

    # UV: the spiral that must overlap itself.
    r0, b, turns, half_w = 0.30, 0.020, 2.0, 0.090
    ang = u * (2.0 * math.pi * turns)
    rad = r0 + b * ang
    cx = rad * np.cos(ang)
    cy = rad * np.sin(ang)
    dx = b * np.cos(ang) - rad * np.sin(ang)
    dy = b * np.sin(ang) + rad * np.cos(ang)
    norm = np.sqrt(dx * dx + dy * dy)
    nx = -dy / norm
    ny = dx / norm
    uu = cx[:, None] + half_w * v[None, :] * nx[:, None]
    vv = cy[:, None] + half_w * v[None, :] * ny[:, None]
    uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    uv = 0.05 + 0.90 * (uv - lo) / max(float((hi - lo).max()), 1e-12)

    corner_uv = uv[tris]  # (T, 3, 2)
    _assert_ribbon_defect(corner_uv, 2.0 * math.pi * b, 2.0 * half_w)
    return verts, tris, corner_uv, None


def _assert_ribbon_defect(corner_uv: np.ndarray, pitch: float, width: float) -> None:
    """Fail the build unless ``folded_ribbon`` really carries the defect.

    A corpus entry that quietly stops reproducing the bug it exists to
    reproduce is worse than no entry at all, because every oracle then passes.
    """
    if pitch >= width:
        raise AssertionError(
            "folded_ribbon UV pitch %.4f must be below the ribbon width %.4f" % (pitch, width)
        )
    e1 = corner_uv[:, 1] - corner_uv[:, 0]
    e2 = corner_uv[:, 2] - corner_uv[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if not (signed > 0.0).all() and not (signed < 0.0).all():
        raise AssertionError(
            "folded_ribbon must be locally injective: %d of %d signed UV areas "
            "disagree with the majority sign" % (int((signed <= 0.0).sum()), signed.shape[0])
        )


def _build_pinched() -> tuple:
    """A blob deliberately damaged with zero-area faces and coincident vertices.

    Three distinct kinds of damage, because they fail differently:

    1. **Exactly zero area** -- an edge collapsed onto a single point, so every
       triangle around it has area 0.0 and no normal at all.
    2. **Near-degenerate slivers** -- vertices nudged to within 1e-9 of an
       existing edge, which survives a zero test and then divides by almost
       nothing.
    3. **Coincident but distinct vertices** -- two indices at the same position,
       which welds in some code paths and not others.

    Nothing here may crash, and no metric may come back NaN.  That is the whole
    requirement; the score is expected to be poor and that is fine.
    """
    unit, tris = _icosphere(2)
    displace = _band_limited_bumps(_fibonacci_directions(5), 5)
    verts = unit * (1.0 + 0.18 * displace(unit))[:, None]

    # 1. Collapse three edges of one triangle fan onto a single point.
    victim = int(tris[0, 0])
    for other in (int(tris[0, 1]), int(tris[0, 2])):
        verts[other] = verts[victim]

    # 2. Slide two vertices onto the line joining their neighbours.
    for tri in (7, 23):
        a, b, c = (int(x) for x in tris[tri])
        verts[c] = verts[a] + (verts[b] - verts[a]) * 0.5 + 1e-9

    # 3. A duplicate position with a distinct index.
    verts[int(tris[40, 0])] = verts[int(tris[41, 1])]
    return verts, tris, None, None


def _build_suzanne(subdivisions: int) -> tuple:
    """Return Suzanne, optionally Catmull-Clark subdivided, as triangle arrays.

    Blender's own asset, so this is the one entry that cannot be procedural.
    Subdivision is applied with a real ``SUBSURF`` modifier and evaluated, then
    triangulated through ``calc_loop_triangles`` -- the same path
    ``core/bridge.py`` uses, so the triangle arrays here match what the
    evaluator will see.

    The **quad** faces are returned alongside them, and they are what
    :func:`to_object` rebuilds from.  Suzanne is a 500-face quad mesh; handing
    Blender the 968-triangle soup instead changes the edge count from 1005 to
    1473 and with it every seam, chart and packing decision, so the two are not
    the same benchmark subject even though they are the same surface.
    """
    import bpy  # noqa: PLC0415 -- optional; the numpy corpus must import cleanly

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.object
    if subdivisions > 0:
        mod = obj.modifiers.new(name="subsurf", type="SUBSURF")
        mod.levels = int(subdivisions)
        mod.render_levels = int(subdivisions)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    mesh = obj.data
    mesh.calc_loop_triangles()
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    faces = [tuple(int(v) for v in poly.vertices) for poly in mesh.polygons]
    return (
        verts.reshape(-1, 3).astype(np.float64),
        tris.reshape(-1, 3).astype(np.int64),
        None,
        faces,
    )


_BUILDERS = {
    "suzanne": lambda: _build_suzanne(0),
    "suzanne_sub2": lambda: _build_suzanne(2),
    "blob_limbs": _build_blob_limbs,
    "displaced_sphere": _build_displaced_sphere,
    "torus_knot": _build_torus_knot,
    "hand": _build_hand,
    "folded_ribbon": _build_folded_ribbon,
    "pinched": _build_pinched,
}

#: Expected invariants, measured once and pinned here.  ``build`` compares and
#: raises on any difference, so a change in numpy, in Blender or in this file
#: that moves the corpus is a loud failure rather than a silent shift in every
#: downstream benchmark cell.  Regenerate by running this module and copying the
#: printed table -- and then say in the commit message what moved and why.
EXPECTED: dict[str, Invariants] = {
    "suzanne": Invariants(507, 968, 3, (0, 1, 1), (26, 8, 8), 0),
    "suzanne_sub2": Invariants(7958, 15744, 3, (0, 1, 1), (104, 32, 32), 0),
    "blob_limbs": Invariants(3806, 7608, 1, (2,), (0,), 0),
    "displaced_sphere": Invariants(642, 1280, 1, (2,), (0,), 0),
    "torus_knot": Invariants(2904, 5808, 1, (0,), (0,), 0),
    "hand": Invariants(4642, 9280, 1, (2,), (0,), 0),
    "folded_ribbon": Invariants(1100, 1752, 1, (1,), (446,), 0),
    "pinched": Invariants(162, 320, 1, (2,), (0,), 0),
}


def build(name: str, *, strict: bool = True) -> CorpusMesh:
    """Build one corpus mesh and verify its invariants.

    Args:
        name: One of :data:`NAMES`.
        strict: When true (the default) a mismatch against :data:`EXPECTED`
            raises.  Pass ``False`` only when deliberately regenerating the
            expected table.

    Returns:
        The built :class:`CorpusMesh`.

    Raises:
        KeyError: If ``name`` is not in the corpus.
        RuntimeError: If the measured invariants differ from the expected ones.
    """
    if name not in _BUILDERS:
        raise KeyError("unknown corpus mesh %r; known: %s" % (name, ", ".join(NAMES)))
    verts, tris, uvs, faces = _BUILDERS[name]()
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int32).reshape(-1, 3)
    inv = invariants_of(verts, tris)
    expected = EXPECTED.get(name)
    if strict and expected is not None and inv != expected:
        raise RuntimeError(
            "corpus mesh %r changed:\n  expected %s\n  measured %s"
            % (name, expected.describe(), inv.describe())
        )
    return CorpusMesh(
        name=name,
        why_hard=WHY_HARD.get(name, ""),
        verts=verts,
        tris=tris,
        uvs=None if uvs is None else np.ascontiguousarray(uvs, dtype=np.float64),
        invariants=inv,
        source="blender" if name in BLENDER_NAMES else "numpy",
        faces=faces,
    )


def build_all(*, names=None, strict: bool = True, skip_blender: bool | None = None):
    """Build several corpus meshes, skipping the Blender-only ones if needed.

    Args:
        names: Which meshes to build; defaults to :data:`NAMES`.
        strict: Passed through to :func:`build`.
        skip_blender: Force-skip the ``bpy``-dependent entries.  Defaults to
            auto-detection, so the same call works inside and outside Blender.

    Returns:
        A list of :class:`CorpusMesh`, in the requested order.
    """
    if skip_blender is None:
        skip_blender = not _has_bpy()
    wanted = tuple(names) if names is not None else NAMES
    out = []
    for name in wanted:
        if skip_blender and name in BLENDER_NAMES:
            continue
        out.append(build(name, strict=strict))
    return out


def _has_bpy() -> bool:
    """Return whether :mod:`bpy` can be imported (i.e. we are inside Blender)."""
    try:
        import bpy  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def to_sample(mesh: CorpusMesh):
    """Return a bpy-free ``UVMeshSample`` for ``mesh``.

    Per-corner UVs become one loop per triangle corner, matching what
    ``make_sample`` builds by default, so a mesh that ships its own layout
    (``folded_ribbon``) arrives with that layout intact and every other mesh
    arrives with zeros.
    """
    from unwrap_variations.core.meshdata import make_sample  # noqa: PLC0415

    uvs = None if mesh.uvs is None else mesh.uvs.reshape(-1, 2)
    return make_sample(mesh.verts, mesh.tris, uvs=uvs, name=mesh.name)


def to_object(mesh: CorpusMesh, *, name: str | None = None):
    """Create a Blender object for ``mesh`` in the current scene.

    The mesh is built with :meth:`bpy.types.Mesh.from_pydata` from the corpus's
    own arrays rather than by re-running an operator, so what Blender unwraps is
    byte-for-byte what the invariants were measured on.  When the entry carries
    original polygons (Suzanne does) those are used instead of the triangle
    soup, because a quad mesh and its triangulation are different benchmark
    subjects -- different edge counts, different charts, different packing.  A
    UV layer always exists afterwards, carrying the corpus layout when there is
    one.

    Args:
        mesh: The corpus mesh to instantiate.
        name: Optional object name; defaults to the corpus name.

    Returns:
        The new Blender object.
    """
    import bpy  # noqa: PLC0415

    obj_name = name or mesh.name
    data = bpy.data.meshes.new(obj_name)
    faces = mesh.faces if mesh.faces else mesh.tris.tolist()
    data.from_pydata(mesh.verts.tolist(), [], faces)
    data.update()
    obj = bpy.data.objects.new(obj_name, data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    layer = data.uv_layers.new(name="UVMap")
    if mesh.uvs is not None:
        data.calc_loop_triangles()
        flat = np.zeros((len(data.loops), 2), dtype=np.float32)
        tri_loops = np.empty(len(data.loop_triangles) * 3, dtype=np.int32)
        data.loop_triangles.foreach_get("loops", tri_loops)
        tri_verts = np.empty(len(data.loop_triangles) * 3, dtype=np.int32)
        data.loop_triangles.foreach_get("vertices", tri_verts)
        lookup = {}
        for t in range(mesh.tris.shape[0]):
            for c in range(3):
                lookup[(t, int(mesh.tris[t, c]))] = mesh.uvs[t, c]
        tri_loops = tri_loops.reshape(-1, 3)
        tri_verts = tri_verts.reshape(-1, 3)
        for t in range(tri_loops.shape[0]):
            for c in range(3):
                got = lookup.get((t, int(tri_verts[t, c])))
                if got is not None:
                    flat[tri_loops[t, c]] = got
        _uv_seq(layer).foreach_set("vector", flat.reshape(-1))
    return obj


def _uv_seq(layer):
    """Return the fast ``layer.uv`` sequence, falling back to ``layer.data``."""
    seq = getattr(layer, "uv", None)
    if seq is not None and len(seq):
        return seq
    return layer.data


# ---------------------------------------------------------------------------
# Exact UV overlap, shared by the visual report and anything else that needs it
# ---------------------------------------------------------------------------


def tri_signed_area_uv(tri) -> float:
    """Return the signed UV area of one triangle; negative means clockwise."""
    t = np.asarray(tri, dtype=np.float64)
    return 0.5 * float(
        (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])
    )


def tri_clip_area_uv(a, b) -> float:
    """Return the exact area of the intersection of two UV triangles.

    Sutherland-Hodgman, with two guards that this clipper needs and that are
    easy to leave out.  Both were found by measurement, and both fail silently.

    1. **Orient both triangles counter-clockwise first.**  Without it, with the
       inside test written for a counter-clockwise clip polygon, every
       counter-clockwise pair is clipped away to nothing.  A scratch diagnostic
       with that bug reported Suzanne's overlap as **exactly zero** -- a false
       clean bill of health on the most consequential measurement in
       ``docs/ORGANIC.md``.
    2. **Reject a zero-area triangle up front.**  A collapsed clip triangle has
       zero-length edges, and a zero-length edge's half-plane test is vacuously
       true, so the routine clips *nothing* and returns the subject triangle's
       whole area for a pair that does not overlap at all.  The same bug existed
       in ``tests/reference.py``.  It bites hardest on the ``pinched`` corpus
       mesh, which exists precisely to carry exactly zero-area faces.

    Measuring true intersection **area** rather than running a boolean
    separating-axis test is what makes the result trustworthy: triangles that
    merely share an edge intersect in a set of measure zero and correctly yield
    ``0.0``, so no adjacency special-casing is needed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if tri_signed_area_uv(a) == 0.0 or tri_signed_area_uv(b) == 0.0:
        return 0.0
    a = a if tri_signed_area_uv(a) > 0.0 else a[[0, 2, 1]]
    b = b if tri_signed_area_uv(b) > 0.0 else b[[0, 2, 1]]
    poly = [tuple(p) for p in a]
    for i in range(3):
        p0, p1 = b[i], b[(i + 1) % 3]
        ex, ey = p1[0] - p0[0], p1[1] - p0[1]
        new = []
        n = len(poly)
        if n == 0:
            return 0.0
        for j in range(n):
            c, d = poly[j], poly[(j + 1) % n]
            sc = ex * (c[1] - p0[1]) - ey * (c[0] - p0[0])
            sd = ex * (d[1] - p0[1]) - ey * (d[0] - p0[0])
            if sc >= 0.0:
                new.append(c)
            if (sc >= 0.0) != (sd >= 0.0):
                t = sc / (sc - sd)
                new.append((c[0] + t * (d[0] - c[0]), c[1] + t * (d[1] - c[1])))
        poly = new
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        total += x0 * y1 - x1 * y0
    return abs(total) * 0.5


def overlapping_triangle_mask(tri_uv: np.ndarray, *, cells: int = 128) -> np.ndarray:
    """Return a ``(T,)`` bool mask of triangles that really overlap another.

    A uniform grid over triangle bounding boxes, then :func:`tri_clip_area_uv`
    on each surviving pair.  Exact rather than approximate on purpose: a
    separating-axis boolean reported 8773 false positives on a flat 2048-triangle
    grid whose true answer is zero, and a picture full of false red is worse
    than no picture at all.

    Args:
        tri_uv: ``(T, 3, 2)`` per-corner UVs.
        cells: Grid resolution per axis.

    Returns:
        ``(T,)`` bool, true where the triangle overlaps at least one other.
    """
    n = tri_uv.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    lo = tri_uv.min(axis=1)
    hi = tri_uv.max(axis=1)
    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    if not finite.any():
        return mask
    gmin = lo[finite].min(axis=0)
    span = np.maximum(hi[finite].max(axis=0) - gmin, 1e-9)
    cell_lo = np.clip(((lo - gmin) / span * cells).astype(np.int64), 0, cells - 1)
    cell_hi = np.clip(((hi - gmin) / span * cells).astype(np.int64), 0, cells - 1)

    buckets: dict[int, list[int]] = {}
    for t in range(n):
        if not finite[t]:
            continue
        for cy in range(cell_lo[t, 1], cell_hi[t, 1] + 1):
            for cx in range(cell_lo[t, 0], cell_hi[t, 0] + 1):
                buckets.setdefault(cy * cells + cx, []).append(t)

    tested: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            i = members[a]
            for b in range(a + 1, len(members)):
                j = members[b]
                if (i, j) in tested:
                    continue
                tested.add((i, j))
                if hi[i, 0] < lo[j, 0] or hi[j, 0] < lo[i, 0]:
                    continue
                if hi[i, 1] < lo[j, 1] or hi[j, 1] < lo[i, 1]:
                    continue
                if tri_clip_area_uv(tri_uv[i], tri_uv[j]) > 1e-13:
                    mask[i] = True
                    mask[j] = True
    return mask


# ---------------------------------------------------------------------------
# Standalone report
# ---------------------------------------------------------------------------


def main() -> int:
    """Print the corpus table.  Returns a process exit code."""
    strict = "--regenerate" not in sys.argv
    inside = _has_bpy()
    print("ORGANIC CORPUS -- %s" % ("inside Blender" if inside else "numpy only, no bpy"))
    print("numpy %s | python %d.%d.%d" % (np.__version__, *sys.version_info[:3]))
    if not strict:
        print("REGENERATING: invariants are printed, not checked")
    print("-" * 108)
    print("%-17s %-7s %-7s %-6s %-10s %-9s %-14s %s"
          % ("mesh", "verts", "tris", "shell", "chi", "boundary", "digest", "why it is hard"))
    print("-" * 116)
    failures = []
    for name in NAMES:
        if not inside and name in BLENDER_NAMES:
            print("%-17s %s" % (name, "SKIPPED (needs Blender)"))
            continue
        try:
            mesh = build(name, strict=strict)
        except Exception as exc:  # noqa: BLE001 -- a report must survive a bad entry
            print("%-17s FAILED: %s" % (name, exc))
            failures.append(name)
            continue
        inv = mesh.invariants
        print("%-17s %-7d %-7d %-6d %-10s %-9s %-14s %s"
              % (name, inv.n_verts, inv.n_tris, inv.shells,
                 ",".join(str(c) for c in inv.chi),
                 ",".join(str(b) for b in inv.boundary_edges),
                 mesh.digest(), mesh.why_hard))
        if not strict:
            print("    Invariants(%d, %d, %d, %s, %s, %d),"
                  % (inv.n_verts, inv.n_tris, inv.shells, inv.chi, inv.boundary_edges,
                     inv.nonmanifold_edges))
    print("-" * 116)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
