"""Seam heuristics: deciding *where* a mesh should be cut before it is unwrapped.

Every seam-driven unwrapper (ABF, LSCM, minimum-stretch) is only as good as the
cut it is handed, which makes this module the place where the add-on actually
differentiates itself: the solver is Blender's, the cuts are ours.

Nothing here imports :mod:`bpy`.  The only inputs are the flat numpy arrays of a
:class:`~unwrap_variations.core.meshdata.UVMeshSample`, so the whole module runs
-- and is unit-tested -- in plain CPython.  :class:`UVMeshSample` is imported
only for type checking, which keeps this file loadable even when the surrounding
package pulls in Blender-only siblings.

Strategy summary
----------------

===============  ============================================================
``NONE``         No seams at all.
``EXISTING``     Whatever the artist already marked.
``SHARP``        The artist's seams *plus* every sharp edge.
``ANGLE``        Sharp edges only, ignoring the artist's seams.
``CURVATURE``    Scale-invariant discrete mean curvature over a threshold.
``DUAL_GRAPH``   Maximum spanning forest of the face dual graph, cut on the
                 complement, then pruned.  Few, well-placed cuts.
``SMART``        Empty -- ``uv.smart_project`` finds its own charts.
===============  ============================================================

Invariants honoured by *every* strategy
---------------------------------------

* A **boundary** edge (one incident face) or a **loose/wire** edge (no incident
  face) is never returned as a seam: there is nothing on the other side to cut.
  This is enforced for ``EXISTING`` too, so the caller can rely on the mask
  being physically meaningful without re-filtering it.
* A **non-manifold** edge (three or more incident faces) cannot lie in the
  interior of a 2-manifold chart, so the geometric strategies (``SHARP``,
  ``ANGLE``, ``CURVATURE``, ``DUAL_GRAPH``) always cut it.
* Nothing raises.  Empty meshes, zero-edge meshes, degenerate triangles,
  out-of-range indices and NaN coordinates all yield a well-formed ``(E,)``
  boolean mask.
* Results are deterministic for a given ``seed``.

The triangle-fan trap
---------------------

``sample.tri_verts`` comes from Blender's ``loop_triangles``: an n-gon is
tessellated, so a quad contributes two triangles that share a *diagonal*.  That
diagonal is a triangle edge but **not** a mesh edge -- it does not exist in
``sample.edge_verts`` and cannot be cut, because Blender has no such edge to
mark.  Every routine below therefore maps triangle edges back onto real mesh
edges and drops the ones that do not match, and the dual graph treats the
unmatched (diagonal) adjacencies as *uncuttable*, contracting the triangles of
one n-gon into a single dual node.  Skipping either half of that silently
produces seams along invisible diagonals, or charts that are split down the
middle of a quad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import is for type checkers only.
    from ..core.meshdata import UVMeshSample

__all__ = [
    "SEAM_STRATEGIES",
    "compute_seams",
    "EdgeTopology",
    "build_edge_topology",
    "edge_dihedral_deg",
    "edge_curvature",
]

#: Strategy identifiers, mirrored in :mod:`unwrap_variations.core.params`.
SEAM_STRATEGIES = (
    "NONE",
    "EXISTING",
    "SHARP",
    "ANGLE",
    "CURVATURE",
    "DUAL_GRAPH",
    "SMART",
)

#: Relative cut-off for calling a triangle normal degenerate.  Relative -- not
#: absolute -- so that scaling a mesh cannot change which triangles are ignored.
_NORMAL_REL_EPS = 1e-12


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeTopology:
    """Per-edge connectivity and geometry derived from a :class:`UVMeshSample`.

    This is the shared substrate of every strategy: computing it once costs one
    sort over ``3T`` triangle edges and is the only super-linear step in the
    module.

    Attributes:
        n_verts: Number of vertex slots (large enough to index every reference).
        n_tris: Number of triangles.
        n_edges: Number of mesh edges, matching ``sample.edge_verts``.
        face_count: ``(E,)`` int64 count of triangles incident on each mesh
            edge.  ``0`` is a loose/wire edge, ``1`` a boundary edge, ``2`` a
            manifold edge and ``>= 3`` a non-manifold edge.
        face_a: ``(E,)`` int64 first incident triangle, ``-1`` if none.
        face_b: ``(E,)`` int64 second incident triangle, ``-1`` if none.  For a
            non-manifold edge these are the two lowest-indexed incident
            triangles, which keeps the result deterministic.
        cos_dihedral: ``(E,)`` float64 dot product of the two incident unit face
            normals, clamped to ``[-1, 1]``.  Forced to ``1.0`` (perfectly flat)
            wherever the edge is not manifold or either face is degenerate, so
            that no strategy invents a seam out of missing data.
        edge_len: ``(E,)`` float64 3D length of each mesh edge.
        edge_v0: ``(E,)`` int64 first endpoint, clipped into range.
        edge_v1: ``(E,)`` int64 second endpoint, clipped into range.
        normals: ``(T, 3)`` float64 unit face normals (zero where degenerate).
        areas: ``(T,)`` float64 triangle areas.
        degenerate: ``(T,)`` bool marking triangles with no usable normal.
        diag_a: ``(D,)`` int64 first triangle of an *uncuttable* adjacency.
        diag_b: ``(D,)`` int64 second triangle of an uncuttable adjacency.
            ``diag_a[i]`` and ``diag_b[i]`` share a tessellation diagonal rather
            than a real mesh edge and therefore belong to the same n-gon.
    """

    n_verts: int
    n_tris: int
    n_edges: int
    face_count: np.ndarray
    face_a: np.ndarray
    face_b: np.ndarray
    cos_dihedral: np.ndarray
    edge_len: np.ndarray
    edge_v0: np.ndarray
    edge_v1: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    degenerate: np.ndarray
    diag_a: np.ndarray
    diag_b: np.ndarray

    # -- convenience masks ---------------------------------------------------

    @property
    def boundary_mask(self) -> np.ndarray:
        """``(E,)`` bool: edges with exactly one incident face."""
        return self.face_count == 1

    @property
    def manifold_mask(self) -> np.ndarray:
        """``(E,)`` bool: edges with exactly two incident faces."""
        return self.face_count == 2

    @property
    def nonmanifold_mask(self) -> np.ndarray:
        """``(E,)`` bool: edges with three or more incident faces."""
        return self.face_count >= 3

    @property
    def cuttable_mask(self) -> np.ndarray:
        """``(E,)`` bool: edges that could ever legitimately be a seam."""
        return self.face_count >= 2


def _clip_indices(arr: np.ndarray, limit: int) -> np.ndarray:
    """Return ``arr`` as int64 clipped into ``[0, limit - 1]``.

    A malformed sample (indices out of range) must degrade rather than raise an
    ``IndexError`` halfway through a generation run, so every index used for
    fancy-indexing passes through here first.
    """
    out = np.asarray(arr, dtype=np.int64)
    if limit <= 0:
        return np.zeros_like(out)
    return np.clip(out, 0, limit - 1)


def _as_rows(arr: np.ndarray, cols: int, dtype: type) -> np.ndarray:
    """Return ``arr`` as an ``(N, cols)`` array of ``dtype``, trimming any tail.

    A ragged array is a caller bug, but this module promises never to raise, so
    a stray element is dropped rather than allowed to explode inside a reshape.
    """
    a = np.asarray(arr, dtype=dtype)
    if a.ndim == 2 and a.shape[1] == cols:
        return a
    flat = a.reshape(-1)
    return flat[: (flat.shape[0] // cols) * cols].reshape(-1, cols)


def _pair_keys(pairs: np.ndarray, base: int) -> np.ndarray:
    """Return an order-independent int64 key per undirected vertex pair.

    ``base`` is one past the largest vertex index in play; with a mesh of ten
    million vertices the key still peaks around ``1e14``, comfortably inside
    int64, so no hashing or tuple building is needed.
    """
    p = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    return p[:, 0] * np.int64(max(base, 1)) + p[:, 1]


def build_edge_topology(sample: "UVMeshSample") -> EdgeTopology:
    """Build the per-edge connectivity every seam strategy is derived from.

    The work is a single lexicographic sort: the three vertex pairs of each
    triangle are hashed to int64 keys and matched against the mesh edge keys
    with :func:`numpy.searchsorted`.  Triangle edges that fail to match are the
    tessellation diagonals of n-gons (see the module docstring) and are grouped
    separately into ``diag_a``/``diag_b`` so the dual graph can contract them.

    Args:
        sample: The mesh snapshot to analyse.

    Returns:
        A fully populated :class:`EdgeTopology`.  Safe on empty meshes.

    Notes:
        If ``edge_verts`` somehow lists the same vertex pair twice, every
        incident face is attributed to the first of the duplicates; the second
        ends up with a face count of zero and is therefore never a seam.
        Blender does not produce duplicate edges, so this only matters for
        hand-built samples.
    """
    edge_verts = _as_rows(sample.edge_verts, 2, np.int64)
    tri_verts = _as_rows(sample.tri_verts, 3, np.int64)
    verts = _as_rows(sample.verts, 3, np.float64)

    n_edges = int(edge_verts.shape[0])
    n_tris = int(tri_verts.shape[0])
    n_verts = int(verts.shape[0])

    # Vertex-index space large enough for every reference we will make, so that
    # a malformed sample cannot collide two distinct edges onto one key.
    base = max(n_verts, 1)
    if tri_verts.size:
        base = max(base, int(tri_verts.max()) + 1)
    if edge_verts.size:
        base = max(base, int(edge_verts.max()) + 1)

    tv = _clip_indices(tri_verts, max(n_verts, 1)) if n_verts else np.zeros_like(tri_verts)
    ev0 = _clip_indices(edge_verts[:, 0], max(n_verts, 1))
    ev1 = _clip_indices(edge_verts[:, 1], max(n_verts, 1))

    # -- per-triangle geometry ----------------------------------------------
    if n_tris and n_verts:
        p = verts[tv]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        finite = np.isfinite(norm) & np.isfinite(cross).all(axis=1)
        norm = np.where(finite, norm, 0.0)
        positive = norm[norm > 0.0]
        # Relative epsilon: scaling the mesh scales every norm identically, so
        # the degenerate set -- and therefore every seam -- is scale invariant.
        scale = float(positive.mean()) if positive.size else 0.0
        degenerate = ~(norm > _NORMAL_REL_EPS * scale) if scale > 0.0 else np.ones(n_tris, bool)
        safe = np.where(degenerate, 1.0, norm)[:, None]
        normals = np.where(degenerate[:, None], 0.0, cross / safe)
        areas = 0.5 * norm
    else:
        normals = np.zeros((n_tris, 3), dtype=np.float64)
        areas = np.zeros(n_tris, dtype=np.float64)
        degenerate = np.ones(n_tris, dtype=bool)

    if n_edges and n_verts:
        d = verts[ev1] - verts[ev0]
        edge_len = np.linalg.norm(d, axis=1)
        edge_len = np.where(np.isfinite(edge_len), edge_len, 0.0)
    else:
        edge_len = np.zeros(n_edges, dtype=np.float64)

    face_count = np.zeros(n_edges, dtype=np.int64)
    face_a = np.full(n_edges, -1, dtype=np.int64)
    face_b = np.full(n_edges, -1, dtype=np.int64)
    cos_dihedral = np.ones(n_edges, dtype=np.float64)
    diag_a = np.zeros(0, dtype=np.int64)
    diag_b = np.zeros(0, dtype=np.int64)

    if n_tris == 0:
        return EdgeTopology(
            n_verts=max(n_verts, 1),
            n_tris=n_tris,
            n_edges=n_edges,
            face_count=face_count,
            face_a=face_a,
            face_b=face_b,
            cos_dihedral=cos_dihedral,
            edge_len=edge_len,
            edge_v0=ev0,
            edge_v1=ev1,
            normals=normals,
            areas=areas,
            degenerate=degenerate,
            diag_a=diag_a,
            diag_b=diag_b,
        )

    # -- triangle edge -> mesh edge ------------------------------------------
    tri_pairs = np.concatenate([tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], axis=0)
    tri_keys = _pair_keys(tri_pairs, base)
    owner = np.tile(np.arange(n_tris, dtype=np.int64), 3)

    if n_edges:
        edge_keys = _pair_keys(np.stack([ev0, ev1], axis=1), base)
        order = np.argsort(edge_keys, kind="stable")
        sorted_keys = edge_keys[order]
        pos = np.clip(np.searchsorted(sorted_keys, tri_keys), 0, n_edges - 1)
        matched = sorted_keys[pos] == tri_keys
        rec_edge = order[pos]
    else:
        matched = np.zeros(tri_keys.shape[0], dtype=bool)
        rec_edge = np.zeros(tri_keys.shape[0], dtype=np.int64)

    real_edge = rec_edge[matched]
    real_face = owner[matched]

    if n_edges and real_edge.size:
        face_count = np.bincount(real_edge, minlength=n_edges).astype(np.int64)
        # Sort incident faces per edge so face_a/face_b are deterministic even
        # for a non-manifold edge with an arbitrary number of incident faces.
        srt = np.lexsort((real_face, real_edge))
        sorted_face = real_face[srt]
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(face_count)[:-1]])
        has_one = face_count >= 1
        face_a[has_one] = sorted_face[offsets[has_one]]
        has_two = face_count >= 2
        face_b[has_two] = sorted_face[offsets[has_two] + 1]

        # Dihedral only means anything on a manifold edge; everywhere else the
        # surface is flat by definition so that no strategy invents a seam.
        man = face_count == 2
        if man.any():
            na = normals[face_a[man]]
            nb = normals[face_b[man]]
            bad = degenerate[face_a[man]] | degenerate[face_b[man]]
            dots = np.einsum("ij,ij->i", na, nb)
            dots = np.where(np.isfinite(dots), dots, 1.0)
            cos_dihedral[man] = np.where(bad, 1.0, np.clip(dots, -1.0, 1.0))

    # -- uncuttable (tessellation diagonal) adjacencies ----------------------
    unmatched = ~matched
    if unmatched.any():
        dk = tri_keys[unmatched]
        do = owner[unmatched]
        o2 = np.argsort(dk, kind="stable")
        dk = dk[o2]
        do = do[o2]
        is_new = np.empty(dk.shape[0], dtype=bool)
        is_new[0] = True
        if dk.shape[0] > 1:
            is_new[1:] = dk[1:] != dk[:-1]
        group_start = np.flatnonzero(is_new)
        group_id = np.cumsum(is_new) - 1
        anchor = do[group_start[group_id]]
        keep = anchor != do
        diag_a = do[keep]
        diag_b = anchor[keep]

    return EdgeTopology(
        n_verts=max(n_verts, 1),
        n_tris=n_tris,
        n_edges=n_edges,
        face_count=face_count,
        face_a=face_a,
        face_b=face_b,
        cos_dihedral=cos_dihedral,
        edge_len=edge_len,
        edge_v0=ev0,
        edge_v1=ev1,
        normals=normals,
        areas=areas,
        degenerate=degenerate,
        diag_a=diag_a,
        diag_b=diag_b,
    )


# ---------------------------------------------------------------------------
# Geometric measures
# ---------------------------------------------------------------------------


def edge_dihedral_deg(topo: EdgeTopology) -> np.ndarray:
    """Return ``(E,)`` float64 normal deviation in degrees per mesh edge.

    This is the angle between the two incident face *normals*, which is the
    convention Blender uses for "sharp edge by angle" (a cube's edges measure
    90, not 270 or 90's supplement).  Non-manifold, boundary, loose and
    degenerate edges report ``0.0`` -- they carry no usable dihedral.
    """
    return np.degrees(np.arccos(np.clip(topo.cos_dihedral, -1.0, 1.0)))


def edge_curvature(topo: EdgeTopology) -> np.ndarray:
    """Return ``(E,)`` float64 dimensionless discrete mean curvature per edge.

    The integrated mean curvature carried by an edge is ``0.5 * |e| * theta``
    (Cohen-Steiner and Morvan's discrete second fundamental form), and the
    barycentric area it owns is a third of its two incident triangles.  Dividing
    the first by the second gives a pointwise mean curvature in units of
    ``1/length``; multiplying back by the local length scale ``sqrt(area)``
    gives the dimensionless quantity returned here::

        k = 0.5 * |e| * theta / sqrt((A1 + A2) / 3)

    Scaling the whole mesh by ``s`` scales ``|e|`` by ``s`` and ``sqrt(area)``
    by ``s`` while leaving ``theta`` untouched, so ``k`` -- and therefore the
    seam set -- is **exactly** scale invariant.  That is the point of the
    normalisation: a strategy whose output changed when an artist worked in
    centimetres instead of metres would be worthless.

    On a well-shaped mesh ``|e| / sqrt(A_e)`` is roughly ``1.8``, so ``k`` is
    approximately the dihedral angle in radians and the default threshold of
    ``0.35`` behaves like a 20-degree crease detector.  Unlike a plain angle
    test it also responds to element size: a smoothly curved surface that is
    finely tessellated bends less per element and stays uncut, while the same
    surface tessellated coarsely reads as faceted and gets seamed.
    """
    if topo.n_edges == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(topo.n_edges, dtype=np.float64)
    man = topo.face_count == 2
    if not man.any():
        return out
    theta = np.arccos(np.clip(topo.cos_dihedral[man], -1.0, 1.0))
    area = (topo.areas[topo.face_a[man]] + topo.areas[topo.face_b[man]]) / 3.0
    scale = np.sqrt(np.where(area > 0.0, area, 1.0))
    k = 0.5 * topo.edge_len[man] * theta / scale
    out[man] = np.where(area > 0.0, k, 0.0)
    return np.where(np.isfinite(out), out, 0.0)


# ---------------------------------------------------------------------------
# Union-find (the one place a Python loop is unavoidable)
# ---------------------------------------------------------------------------


def _union_all(parent: list[int], size: list[int], left: list[int], right: list[int]) -> None:
    """Union every ``left[i]`` with ``right[i]`` in place.

    Plain list-backed union-find with path halving and union by size.  Lists
    beat numpy here because the access pattern is scalar and data-dependent;
    the caller feeds it ``list`` objects produced by ``ndarray.tolist()``, which
    is an order of magnitude faster to iterate than the array itself.
    """
    for i in range(len(left)):
        x = left[i]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = right[i]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x == y:
            continue
        if size[x] < size[y]:
            x, y = y, x
        parent[y] = x
        size[x] += size[y]


def _kruskal_reject(
    parent: list[int], size: list[int], left: list[int], right: list[int]
) -> np.ndarray:
    """Run Kruskal over pre-sorted candidate edges, returning the rejects.

    Args:
        parent: Union-find parent list, already carrying any mandatory merges.
        size: Union-find subtree sizes.
        left: Source node per candidate edge, in descending preference order.
        right: Destination node per candidate edge.

    Returns:
        ``(n,)`` bool, ``True`` where the edge was *rejected* because its two
        endpoints were already connected.  Those are exactly the edges outside
        the spanning forest, i.e. the cuts.
    """
    n = len(left)
    rejected = np.zeros(n, dtype=bool)
    if n == 0:
        return rejected
    flags = bytearray(n)
    for i in range(n):
        x = left[i]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = right[i]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x == y:
            flags[i] = 1
            continue
        if size[x] < size[y]:
            x, y = y, x
        parent[y] = x
        size[x] += size[y]
    rejected[:] = np.frombuffer(bytes(flags), dtype=np.uint8).astype(bool)
    return rejected


def _peel_dangling(
    seam: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    prunable: np.ndarray,
    protected_vert: np.ndarray,
    n_verts: int,
) -> np.ndarray:
    """Iteratively strip dangling seam edges until nothing more can go.

    A seam edge is dangling when one of its endpoints carries exactly one seam
    edge and is not itself on an opening of the surface.  Removing it can expose
    a new leaf, so the process repeats to a fixpoint.

    The naive implementation -- recompute every vertex degree and strip all
    current leaves, repeat -- is beautifully vectorised but takes one pass per
    two edges of the longest chain, and the cut graph of a large flat region is
    a *random spanning tree* whose chains run into the thousands.  Instead this
    peels with an explicit stack over a CSR vertex-to-edge adjacency, which
    touches every edge a bounded number of times: O(E) total, one pass, and
    deterministic given a deterministic input order.

    Args:
        seam: ``(E,)`` bool seam mask; not modified in place.
        v0: ``(E,)`` int64 first endpoint of each edge.
        v1: ``(E,)`` int64 second endpoint of each edge.
        prunable: ``(E,)`` bool, edges that are allowed to be removed.  Edges
            outside this set still count towards vertex degree, they simply
            cannot themselves be stripped.
        protected_vert: ``(V,)`` bool, vertices where a cut may legitimately
            terminate.
        n_verts: Number of vertex slots.

    Returns:
        A new ``(E,)`` bool seam mask.
    """
    live = np.flatnonzero(seam)
    if live.size == 0:
        return seam.copy()

    ends = np.concatenate([v0[live], v1[live]])
    eids = np.concatenate([live, live])
    order = np.argsort(ends, kind="stable")
    ends = ends[order]
    eids = eids[order]
    counts = np.bincount(ends, minlength=n_verts)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])

    degree = counts.tolist()
    adj_start = starts.tolist()
    adj = eids.tolist()
    alive = seam.tolist()
    can_cut = prunable.tolist()
    protected = protected_vert.tolist()
    end0 = v0.tolist()
    end1 = v1.tolist()

    stack = np.flatnonzero((counts == 1) & ~protected_vert).tolist()
    while stack:
        v = stack.pop()
        if degree[v] != 1 or protected[v]:
            continue
        edge = -1
        for k in range(adj_start[v], adj_start[v + 1]):
            candidate = adj[k]
            if alive[candidate]:
                edge = candidate
                break
        if edge < 0:
            degree[v] = 0
            continue
        if not can_cut[edge]:
            continue
        alive[edge] = False
        degree[v] -= 1
        other = end1[edge] if end0[edge] == v else end0[edge]
        degree[other] -= 1
        if degree[other] == 1 and not protected[other]:
            stack.append(other)
    return np.asarray(alive, dtype=bool)


def _roots(parent: list[int]) -> np.ndarray:
    """Return ``(N,)`` int64 fully-resolved component root per node.

    Path halving leaves shallow but non-flat trees; a handful of vectorised
    pointer-doubling passes flattens them without another Python loop.
    """
    p = np.asarray(parent, dtype=np.int64)
    for _ in range(64):
        q = p[p]
        if np.array_equal(q, p):
            break
        p = q
    return p


# ---------------------------------------------------------------------------
# The dual-graph strategy
# ---------------------------------------------------------------------------


def _dual_graph_seams(topo: EdgeTopology, *, curvature_threshold: float, seed: int) -> np.ndarray:
    """Cut a mesh along the complement of a maximum spanning forest of its dual.

    Algorithm
    ---------

    1. **Build the dual.**  Nodes are triangles; an arc joins the two triangles
       sharing a mesh edge.  Triangles that only share a tessellation diagonal
       are *merged up front*, which contracts each n-gon into a single dual
       node -- Blender cannot cut a quad down its diagonal, so pretending the
       arc is optional would produce impossible seams.
    2. **Weight by coplanarity.**  ``c = clamp(dot(n1, n2), -1, 1)``, which is
       ``1 - w`` for the sharpness ``w = 1 - clamp(dot(n1, n2))`` named in the
       architecture document.  A maximum spanning forest over ``c`` keeps the
       flattest adjacencies intact and therefore pushes the complement -- the
       cut -- onto the creases.  (Maximising ``w`` instead would keep the sharp
       adjacencies and cut across flat regions, the exact opposite of the
       stated intent; see the note in the module report.)
    3. **Kruskal.**  Arcs are sorted by coplanarity descending, ties broken by a
       seeded permutation so that different seeds yield genuinely different but
       equally valid cuts -- on a cube every arc ties, so each seed produces a
       different one of the eleven cube unfoldings.  Every rejected arc is a
       seam.  For a closed genus-0 component this yields ``V - 1`` cut edges
       forming a spanning tree of the component's vertices, which is precisely
       the classic unfolding: a cube comes out at ``12 - 5 = 7`` cut edges, the
       cross net.
    4. **Prune dangling ends.**  Kruskal's complement contains long dead-end
       chains wandering across flat regions -- confetti that costs seam length
       and buys nothing.  Seam edges whose endpoint has seam-degree one and is
       not on an opening of the surface are stripped, iteratively, until nothing
       changes.  What survives is the *2-core* of the cut graph plus the paths
       that terminate on a real boundary: clean, closed-ish cuts.  A flat grid
       loses its cut entirely (a disk needs no seam); a tube collapses to the
       single line joining its two boundary loops; a torus collapses to the two
       loops that actually reduce its genus.

    Two guards keep the prune from doing harm:

    * A dangling edge sharper than ``curvature_threshold`` (measured as
      ``1 - dot(n1, n2)``) is kept, because a cut that follows a real crease
      relieves distortion even when it dead-ends.  This is what makes
      ``curvature_threshold`` a meaningful knob for this strategy: low values
      protect nearly everything (more, longer seams), high values prune
      aggressively (fewer, shorter seams).  It is also what keeps a cube at its
      7 cross-net edges: every one of them is a 90-degree crease.
    * The cut graph of a **closed** component is a spanning tree of its
      vertices, and a tree is nothing *but* dangling ends -- peeling one to a
      fixpoint erases it completely and leaves a closed surface with no valid
      disk parameterisation at all.  So if pruning empties a closed component,
      that component's original cut is restored wholesale.  A smooth closed
      blob therefore keeps its full spanning cut, while a cube keeps its 7
      edges no matter how aggressive the threshold is.

    Args:
        topo: Prebuilt edge topology.
        curvature_threshold: Sharpness above which a dangling cut is protected.
        seed: Tie-break seed; different seeds give different valid cuts.

    Returns:
        ``(E,)`` bool seam mask.
    """
    n_edges = topo.n_edges
    seam = np.zeros(n_edges, dtype=bool)
    if n_edges == 0 or topo.n_tris == 0:
        return seam

    forced = topo.face_count >= 3
    seam |= forced

    n_tris = topo.n_tris
    parent = list(range(n_tris))
    size = [1] * n_tris

    # (1) contract the uncuttable n-gon diagonals.
    if topo.diag_a.size:
        _union_all(parent, size, topo.diag_a.tolist(), topo.diag_b.tolist())

    # (2, 3) maximum spanning forest over coplanarity.
    manifold = np.flatnonzero(topo.face_count == 2)
    if manifold.size:
        coplanarity = topo.cos_dihedral[manifold]
        rng = np.random.default_rng(seed)
        tie = rng.permutation(manifold.size)
        order = np.lexsort((tie, -coplanarity))
        ordered = manifold[order]
        rejected = _kruskal_reject(
            parent,
            size,
            topo.face_a[ordered].tolist(),
            topo.face_b[ordered].tolist(),
        )
        seam[ordered[rejected]] = True

    # (4) prune dangling ends.
    thr = float(curvature_threshold)
    if not math.isfinite(thr):
        thr = 0.35
    sharp = (1.0 - topo.cos_dihedral) >= max(thr, 0.0)
    prunable = seam & ~sharp & ~forced
    if not prunable.any():
        return seam

    # A vertex touching a boundary or non-manifold edge already sits on an
    # opening, so a cut that terminates there is doing useful work.
    protected_vert = np.zeros(topo.n_verts, dtype=bool)
    opener = (topo.face_count == 1) | forced
    if opener.any():
        protected_vert[topo.edge_v0[opener]] = True
        protected_vert[topo.edge_v1[opener]] = True

    pruned = _peel_dangling(
        seam, topo.edge_v0, topo.edge_v1, prunable, protected_vert, topo.n_verts
    )

    # Restore any closed component the peel emptied: with no boundary to
    # terminate on, its whole cut is dangling ends, and an uncut closed surface
    # has no disk parameterisation at all.
    has_face = topo.face_count >= 1
    if not has_face.any():
        return pruned
    roots = _roots(parent)
    comp = np.zeros(n_edges, dtype=np.int64)
    comp[has_face] = roots[topo.face_a[has_face]]
    open_root = np.zeros(n_tris, dtype=bool)
    if opener.any():
        open_root[comp[opener]] = True
    closed_edges = has_face & ~open_root[comp]
    if not closed_edges.any():
        return pruned
    survives = np.zeros(n_tris, dtype=bool)
    kept = np.flatnonzero(pruned & closed_edges)
    if kept.size:
        survives[comp[kept]] = True
    pruned |= seam & closed_edges & ~survives[comp]
    return pruned


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_existing(
    sample: "UVMeshSample", existing: np.ndarray | None, n_edges: int
) -> np.ndarray:
    """Return the caller's seam mask, or the sample's, or an empty one."""
    if existing is not None:
        arr = np.asarray(existing).reshape(-1)
        if arr.shape[0] == n_edges:
            return arr.astype(bool)
    src = np.asarray(sample.edge_seam).reshape(-1)
    if src.shape[0] == n_edges:
        return src.astype(bool)
    return np.zeros(n_edges, dtype=bool)


def _finite(value: float, fallback: float) -> float:
    """Return ``value`` as a finite float, falling back when it is not."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _as_seed(value: object) -> int:
    """Return ``value`` as a non-negative 32-bit seed, tolerating junk."""
    try:
        return abs(int(value)) & 0xFFFFFFFF  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def compute_seams(
    sample: "UVMeshSample",
    strategy: str,
    *,
    angle_deg: float = 30.0,
    curvature_threshold: float = 0.35,
    existing: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Compute a seam mask for ``sample`` under the requested ``strategy``.

    Args:
        sample: The mesh to cut.  Only its arrays are read; nothing is mutated.
        strategy: One of :data:`SEAM_STRATEGIES`, case-insensitive.  An
            unrecognised value is treated as ``EXISTING``, the only choice that
            leaves the artist's mesh exactly as it was found -- silently
            clearing or inventing seams because of a typo would be worse.
        angle_deg: Normal-deviation threshold in **degrees** for ``ANGLE`` and
            ``SHARP``.  An edge is sharp when its dihedral strictly exceeds it.
        curvature_threshold: Dimensionless threshold for ``CURVATURE`` (see
            :func:`edge_curvature`), and the dangling-cut protection level for
            ``DUAL_GRAPH``.
        existing: Optional ``(E,)`` seam mask overriding ``sample.edge_seam``.
            Ignored when its length does not match the edge count.
        seed: Determinism seed.  Only ``DUAL_GRAPH`` consumes it, where it
            selects among equally-good cuts.

    Returns:
        ``(E,)`` bool.  Boundary and loose edges are always ``False``; the array
        is freshly allocated and owned by the caller.

    Notes:
        This function never raises.  Degenerate triangles, non-manifold edges,
        out-of-range indices, NaN positions, zero-edge meshes and empty meshes
        all produce a well-formed mask.
    """
    n_edges = int(_as_rows(sample.edge_verts, 2, np.int64).shape[0])
    if n_edges == 0:
        return np.zeros(0, dtype=bool)

    strat = str(strategy or "").strip().upper()
    if strat not in SEAM_STRATEGIES:
        strat = "EXISTING"

    # smart_project performs its own chart search inside Blender and ignores
    # seams entirely, so handing it any would only confuse the comparison
    # between candidates.  NONE is the explicit "unwrap the whole thing as one
    # sheet" control case.
    if strat in ("NONE", "SMART"):
        return np.zeros(n_edges, dtype=bool)

    topo = build_edge_topology(sample)
    cuttable = topo.cuttable_mask

    if strat == "EXISTING":
        return _resolve_existing(sample, existing, n_edges) & cuttable

    # A non-manifold edge cannot be interior to a 2-manifold chart: whatever the
    # heuristic says, it has to be cut.
    forced = topo.nonmanifold_mask
    manifold = topo.manifold_mask

    if strat in ("ANGLE", "SHARP"):
        limit = _finite(angle_deg, 30.0)
        seam = (edge_dihedral_deg(topo) > limit) & manifold
        seam |= forced
        if strat == "SHARP":
            seam |= _resolve_existing(sample, existing, n_edges) & cuttable
        return seam

    if strat == "CURVATURE":
        limit = max(_finite(curvature_threshold, 0.35), 0.0)
        seam = (edge_curvature(topo) > limit) & manifold
        seam |= forced
        return seam

    return _dual_graph_seams(
        topo,
        curvature_threshold=_finite(curvature_threshold, 0.35),
        seed=_as_seed(seed),
    )
