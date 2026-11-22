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
``CONCAVE``      Concavity-weighted shortest paths: the topological minimum
                 cut, routed down the valleys an artist would hide a seam in.
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
  ``ANGLE``, ``CURVATURE``, ``DUAL_GRAPH``, ``CONCAVE``) always cut it.
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

Measuring a cut
---------------

``CONCAVE`` needs to know whether the cut it has made turns every chart into a
disk, and there is exactly one way to ask that honestly: the Euler
characteristic of the **cut** surface, ``chi = V - E + F``, per chart, over the
triangulated submesh.  Two traps around it, both of which have real teeth:

* ``sample.n_edges`` is the *quad* mesh's edge count and excludes tessellation
  diagonals, so anything that reasons about cut length or Euler counts from it
  is quietly wrong.  Nothing here uses it.
* An Euler count taken over the *uncut* mesh cannot see a **slit** -- a cut
  whose two sides stay in the same chart -- because both sides share the same
  original vertices.  A slit sphere is a perfectly good disk and such a count
  reports it as ``chi = 2``.  :func:`cut_topology` builds the cut surface
  explicitly, welding corners only across sides that survive the cut.
"""

from __future__ import annotations

import heapq
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
    "edge_signed_dihedral_deg",
    "edge_curvature",
    "concavity_weights",
    "cut_topology",
    "CutTopology",
    "ConcaveReport",
    "concave_seams",
]

#: Strategy identifiers, mirrored in :mod:`unwrap_variations.core.params`.
SEAM_STRATEGIES = (
    "NONE",
    "EXISTING",
    "SHARP",
    "ANGLE",
    "CURVATURE",
    "DUAL_GRAPH",
    "CONCAVE",
    "SMART",
)

#: Relative cut-off for calling a triangle normal degenerate.  Relative -- not
#: absolute -- so that scaling a mesh cannot change which triangles are ignored.
_NORMAL_REL_EPS = 1e-12

#: How hard ``CONCAVE`` discounts a valley.  An edge at the mesh's own 90th
#: percentile of concavity costs ``1 - bias`` times its length; a ridge at the
#: same percentile costs ``1 + bias``.  At 0.8 that is a 9:1 preference, which
#: is enough to bend a path a long way round a ridge but not so much that a
#: pathologically long detour through a shallow crease ever wins.
_CONCAVE_BIAS = 0.8

#: Percentile of ``|signed dihedral|`` used as the per-mesh angular reference.
#: Relative, not absolute, so a smooth organic model and a faceted one both get
#: a meaningful spread instead of one of them reading as uniformly flat.
_CONCAVE_REF_PERCENTILE = 90.0

#: Numerical floor, as a fraction of ``area ** 1.5``, below which a closed
#: shell's signed volume is treated as meaningless rather than as evidence that
#: the mesh is inside out.  A sphere measures 0.094 and a cube 0.41; only a
#: degenerate or doubled shell lands near zero.
_CLOSED_VOLUME_RATIO = 1e-6

#: Hard ceiling on the extra cuts the distortion escalation may add, so that a
#: hostile threshold cannot turn a mesh into confetti.
_CONCAVE_MAX_EXTRA_CUTS = 24

#: Interior Gaussian curvature (radians, absolute, summed over a chart's
#: interior vertices) above which a chart is escalated even when no single
#: vertex is a cone.  ``2 pi`` is a full cone's worth of curvature trapped
#: inside one chart; a flat chart measures 0 and a whole sphere measures 4 pi.
_CONCAVE_CHART_CURVATURE_BUDGET = 2.0 * math.pi


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


def _finite_verts(sample: "UVMeshSample") -> np.ndarray:
    """Return ``(V, 3)`` float64 positions with every non-finite value zeroed.

    ``build_edge_topology`` guards its *results* against NaN, which is enough
    when the only arithmetic is one cross product.  The concave strategy does
    more with the coordinates -- centroids, corner angles, shortest paths -- and
    ``inf - inf`` emits a ``RuntimeWarning`` from numpy before the guard ever
    sees it, which is noise in a user's console and, under a ``-W error``
    harness, an exception out of a module that promises never to raise.
    Zeroing the coordinate makes the triangle degenerate, which is the honest
    reading of a vertex with no usable position.
    """
    verts = _as_rows(sample.verts, 3, np.float64)
    if verts.size and not np.isfinite(verts).all():
        verts = np.where(np.isfinite(verts), verts, 0.0)
    return verts


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


def edge_signed_dihedral_deg(
    sample: "UVMeshSample",
    topo: EdgeTopology | None = None,
    *,
    orient: bool = True,
) -> np.ndarray:
    """Return ``(E,)`` float64 dihedral in degrees, **signed**: ``+`` is convex.

    :func:`edge_dihedral_deg` is ``arccos`` of a dot product and therefore lies
    in ``[0, 180]`` with **no sign at all** -- measured, a cube's twelve outer
    edges and the single inner edge of an L-shaped solid both report exactly
    ``90.000``.  Convex and concave are indistinguishable through it, so
    anything that wants to tell a ridge from a valley has to compute the sign
    separately.  That is what this function is for.

    The magnitude is taken from :func:`edge_dihedral_deg` unchanged.  The sign
    comes from the symmetric test::

        convexity = dot(n_a - n_b, c_a - c_b)

    where ``n`` is a unit face normal and ``c`` the face centroid.  Reading it
    one half at a time, ``dot(n_a, c_b - c_a)`` asks "does my neighbour sit
    below my plane?"; below means the surface folds away from the outside,
    i.e. a ridge.  Adding the mirrored half makes the expression invariant
    under swapping the two faces, which matters because ``face_a``/``face_b``
    order is an implementation detail of :func:`build_edge_topology`.  Measured
    on hand-built solids: an outward-wound cube gives ``+90`` on all twelve
    edges, the same cube inverted gives ``-90`` on all twelve, and an L-shaped
    extrusion gives ``+90`` on seventeen folds and ``-90`` on exactly one --
    the inner corner of the L.

    Args:
        sample: The mesh.  Triangle vertex indices and positions are read; UVs
            are not.
        topo: Optional prebuilt topology, to avoid recomputing it.
        orient: When ``True`` (the default) a **closed** shell's dihedral signs
            are flipped if its own signed volume says the winding points
            *inward*.  "Concave" is meaningless without an inside and an
            outside, and an inside-out import is common enough --
            ``tests.conftest.make_cube`` is one, measured signed volume
            ``-1.0`` -- that taking the winding at face value would silently
            invert every seam decision on it.  Open shells are left exactly as
            wound; see :func:`_shell_orientation` for why guessing there does
            more harm than good.

    Returns:
        ``(E,)`` float64 in ``[-180, 180]``.  Boundary, loose, non-manifold and
        degenerate edges report ``0.0``: they carry no usable dihedral.
    """
    if topo is None:
        topo = build_edge_topology(sample)
    out = np.zeros(topo.n_edges, dtype=np.float64)
    if topo.n_edges == 0 or topo.n_tris == 0:
        return out

    verts = _finite_verts(sample)
    if verts.shape[0] == 0:
        return out
    tv = _clip_indices(_as_rows(sample.tri_verts, 3, np.int64), verts.shape[0])
    if tv.shape[0] != topo.n_tris:
        return out

    man = topo.face_count == 2
    if not man.any():
        return out

    centroids = verts[tv].mean(axis=1)
    fa = topo.face_a[man]
    fb = topo.face_b[man]
    convexity = np.einsum(
        "ij,ij->i", topo.normals[fa] - topo.normals[fb], centroids[fa] - centroids[fb]
    )
    convexity = np.where(np.isfinite(convexity), convexity, 0.0)
    magnitude = np.degrees(np.arccos(np.clip(topo.cos_dihedral[man], -1.0, 1.0)))
    # A tie (exactly coplanar centroids) is called convex, which is the
    # conservative choice: it declines to hand out a concavity discount.
    out[man] = np.where(convexity < 0.0, -magnitude, magnitude)

    if orient:
        out = out * _shell_orientation(verts, tv, topo)
    return out


def _shell_orientation(
    verts: np.ndarray, tv: np.ndarray, topo: EdgeTopology
) -> np.ndarray:
    """Return ``(E,)`` float64 of ``+1``/``-1``: does this edge's shell face out?

    The signed volume of a **closed** shell is positive when its faces are
    wound counter-clockwise seen from outside and negative when the mesh is
    inside out.  That is a theorem, and it is the only case this function acts
    on.

    An open shell is deliberately left alone even though the same sum can be
    computed for it, because for an open shell the sum is not a volume and does
    not mean what it looks like it means.  Measured: a strip folded into a
    shallow V -- a shape with no inside at all -- reports a volume-to-area
    ratio of 0.050 against a sphere's 0.094, close enough that any threshold
    which catches inside-out spheres also inverts the sign of every edge on the
    strip, turning its one genuine crease into a ridge.  "Concave" is undefined
    without an inside; where there is no inside, the winding's answer stands.

    The sum is taken about each shell's own area-weighted centroid.  For a
    closed shell that changes nothing (the volume is translation invariant
    anyway) and it keeps the intermediate magnitudes small for a mesh sitting a
    long way from the origin.
    """
    n_edges = topo.n_edges
    ones = np.ones(n_edges, dtype=np.float64)
    if topo.n_tris == 0:
        return ones

    labels, n_shells = _face_components(topo)
    p = verts[tv]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.where(np.isfinite(areas), areas, 0.0)
    tri_centroid = p.mean(axis=1)

    weight = np.bincount(labels, weights=areas, minlength=n_shells)
    shell_centre = np.stack(
        [
            np.bincount(labels, weights=areas * tri_centroid[:, k], minlength=n_shells)
            / np.where(weight > 0.0, weight, 1.0)
            for k in range(3)
        ],
        axis=1,
    )
    rel = p - shell_centre[labels][:, None, :]
    vol6 = np.einsum("ij,ij->i", rel[:, 0], np.cross(rel[:, 1], rel[:, 2]))
    vol6 = np.where(np.isfinite(vol6), vol6, 0.0)
    volume = np.bincount(labels, weights=vol6, minlength=n_shells) / 6.0

    floor = _CLOSED_VOLUME_RATIO * np.power(np.maximum(weight, 0.0), 1.5)
    shell_sign = np.where(~_open_shells(topo, labels, n_shells) & (volume < -floor), -1.0, 1.0)

    has_face = topo.face_count >= 1
    out = ones.copy()
    out[has_face] = shell_sign[labels[topo.face_a[has_face]]]
    return out


def _open_shells(topo: EdgeTopology, labels: np.ndarray, n_shells: int) -> np.ndarray:
    """Return ``(n_shells,)`` bool: which shells are **not** closed 2-manifolds.

    A closed 2-manifold shell pairs every one of its triangle sides with exactly
    one other, so ``3F == 2E``.  Counting that directly is what makes the test
    exact; the obvious shortcut -- "does any edge have a single incident face?"
    -- is not equivalent and gets a real case wrong.

    :func:`_face_components` joins triangles across ``face_count == 2`` edges
    only, so a shell can be bounded **entirely** by non-manifold edges and then
    owns no ``face_count == 1`` edge at all.  Measured on a tube with two fins
    welded along every rim edge: ``face_count`` is 2 or 3 everywhere, the rim
    test finds nothing, the shell is declared closed, and an inside-out tube has
    the sign of every one of its edges flipped on a number that is not a volume
    -- the exact mistake :func:`_shell_orientation`'s docstring says it exists
    to avoid.

    Args:
        topo: Prebuilt edge topology.
        labels: ``(T,)`` shell id per triangle, from :func:`_face_components`.
        n_shells: Shell count.

    Returns:
        ``(n_shells,)`` bool, ``True`` where the shell is open or non-manifold.
    """
    if n_shells == 0:
        return np.zeros(0, dtype=bool)
    faces = np.bincount(labels, minlength=n_shells).astype(np.int64)
    paired = np.zeros(n_shells, dtype=np.int64)
    man = topo.face_count == 2
    if man.any():
        paired += np.bincount(labels[topo.face_a[man]], minlength=n_shells).astype(np.int64)
    if topo.diag_a.size:
        # An n-gon tessellation diagonal is a genuine adjacency with no mesh
        # edge behind it; it pairs two sides just as a manifold edge does.
        paired += np.bincount(labels[topo.diag_a], minlength=n_shells).astype(np.int64)
    is_open = 3 * faces != 2 * paired
    # Belt and braces: an edge with one incident face, or with three or more,
    # means the shell it touches is not a closed 2-manifold whatever the count
    # arithmetic says.
    for mask in (topo.face_count == 1, topo.face_count >= 3):
        if mask.any():
            is_open[labels[topo.face_a[mask]]] = True
    fan = topo.face_count >= 3
    if fan.any():
        is_open[labels[topo.face_b[fan]]] = True
    return is_open


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


def _face_components(
    topo: EdgeTopology, cut: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """Return ``(labels, count)``: which chart each triangle belongs to.

    Triangles are joined across every manifold mesh edge that is not in ``cut``
    and across every n-gon tessellation diagonal, which are uncuttable by
    definition (see the module docstring).  With ``cut=None`` the result is the
    mesh's **shells**; with a seam mask it is the **charts** that mask induces.

    Note that this sees only chart *partitioning*.  A slit -- a cut whose two
    sides stay in the same chart -- does not show up here at all, which is
    exactly why disk-ness is tested by :func:`cut_topology` on the cut surface
    and not by counting components.

    Args:
        topo: Prebuilt edge topology.
        cut: Optional ``(E,)`` bool seam mask.

    Returns:
        ``(labels, count)`` with ``labels`` shaped ``(T,)`` and dense in
        ``[0, count)``.
    """
    n_tris = topo.n_tris
    if n_tris == 0:
        return np.zeros(0, dtype=np.int64), 0
    parent = list(range(n_tris))
    size = [1] * n_tris
    if topo.diag_a.size:
        _union_all(parent, size, topo.diag_a.tolist(), topo.diag_b.tolist())
    join = topo.face_count == 2
    if cut is not None:
        join = join & ~np.asarray(cut, dtype=bool)
    if join.any():
        _union_all(parent, size, topo.face_a[join].tolist(), topo.face_b[join].tolist())
    roots = _roots(parent)
    uniq, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64).reshape(-1), int(uniq.size)


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
# The cut surface: triangle sides, charts, Euler characteristic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TriangleSides:
    """The 3T directed sides of a triangle soup, grouped into shared edges.

    ``build_edge_topology`` answers "which triangles touch this *mesh* edge".
    This answers the mirror question -- "which triangle corners are welded to
    which" -- and it is the structure disk-ness has to be measured on, because
    a **slit** (a cut whose two sides stay in one chart) is invisible to any
    count of mesh edges but doubles a side here.

    Corner ``c = 3 * t + j`` carries vertex ``tri_verts[t, j]``; side ``c`` runs
    from corner ``c`` to corner ``side_next[c]``.  Sides sharing an unordered
    vertex pair form a *group*: size 1 is a mesh boundary, size 2 an interior
    edge or an n-gon diagonal, size 3+ non-manifold.

    Attributes:
        n_tris: Triangle count.
        side_tri: ``(3T,)`` owning triangle per side.
        side_next: ``(3T,)`` the other corner of each side.
        corner_vert: ``(3T,)`` vertex index per corner.
        corner_angle: ``(3T,)`` interior angle at each corner, radians; ``0``
            on a degenerate triangle.
        group: ``(3T,)`` group id per side.
        n_groups: Number of groups.
        group_size: ``(G,)`` sides per group.
        group_first: ``(G,)`` lowest-sorted side of the group.
        group_second: ``(G,)`` second side, ``-1`` when the group has one.
        group_edge: ``(G,)`` matching mesh edge index, ``-1`` for an n-gon
            tessellation diagonal, which has no mesh edge and cannot be cut.
    """

    n_tris: int
    side_tri: np.ndarray
    side_next: np.ndarray
    corner_vert: np.ndarray
    corner_angle: np.ndarray
    group: np.ndarray
    n_groups: int
    group_size: np.ndarray
    group_first: np.ndarray
    group_second: np.ndarray
    group_edge: np.ndarray


def _empty_sides() -> _TriangleSides:
    """Return the sides structure of a mesh with no triangles."""
    z64 = np.zeros(0, dtype=np.int64)
    return _TriangleSides(
        n_tris=0,
        side_tri=z64,
        side_next=z64,
        corner_vert=z64,
        corner_angle=np.zeros(0, dtype=np.float64),
        group=z64,
        n_groups=0,
        group_size=z64,
        group_first=z64,
        group_second=z64,
        group_edge=z64,
    )


def _triangle_sides(sample: "UVMeshSample", topo: EdgeTopology) -> _TriangleSides:
    """Build the side/corner structure of ``sample``'s triangulation.

    One stable sort over ``3T`` keys, the same trick
    :func:`build_edge_topology` uses, so this costs about what building the
    topology did.  Safe on empty and malformed samples.
    """
    n_tris = topo.n_tris
    if n_tris == 0:
        return _empty_sides()

    verts = _finite_verts(sample)
    tv = _as_rows(sample.tri_verts, 3, np.int64)
    if tv.shape[0] != n_tris:
        return _empty_sides()
    limit = verts.shape[0]
    tv = _clip_indices(tv, limit) if limit else np.zeros_like(tv)

    n_sides = 3 * n_tris
    corner = np.arange(n_sides, dtype=np.int64)
    side_tri = corner // 3
    side_next = 3 * side_tri + (corner + 1) % 3
    corner_vert = tv.reshape(-1)

    # -- corner angles --------------------------------------------------------
    corner_angle = np.zeros(n_sides, dtype=np.float64)
    if limit:
        p = verts[tv]
        for j in range(3):
            a = p[:, (j + 1) % 3] - p[:, j]
            b = p[:, (j + 2) % 3] - p[:, j]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            good = np.isfinite(na) & np.isfinite(nb) & (na > 0.0) & (nb > 0.0)
            cos = np.einsum("ij,ij->i", a, b) / np.where(good, na * nb, 1.0)
            corner_angle[j::3] = np.where(
                good, np.arccos(np.clip(cos, -1.0, 1.0)), 0.0
            )

    # -- group sides by unordered vertex pair ---------------------------------
    base = max(topo.n_verts, 1)
    sv0 = corner_vert
    sv1 = corner_vert[side_next]
    keys = np.minimum(sv0, sv1) * np.int64(base) + np.maximum(sv0, sv1)

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    is_new = np.empty(n_sides, dtype=bool)
    is_new[0] = True
    if n_sides > 1:
        is_new[1:] = sorted_keys[1:] != sorted_keys[:-1]
    gid_sorted = np.cumsum(is_new) - 1
    group = np.empty(n_sides, dtype=np.int64)
    group[order] = gid_sorted
    n_groups = int(gid_sorted[-1]) + 1

    group_size = np.bincount(gid_sorted, minlength=n_groups).astype(np.int64)
    starts = np.flatnonzero(is_new)
    group_first = order[starts]
    group_second = np.where(
        group_size >= 2, order[np.minimum(starts + 1, n_sides - 1)], -1
    ).astype(np.int64)

    # -- group -> mesh edge (unmatched groups are n-gon diagonals) ------------
    group_edge = np.full(n_groups, -1, dtype=np.int64)
    if topo.n_edges:
        edge_keys = np.minimum(topo.edge_v0, topo.edge_v1) * np.int64(base) + np.maximum(
            topo.edge_v0, topo.edge_v1
        )
        eorder = np.argsort(edge_keys, kind="stable")
        sorted_ekeys = edge_keys[eorder]
        gk = keys[group_first]
        pos = np.clip(np.searchsorted(sorted_ekeys, gk), 0, topo.n_edges - 1)
        matched = sorted_ekeys[pos] == gk
        group_edge = np.where(matched, eorder[pos], -1).astype(np.int64)

    return _TriangleSides(
        n_tris=n_tris,
        side_tri=side_tri,
        side_next=side_next,
        corner_vert=corner_vert,
        corner_angle=corner_angle,
        group=group,
        n_groups=n_groups,
        group_size=group_size,
        group_first=group_first,
        group_second=group_second,
        group_edge=group_edge,
    )


@dataclass(frozen=True)
class CutTopology:
    """The topology a seam mask actually produces, measured not assumed.

    Attributes:
        tri_chart: ``(T,)`` chart id per triangle.
        n_charts: Number of charts.
        chart_tris: ``(C,)`` triangle count per chart.
        chart_chi: ``(C,)`` Euler characteristic ``V - E + F`` per chart,
            computed over the **cut** triangulated submesh: a slit's two sides
            count as two edges and its interior vertices as two vertices.  A
            chart is a disk exactly when this is ``1``.
        chart_loops: ``(C,)`` number of boundary components per chart.
        chart_curvature: ``(C,)`` sum of ``|angle defect|`` over the chart's
            **interior** cut-vertices, radians.  This is the discrete
            integrated Gaussian curvature the flattener has to absorb; it is
            ``0`` exactly when the chart is developable.
        corner_vertex: ``(3T,)`` cut-vertex id per corner (itself a corner
            index, the union-find representative).
        corner_loop: ``(3T,)`` boundary-component id per corner, ``-1`` when
            the corner is interior.
        side_open: ``(3T,)`` bool, ``True`` where the side is not welded to a
            neighbour -- a mesh boundary, a non-manifold fan or a cut.  Those
            sides are free to travel along: they are already open.
        vertex_defect: ``(3T,)`` angle defect per cut-vertex, written at the
            representative corner only and ``0`` elsewhere.
        vertex_interior: ``(3T,)`` bool, ``True`` at representatives of
            cut-vertices that touch no boundary.
    """

    tri_chart: np.ndarray
    n_charts: int
    chart_tris: np.ndarray
    chart_chi: np.ndarray
    chart_loops: np.ndarray
    chart_curvature: np.ndarray
    corner_vertex: np.ndarray
    corner_loop: np.ndarray
    side_open: np.ndarray
    vertex_defect: np.ndarray
    vertex_interior: np.ndarray

    @property
    def disks(self) -> np.ndarray:
        """``(C,)`` bool: charts that pass the exact Euler disk test."""
        return self.chart_chi == 1


def _empty_cut_topology() -> CutTopology:
    """Return the cut topology of a mesh with no triangles."""
    z64 = np.zeros(0, dtype=np.int64)
    return CutTopology(
        tri_chart=z64,
        n_charts=0,
        chart_tris=z64,
        chart_chi=z64,
        chart_loops=z64,
        chart_curvature=np.zeros(0, dtype=np.float64),
        corner_vertex=z64,
        corner_loop=z64,
        side_open=np.zeros(0, dtype=bool),
        vertex_defect=np.zeros(0, dtype=np.float64),
        vertex_interior=np.zeros(0, dtype=bool),
    )


def cut_topology(
    sample: "UVMeshSample",
    seam: np.ndarray,
    *,
    topo: EdgeTopology | None = None,
    sides: _TriangleSides | None = None,
) -> CutTopology:
    """Measure the charts a seam mask produces, and whether each is a disk.

    This is the exact Euler test, ``chi = V - E + F``, over the triangulated
    submesh **after** the cut has been applied.  The distinction matters: an
    Euler count over the uncut submesh cannot see a slit at all, so it would
    report a slit sphere -- which is a perfectly good disk -- as ``chi = 2``.
    Here the slit's two sides are two edges and its interior vertices are two
    vertices, and the sphere comes out at ``1``.

    Note also that ``sample.n_edges`` is the *quad* mesh's edge count and
    excludes tessellation diagonals; nothing in this function uses it.

    Args:
        sample: The mesh.
        seam: ``(E,)`` bool seam mask over ``sample.edge_verts``.
        topo: Optional prebuilt edge topology.
        sides: Optional prebuilt side structure.

    Returns:
        A :class:`CutTopology`.  Never raises.
    """
    if topo is None:
        topo = build_edge_topology(sample)
    if sides is None:
        sides = _triangle_sides(sample, topo)
    n_tris = sides.n_tris
    if n_tris == 0:
        return _empty_cut_topology()

    mask = np.asarray(seam, dtype=bool).reshape(-1)
    if mask.shape[0] != topo.n_edges:
        mask = np.zeros(topo.n_edges, dtype=bool)

    n_sides = 3 * n_tris
    pair = sides.group_size == 2
    cut_group = np.ones(sides.n_groups, dtype=bool)
    cut_group[pair] = False
    real = pair & (sides.group_edge >= 0)
    if real.any() and topo.n_edges:
        cut_group[real] = mask[sides.group_edge[real]]
    glued = pair & ~cut_group
    glued_ids = np.flatnonzero(glued)

    s1 = sides.group_first[glued_ids]
    s2 = sides.group_second[glued_ids]

    # -- charts ---------------------------------------------------------------
    parent = list(range(n_tris))
    size = [1] * n_tris
    if glued_ids.size:
        _union_all(parent, size, sides.side_tri[s1].tolist(), sides.side_tri[s2].tolist())
    uniq, tri_chart = np.unique(_roots(parent), return_inverse=True)
    tri_chart = tri_chart.astype(np.int64).reshape(-1)
    n_charts = int(uniq.size)

    # -- cut-vertices ---------------------------------------------------------
    cparent = list(range(n_sides))
    csize = [1] * n_sides
    if glued_ids.size:
        c1a, c1b = s1, sides.side_next[s1]
        c2a, c2b = s2, sides.side_next[s2]
        same = sides.corner_vert[c1a] == sides.corner_vert[c2a]
        left = np.concatenate([c1a, c1b])
        right = np.concatenate([np.where(same, c2a, c2b), np.where(same, c2b, c2a)])
        _union_all(cparent, csize, left.tolist(), right.tolist())
    croot = _roots(cparent)

    # -- boundary sides, loops and interior flags -----------------------------
    side_glued = glued[sides.group]
    open_sides = np.flatnonzero(~side_glued)
    on_boundary = np.zeros(n_sides, dtype=bool)
    corner_loop = np.full(n_sides, -1, dtype=np.int64)
    chart_loops = np.zeros(n_charts, dtype=np.int64)
    if open_sides.size:
        r0 = croot[open_sides]
        r1 = croot[sides.side_next[open_sides]]
        on_boundary[r0] = True
        on_boundary[r1] = True
        bparent = list(range(n_sides))
        bsize = [1] * n_sides
        _union_all(bparent, bsize, r0.tolist(), r1.tolist())
        broot = _roots(bparent)
        touched = np.unique(np.concatenate([r0, r1]))
        labels = np.unique(broot[touched])
        remap = np.full(n_sides, -1, dtype=np.int64)
        remap[labels] = np.arange(labels.size, dtype=np.int64)
        loop_of_vertex = np.full(n_sides, -1, dtype=np.int64)
        loop_of_vertex[touched] = remap[broot[touched]]
        corner_loop = loop_of_vertex[croot]
        # A boundary component never spans two charts -- cut-vertices only weld
        # across glued sides, which are intra-chart by construction -- so one
        # representative per label is enough to say which chart owns each loop.
        first = np.unique(loop_of_vertex[touched], return_index=True)[1]
        chart_loops = np.bincount(
            tri_chart[touched[first] // 3], minlength=n_charts
        ).astype(np.int64)

    # -- Euler characteristic per chart --------------------------------------
    reps = np.flatnonzero(croot == np.arange(n_sides, dtype=np.int64))
    v_count = np.bincount(tri_chart[reps // 3], minlength=n_charts).astype(np.int64)
    e_open = np.bincount(
        tri_chart[sides.side_tri[open_sides]], minlength=n_charts
    ).astype(np.int64)
    e_glued = np.bincount(
        tri_chart[sides.side_tri[s1]], minlength=n_charts
    ).astype(np.int64)
    f_count = np.bincount(tri_chart, minlength=n_charts).astype(np.int64)
    chart_chi = v_count - (e_open + e_glued) + f_count

    # -- angle defect per cut-vertex -----------------------------------------
    angle_sum = np.bincount(croot, weights=sides.corner_angle, minlength=n_sides)
    vertex_defect = np.zeros(n_sides, dtype=np.float64)
    interior = np.zeros(n_sides, dtype=bool)
    interior[reps] = ~on_boundary[reps]
    vertex_defect[reps] = np.where(
        interior[reps], 2.0 * math.pi - angle_sum[reps], math.pi - angle_sum[reps]
    )
    vertex_defect = np.where(np.isfinite(vertex_defect), vertex_defect, 0.0)
    chart_curvature = np.bincount(
        tri_chart[reps // 3],
        weights=np.abs(vertex_defect[reps]) * interior[reps],
        minlength=n_charts,
    )

    return CutTopology(
        tri_chart=tri_chart,
        n_charts=n_charts,
        chart_tris=f_count,
        chart_chi=chart_chi,
        chart_loops=chart_loops,
        chart_curvature=chart_curvature,
        corner_vertex=croot,
        corner_loop=corner_loop,
        side_open=~side_glued,
        vertex_defect=vertex_defect,
        vertex_interior=interior,
    )


# ---------------------------------------------------------------------------
# Concavity weighting and shortest paths
# ---------------------------------------------------------------------------


def concavity_weights(
    sample: "UVMeshSample",
    topo: EdgeTopology | None = None,
    *,
    bias: float = _CONCAVE_BIAS,
    signed: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``(E,)`` float64 path cost per edge, cheap in a valley.

    The cost is the edge's own 3D length scaled by ``1 + bias * t``, where
    ``t`` is the signed dihedral divided by the mesh's own 90th percentile of
    ``|signed dihedral|`` and clipped to ``[-1, 1]``.  A deep valley therefore
    costs ``1 - bias`` times its length and a sharp ridge ``1 + bias``.

    Two properties this shape was chosen for:

    * **Scale invariance.**  The angle is dimensionless and the reference is a
      percentile of angles, so scaling the mesh scales every weight by exactly
      the same factor and cannot reorder a single shortest path.
    * **Adaptivity.**  The reference is per mesh.  A hard-surface model with
      90-degree creases and a smooth organic blob whose sharpest fold is two
      degrees both get the full ``[1 - bias, 1 + bias]`` spread, so the
      strategy has an opinion on a smooth mesh -- which is precisely where
      ``SHARP`` has none (measured: zero edges at 30 degrees on a sphere, a
      torus and an icosphere).

    Boundary edges cost ``0``: they are already open, so travelling along one
    adds no seam.  Loose edges are unusable and are excluded by the caller.

    Args:
        sample: The mesh.
        topo: Optional prebuilt topology.
        bias: Strength of the concavity preference, clamped to ``[0, 0.95]``.
        signed: Optional precomputed :func:`edge_signed_dihedral_deg`.

    Returns:
        ``(E,)`` float64, finite and non-negative.
    """
    if topo is None:
        topo = build_edge_topology(sample)
    if topo.n_edges == 0:
        return np.zeros(0, dtype=np.float64)
    if signed is None:
        signed = edge_signed_dihedral_deg(sample, topo)

    strength = min(max(_finite(bias, _CONCAVE_BIAS), 0.0), 0.95)
    man = topo.face_count == 2
    ref = 0.0
    if man.any():
        ref = float(np.percentile(np.abs(signed[man]), _CONCAVE_REF_PERCENTILE))
    if not math.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    t = np.clip(np.asarray(signed, dtype=np.float64) / ref, -1.0, 1.0)
    length = np.where(np.isfinite(topo.edge_len), topo.edge_len, 0.0)
    weight = length * (1.0 + strength * t)
    weight = np.where(topo.face_count == 1, 0.0, weight)
    weight = np.where(np.isfinite(weight), weight, 0.0)
    return np.maximum(weight, 0.0)


def _build_csr(
    n_verts: int,
    v0: np.ndarray,
    v1: np.ndarray,
    weight: np.ndarray,
    usable: np.ndarray,
) -> tuple[list[int], list[int], list[int], list[float]]:
    """Return a CSR adjacency as plain lists, which iterate far faster than arrays.

    Args:
        n_verts: Vertex slot count.
        v0: ``(E,)`` first endpoint per edge.
        v1: ``(E,)`` second endpoint per edge.
        weight: ``(E,)`` non-negative cost per edge.
        usable: ``(E,)`` bool, which edges are in the graph.

    Returns:
        ``(indptr, indices, edge_ids, weights)``.
    """
    idx = np.flatnonzero(usable)
    src = np.concatenate([v0[idx], v1[idx]])
    dst = np.concatenate([v1[idx], v0[idx]])
    eid = np.concatenate([idx, idx])
    wgt = np.concatenate([weight[idx], weight[idx]])
    order = np.argsort(src, kind="stable")
    counts = np.bincount(src, minlength=n_verts)
    indptr = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    return (
        indptr.tolist(),
        dst[order].tolist(),
        eid[order].tolist(),
        wgt[order].tolist(),
    )


def _dijkstra(
    graph: tuple[list[int], list[int], list[int], list[float]],
    n_verts: int,
    sources: list[int],
    *,
    stop: list[bool] | None = None,
) -> tuple[list[float], list[int], list[int], int]:
    """Multi-source Dijkstra over a CSR graph, written against ``heapq``.

    There is no scipy in Blender (ORGANIC.md section 0.5), so this is ours.
    ``heapq`` is a C-implemented binary heap over a plain list and beats a
    hand-rolled numpy heap comfortably, because the inner loop is scalar and
    numpy scalar arithmetic is slow.

    Determinism: the heap holds ``(distance, vertex)`` tuples, so equal
    distances always settle in ascending vertex order.

    Args:
        graph: The tuple returned by :func:`_build_csr`.
        n_verts: Vertex slot count.
        sources: Zero-distance start vertices.
        stop: Optional ``(V,)`` truthy sequence; the search returns as soon as
            one of those vertices settles.  Sources must not be flagged, or the
            search stops before taking a single edge.

    Returns:
        ``(dist, prev_vertex, prev_edge, hit)``.  ``hit`` is the vertex that
        triggered ``stop``, or ``-1``.
    """
    indptr, indices, edge_ids, weights = graph
    inf = float("inf")
    dist = [inf] * n_verts
    prev_v = [-1] * n_verts
    prev_e = [-1] * n_verts
    done = bytearray(n_verts)
    heap: list[tuple[float, int]] = []
    for s in sources:
        if 0 <= s < n_verts and dist[s] != 0.0:
            dist[s] = 0.0
            heapq.heappush(heap, (0.0, s))
    hit = -1
    while heap:
        d, v = heapq.heappop(heap)
        if done[v]:
            continue
        done[v] = 1
        if stop is not None and stop[v]:
            hit = v
            break
        for k in range(indptr[v], indptr[v + 1]):
            u = indices[k]
            if done[u]:
                continue
            nd = d + weights[k]
            if nd < dist[u]:
                dist[u] = nd
                prev_v[u] = v
                prev_e[u] = edge_ids[k]
                heapq.heappush(heap, (nd, u))
    return dist, prev_v, prev_e, hit


def _trace_edges(prev_v: list[int], prev_e: list[int], target: int) -> list[int]:
    """Return the edge ids from ``target`` back to its Dijkstra source."""
    out: list[int] = []
    v = target
    guard = 0
    limit = len(prev_v) + 1
    while 0 <= v < len(prev_v) and prev_e[v] >= 0 and guard < limit:
        out.append(prev_e[v])
        v = prev_v[v]
        guard += 1
    return out


# ---------------------------------------------------------------------------
# The concave strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcaveReport:
    """What :func:`concave_seams` did, in numbers rather than adjectives.

    Attributes:
        shells: Face-connected components of the input mesh.
        charts: Charts the returned cut produces.
        disks: Charts passing the exact Euler test ``chi == 1``.
        non_disks: Charts that do not.  Anything but ``0`` is a bug.
        cut_edges: Mesh edges marked.
        cut_length: Their total 3D length.
        concave_edges: Marked manifold edges with a negative signed dihedral.
        concave_length: Their total 3D length.
        concave_fraction: ``concave_length`` over the marked manifold length.
        baseline_fraction: The same fraction over the whole mesh's cuttable
            length.  Without it ``concave_fraction`` means nothing: on a mesh
            that is half concave, hitting 50% is hitting nothing.
        joins: Shortest paths added to connect two boundary loops.
        handle_loops: Non-separating loops cut to remove a handle.
        arcs: Paths cut between two curvature extremes of a closed shell.
        extra_cuts: Cuts added by the distortion escalation.
        splits: How many of those were boundary-to-boundary splits rather than
            slits, i.e. driven by a chart's total curvature rather than by one
            cone singularity.
        fallbacks: Charts opened by the spanning-tree fallback because the
            measured loop could not make them disks.  Should be ``0``.
        rounds: Topology rounds run.
    """

    shells: int = 0
    charts: int = 0
    disks: int = 0
    non_disks: int = 0
    cut_edges: int = 0
    cut_length: float = 0.0
    concave_edges: int = 0
    concave_length: float = 0.0
    concave_fraction: float = 0.0
    baseline_fraction: float = 0.0
    joins: int = 0
    handle_loops: int = 0
    arcs: int = 0
    extra_cuts: int = 0
    splits: int = 0
    fallbacks: int = 0
    rounds: int = 0


@dataclass(frozen=True)
class _CutGraph:
    """The 1-skeleton of the cut surface, as a graph a path can be routed on.

    Routing over the mesh's *original* vertices is the obvious thing to do and
    it is wrong, in a way that only shows up once a cut already exists.  Cut a
    torus along a meridian and it becomes a cylinder with two boundary loops --
    but both loops occupy the **same original vertices**, so a shortest path
    between them measures zero, marks nothing, and the algorithm stalls
    (measured: exactly this, before this structure existed).  On the cut
    surface the two sides of a slit are two distinct nodes and the path between
    them is the lengthwise cut the cylinder actually needs.

    Nodes are cut-vertices, densely numbered.  Arcs are triangle *sides*, which
    is what lets an already-open side cost nothing: travelling along an
    existing boundary or an existing cut adds no new seam.

    Attributes:
        n_nodes: Node count.
        node_of_corner: ``(3T,)`` node id per corner.
        side_node0: ``(3T,)`` node at the start of each side.
        side_node1: ``(3T,)`` node at the end of each side.
        side_cost: ``(3T,)`` cost of cutting each side.
        side_edge: ``(3T,)`` mesh edge behind each side, ``-1`` for an n-gon
            diagonal, which cannot be cut.
        side_ok: ``(3T,)`` bool, sides usable as graph arcs.
        side_open: ``(3T,)`` bool, sides that are already open and free.
        node_chart: ``(n_nodes,)`` chart per node.
        node_loop: ``(n_nodes,)`` boundary component per node, ``-1`` inside.
        node_defect: ``(n_nodes,)`` angle defect per node.
        node_interior: ``(n_nodes,)`` bool, node touches no boundary.
    """

    n_nodes: int
    node_of_corner: np.ndarray
    side_node0: np.ndarray
    side_node1: np.ndarray
    side_cost: np.ndarray
    side_edge: np.ndarray
    side_ok: np.ndarray
    side_open: np.ndarray
    node_chart: np.ndarray
    node_loop: np.ndarray
    node_defect: np.ndarray
    node_interior: np.ndarray


class _ConcaveCutter:
    """Working state for one :func:`concave_seams` call.

    A class rather than a dozen free functions with eleven arguments each: the
    mesh, its topology, the side structure, the weights and the growing seam
    mask are threaded through every step and are genuinely one object.
    """

    __slots__ = (
        "sample",
        "topo",
        "sides",
        "signed",
        "weight",
        "seam",
        "cuttable",
        "joins",
        "handle_loops",
        "arcs",
        "extra_cuts",
        "splits",
        "fallbacks",
    )

    def __init__(
        self,
        sample: "UVMeshSample",
        topo: EdgeTopology,
        sides: _TriangleSides,
        *,
        bias: float,
    ) -> None:
        self.sample = sample
        self.topo = topo
        self.sides = sides
        self.signed = edge_signed_dihedral_deg(sample, topo)
        self.weight = concavity_weights(sample, topo, bias=bias, signed=self.signed)
        self.seam = topo.nonmanifold_mask.copy()
        self.cuttable = topo.face_count >= 2
        self.joins = 0
        self.handle_loops = 0
        self.arcs = 0
        self.extra_cuts = 0
        self.splits = 0
        self.fallbacks = 0

    # -- measuring and graph building ----------------------------------------

    def measure(self) -> CutTopology:
        """Re-measure the charts the current seam mask produces."""
        return cut_topology(self.sample, self.seam, topo=self.topo, sides=self.sides)

    def graph(self, ct: CutTopology) -> _CutGraph:
        """Build the cut surface's routing graph for the current cut."""
        sides = self.sides
        n_sides = 3 * sides.n_tris
        reps = np.flatnonzero(ct.corner_vertex == np.arange(n_sides, dtype=np.int64))
        node_of = np.full(n_sides, -1, dtype=np.int64)
        node_of[reps] = np.arange(reps.size, dtype=np.int64)
        node_of_corner = node_of[ct.corner_vertex]

        side_edge = sides.group_edge[sides.group]
        safe_edge = np.maximum(side_edge, 0)
        # An already-open side (mesh boundary, non-manifold fan, existing cut)
        # is free: it is a boundary, so running along it adds nothing.  A glued
        # side costs its mesh edge's concavity weight -- unless it has no mesh
        # edge at all, which makes it an n-gon diagonal and uncuttable.
        cost = np.where(ct.side_open, 0.0, self.weight[safe_edge])
        side_ok = ct.side_open | (side_edge >= 0)

        return _CutGraph(
            n_nodes=int(reps.size),
            node_of_corner=node_of_corner,
            side_node0=node_of_corner,
            side_node1=node_of_corner[sides.side_next],
            side_cost=cost,
            side_edge=side_edge,
            side_ok=side_ok,
            side_open=ct.side_open,
            node_chart=ct.tri_chart[reps // 3],
            node_loop=ct.corner_loop[reps],
            node_defect=ct.vertex_defect[reps],
            node_interior=ct.vertex_interior[reps],
        )

    def _csr(self, g: _CutGraph, usable: np.ndarray):
        """CSR adjacency over ``usable`` sides; arc ids are side indices."""
        return _build_csr(g.n_nodes, g.side_node0, g.side_node1, g.side_cost, usable)

    def _chart_sides(self, g: _CutGraph, chart: int) -> np.ndarray:
        """``(3T,)`` bool: usable sides belonging to ``chart``."""
        return g.side_ok & (g.node_chart[g.side_node0] == chart)

    def _mark_sides(self, g: _CutGraph, side_ids) -> int:
        """Cut the mesh edges behind ``side_ids``; return how many were new."""
        ids = np.asarray(list(side_ids), dtype=np.int64)
        if ids.size == 0:
            return 0
        edges = g.side_edge[ids]
        edges = edges[edges >= 0]
        if edges.size == 0:
            return 0
        edges = edges[self.cuttable[edges]]
        if edges.size == 0:
            return 0
        added = int((~self.seam[edges]).sum())
        self.seam[edges] = True
        # A side that has just been cut is open, and running along an open side
        # costs nothing, so the next path in this same round can reuse it for
        # free.  Without this the second and third cuts of a round pay again
        # for ground the first one already opened.
        if added:
            g.side_cost[np.isin(g.side_edge, edges)] = 0.0
        return added

    # -- the three topological cuts ------------------------------------------

    def join_loops(self, ct: CutTopology, g: _CutGraph, chart: int) -> int:
        """Connect every boundary loop of ``chart`` with shortest weighted paths.

        Prim over the loops: grow the connected set to the nearest node of a
        loop it does not yet contain.  On Suzanne's head -- genus 0, two
        boundary loops, measured ``chi = 0`` -- there is exactly one such path
        and it is the entire cut the head topologically needs (ORGANIC.md
        section 0.1).  Nothing about that is hard-coded; it falls out of the
        measurement.
        """
        usable = self._chart_sides(g, chart)
        nodes = np.flatnonzero(g.node_chart == chart)
        loops = g.node_loop[nodes]
        labels = np.unique(loops[loops >= 0])
        if labels.size < 2:
            return 0
        added = 0
        joined = set(nodes[loops == labels[0]].tolist())
        pending = [set(nodes[loops == label].tolist()) for label in labels[1:]]
        while pending:
            stop = [False] * g.n_nodes
            waiting = False
            for group in pending:
                for v in group:
                    if v not in joined:
                        stop[v] = True
                        waiting = True
            if not waiting:
                break
            graph = self._csr(g, usable)
            _, prev_v, prev_e, hit = _dijkstra(graph, g.n_nodes, sorted(joined), stop=stop)
            if hit < 0:
                break
            path = _trace_edges(prev_v, prev_e, hit)
            # A path that marks nothing still counts as progress: it means the
            # route ran entirely along sides that were already open, so the two
            # loops are joined for free.
            self._mark_sides(g, path)
            added += 1
            self.joins += 1
            for side in path:
                joined.add(int(g.side_node0[side]))
                joined.add(int(g.side_node1[side]))
            reached = next((grp for grp in pending if hit in grp), None)
            if reached is None:
                break
            pending.remove(reached)
            joined |= reached
        return added

    def cut_arc(self, g: _CutGraph, chart: int, seed: int) -> int:
        """Cut a path between two curvature extremes of a closed chart.

        The source is a maximum of ``|angle defect|`` -- the sharpest cone the
        chart owns -- and the target is the node farthest from it under the
        concavity-weighted metric, which on a smooth closed shell is the
        antipode and on a faceted one the opposite corner.  ``seed`` rotates
        the source among the sharpest few, so different candidates in a grid
        get genuinely different but equally valid cuts, exactly the way
        ``DUAL_GRAPH`` uses its seed on a cube.
        """
        usable = self._chart_sides(g, chart)
        nodes = np.flatnonzero(g.node_chart == chart)
        if nodes.size < 2 or not usable.any():
            return 0
        interior = nodes[g.node_interior[nodes]]
        ranked = interior if interior.size else nodes
        order = np.lexsort((ranked, -np.abs(g.node_defect[ranked])))
        top = ranked[order][: max(1, min(8, ranked.size))]
        source = int(top[seed % top.size])

        dist, prev_v, prev_e, _ = _dijkstra(self._csr(g, usable), g.n_nodes, [source])
        best, best_d = -1, -1.0
        for v in nodes.tolist():
            d = dist[v]
            if math.isfinite(d) and d > best_d:
                best_d, best = d, int(v)
        if best < 0 or best == source:
            return 0
        added = self._mark_sides(g, _trace_edges(prev_v, prev_e, best))
        if added:
            self.arcs += 1
        return added

    def cut_handle_loop(self, g: _CutGraph, chart: int) -> int:
        """Cut the cheapest non-contractible loop of a chart that has a handle.

        Tree-cotree (Eppstein; Erickson and Whittlesey): take a shortest-path
        tree ``T`` over the chart's cut-vertices, then a spanning tree of its
        dual built only from edges outside ``T``.  Every edge in neither tree
        closes a non-contractible loop with ``T``, and the dual tree is grown
        from the *expensive* end so the loops left over are the cheap ones.
        The single cheapest is cut and the caller re-measures ``chi``.

        A torus goes ``chi = 0, b = 0`` -> loop cut -> ``chi = 0, b = 2``, and
        :meth:`join_loops` then takes it to ``chi = 1``: a disk, measured
        rather than assumed.
        """
        usable = self._chart_sides(g, chart)
        nodes = np.flatnonzero(g.node_chart == chart)
        if nodes.size == 0 or not usable.any():
            return 0
        root = int(nodes[0])
        dist, prev_v, prev_e = _dijkstra(self._csr(g, usable), g.n_nodes, [root])[:3]

        # Tree membership is tracked per *mesh edge*, not per side: an interior
        # mesh edge owns two sides, and treating the twin of a tree side as a
        # free candidate would offer a loop that is really no loop at all.
        tree_edge = np.zeros(self.topo.n_edges, dtype=bool)
        used = [int(g.side_edge[s]) for s in prev_e if s >= 0]
        used = [e for e in used if e >= 0]
        if used:
            tree_edge[np.asarray(used, dtype=np.int64)] = True

        one_per_edge = np.zeros(g.side_ok.shape[0], dtype=bool)
        one_per_edge[self.sides.group_first] = True
        candidate = usable & one_per_edge & ~g.side_open & (g.side_edge >= 0)
        candidate &= ~tree_edge[np.maximum(g.side_edge, 0)]
        ids = np.flatnonzero(candidate)
        if ids.size == 0:
            return 0

        du = np.asarray([dist[v] for v in g.side_node0[ids].tolist()])
        dv = np.asarray([dist[v] for v in g.side_node1[ids].tolist()])
        cost = g.side_cost[ids] + du + dv
        keep = np.isfinite(cost)
        ids, cost = ids[keep], cost[keep]
        if ids.size == 0:
            return 0

        # Grow the dual tree from the expensive end so the cheap loops survive
        # as leftovers.  n-gon diagonals are contracted first: Blender cannot
        # cut a quad down its diagonal, so those triangles are one dual node.
        parent = list(range(self.topo.n_tris))
        size = [1] * self.topo.n_tris
        if self.topo.diag_a.size:
            _union_all(parent, size, self.topo.diag_a.tolist(), self.topo.diag_b.tolist())
        order = np.lexsort((ids, -cost))
        edges = g.side_edge[ids[order]]
        rejected = _kruskal_reject(
            parent,
            size,
            self.topo.face_a[edges].tolist(),
            self.topo.face_b[edges].tolist(),
        )
        leftover = ids[order][rejected]
        if leftover.size == 0:
            return 0
        left_cost = cost[order][rejected]
        pick = int(leftover[int(np.lexsort((leftover, left_cost))[0])])

        # The loop is the side plus the tree path between its endpoints, which
        # is the symmetric difference of the two root paths: the shared suffix
        # appears in both and cancels.
        a = set(_trace_edges(prev_v, prev_e, int(g.side_node0[pick])))
        b = set(_trace_edges(prev_v, prev_e, int(g.side_node1[pick])))
        added = self._mark_sides(g, (a ^ b) | {pick})
        if added:
            self.handle_loops += 1
        return added

    def open_by_spanning_tree(self, ct: CutTopology, chart: int) -> int:
        """Last-resort fallback: cut everything but a spanning tree of the dual.

        A tree of triangles glued edge to edge is always a disk, so this cannot
        fail to open a chart.  It is long and ugly, which is why it is a
        fallback and why :class:`ConcaveReport` counts it: a non-zero
        ``fallbacks`` means the measured loop above gave up on something.
        """
        inside = ct.tri_chart == chart
        allowed = np.zeros(self.topo.n_edges, dtype=bool)
        man = self.topo.face_count == 2
        if man.any():
            allowed[man] = inside[self.topo.face_a[man]] & inside[self.topo.face_b[man]]
        ids = np.flatnonzero(allowed)
        if ids.size == 0:
            return 0
        parent = list(range(self.topo.n_tris))
        size = [1] * self.topo.n_tris
        if self.topo.diag_a.size:
            _union_all(parent, size, self.topo.diag_a.tolist(), self.topo.diag_b.tolist())
        order = np.lexsort((ids, -self.weight[ids]))
        rejected = _kruskal_reject(
            parent,
            size,
            self.topo.face_a[ids[order]].tolist(),
            self.topo.face_b[ids[order]].tolist(),
        )
        cut = ids[order][rejected]
        cut = cut[self.cuttable[cut]]
        added = int((~self.seam[cut]).sum())
        self.seam[cut] = True
        if added:
            self.fallbacks += 1
        return added

    # -- the distortion escalation -------------------------------------------

    def _slit_to(self, g: _CutGraph, chart: int, target: int) -> list[int]:
        """Shortest path from ``chart``'s boundary to ``target``; sides, or []."""
        usable = self._chart_sides(g, chart)
        nodes = np.flatnonzero(g.node_chart == chart)
        rim = nodes[g.node_loop[nodes] >= 0].tolist()
        if not rim or target in rim:
            return []
        stop = [False] * g.n_nodes
        stop[target] = True
        _, prev_v, prev_e, hit = _dijkstra(self._csr(g, usable), g.n_nodes, rim, stop=stop)
        if hit < 0:
            return []
        return _trace_edges(prev_v, prev_e, hit)

    def _split_chart(self, g: _CutGraph, chart: int) -> list[int]:
        """A boundary-to-boundary arc bisecting ``chart``, as a list of sides.

        A slit relieves the curvature it passes through but leaves the chart
        one chart; only an arc running boundary to boundary divides the chart's
        remaining curvature in two.

        The arc is aimed at the interior node **farthest from the chart's own
        boundary** under the concavity-weighted metric -- the pole of
        inaccessibility, near enough -- and not at the node of highest
        curvature.  Aiming at curvature is the obvious choice and it is wrong:
        the sharpest interior vertex is usually the one just inside the
        boundary, so the arc carves off a sliver and the chart stays over
        budget.  Measured before this change, a UV sphere shed 24 slivers, hit
        the escalation cap and still had one oversized chart.

        The second leg is routed with the first leg's interior nodes removed,
        so the two legs cannot overlap and the result really does separate.
        """
        usable = self._chart_sides(g, chart)
        nodes = np.flatnonzero(g.node_chart == chart)
        rim = nodes[g.node_loop[nodes] >= 0].tolist()
        if not rim:
            return []
        dist, prev_v, prev_e, _ = _dijkstra(self._csr(g, usable), g.n_nodes, rim)
        target, best = -1, -1.0
        for v in nodes[g.node_interior[nodes]].tolist():
            d = dist[v]
            if math.isfinite(d) and d > best:
                best, target = d, int(v)
        if target < 0:
            return []
        first = _trace_edges(prev_v, prev_e, target)
        if not first:
            return []
        blocked = set()
        for side in first:
            blocked.add(int(g.side_node0[side]))
            blocked.add(int(g.side_node1[side]))
        blocked.discard(target)
        if blocked:
            bad = np.zeros(g.n_nodes, dtype=bool)
            bad[np.asarray(sorted(blocked), dtype=np.int64)] = True
            usable = usable & ~bad[g.side_node0] & ~bad[g.side_node1]
        stop = [False] * g.n_nodes
        for v in rim:
            if v not in blocked and v != target:
                stop[v] = True
        if not any(stop):
            return first
        _, prev_v, prev_e, hit = _dijkstra(
            self._csr(g, usable), g.n_nodes, [target], stop=stop
        )
        if hit < 0:
            return first
        return first + _trace_edges(prev_v, prev_e, hit)

    def escalate(self, *, cone: float, chart_budget: float, budget: int) -> int:
        """Add cuts only where a curvature measurement asks for one.

        The measurement is exact discrete Gaussian curvature -- the angle
        defect of each interior cut-vertex -- not a proxy.  Two things it can
        say, and each has its own answer:

        * **Too much curvature in one chart**: more than ``chart_budget``
          radians summed over the interior.  No amount of slitting fixes that
          -- a slit relieves only the vertices it passes -- so the answer is a
          boundary-to-boundary *split*, which halves the chart's curvature.
          Without this a sphere would stay one 4-pi chart and the escalation
          would crawl to its cap achieving nothing (measured: exactly that).
        * **A cone singularity**: one interior vertex over ``cone`` radians of
          defect.  No flattening absorbs that without stretching, and the fix
          is a *slit* from the boundary to the vertex, which moves the cone
          onto the boundary where the angle is free.  This is Sheffer and
          Hart's Seamster criterion, and it is what turns a cube's topological
          arc into something close to its cross net.

        **Splits go first, and the order is not cosmetic.**  Measured on
        Suzanne with the slits taking priority: the cone rule fired on every
        one of the 24 permitted cuts, no split ever ran, and the result was 3
        charts with ``stretch_l2`` 1.394 and an overlap of 3.9e-2, scoring
        16.6.  With splits first the same mesh comes out at 8 charts,
        ``stretch_l2`` 1.045 and overlap exactly 0.  A chart carrying more
        curvature than one disk can hold has to be divided before it is worth
        refining, and slitting to individual cones inside it is refinement.

        Progress is guaranteed: a split strictly divides a chart and a slit
        puts its target on the boundary, so the same vertex is never chosen
        twice.  The loop therefore stops on its own well before ``budget``.
        """
        added = 0
        for _ in range(max(0, budget)):
            ct = self.measure()
            g = self.graph(ct)
            interior = np.flatnonzero(g.node_interior)
            if interior.size == 0:
                break
            defect = np.abs(g.node_defect[interior])

            heavy = np.flatnonzero(ct.chart_curvature > chart_budget)
            if heavy.size:
                worst_chart = int(heavy[int(np.argmax(ct.chart_curvature[heavy]))])
                path = self._split_chart(g, worst_chart)
                kind = "split"
            else:
                over = defect > cone
                cones = interior[over]
                if cones.size == 0:
                    break
                pick = int(cones[int(np.lexsort((cones, -defect[over]))[0])])
                path = self._slit_to(g, int(g.node_chart[pick]), pick)
                kind = "slit"

            grew = self._mark_sides(g, path)
            if grew == 0:
                break
            added += grew
            self.extra_cuts += 1
            if kind == "split":
                self.splits += 1
        return added

    # -- driver ---------------------------------------------------------------

    def run(self, *, curvature_threshold: float, seed: int, max_extra_cuts: int) -> int:
        """Cut until every chart is a disk, then until distortion is satisfied.

        Returns the number of topology rounds run.
        """
        ct = self.measure()
        limit = 8 + 2 * int(ct.n_charts + int(ct.chart_loops.sum()))
        rounds = 0
        for _ in range(max(1, limit)):
            bad = np.flatnonzero(ct.chart_chi != 1)
            if bad.size == 0:
                break
            rounds += 1
            g = self.graph(ct)
            changed = 0
            for chart in bad.tolist():
                if int(ct.chart_loops[chart]) >= 2:
                    changed += self.join_loops(ct, g, chart)
                elif int(ct.chart_chi[chart]) < 1:
                    changed += self.cut_handle_loop(g, chart)
                else:
                    changed += self.cut_arc(g, chart, seed)
            if changed == 0:
                break
            ct = self.measure()

        # Whatever the measured loop could not open, open by brute force.
        for chart in np.flatnonzero(ct.chart_chi != 1).tolist():
            self.open_by_spanning_tree(ct, chart)

        scale = min(max(_finite(curvature_threshold, 0.35), 0.02), 0.95)
        # ``curvature_threshold`` has to move the answer on the mesh this
        # strategy exists for, and on an absolute criterion alone it does not:
        # Suzanne's head carries **188 radians** of interior absolute Gaussian
        # curvature against a sphere's 4 pi, so every chart stays over every
        # budget in the legal range and the cap decides everything.  Measured,
        # thresholds 0.05 through 0.90 all produced the identical 27-chart cut.
        # Letting the threshold spend the allowance as well makes it a single
        # monotone "how hard should I cut" dial on every mesh, which is the
        # same thing it means for ``CURVATURE`` and ``DUAL_GRAPH``.
        allowance = max(2, int(round(max_extra_cuts * (1.0 - scale))))
        self.escalate(
            cone=scale * math.pi,
            chart_budget=scale * 4.0 * math.pi,
            budget=min(allowance, max_extra_cuts),
        )
        return rounds


def concave_seams(
    sample: "UVMeshSample",
    *,
    curvature_threshold: float = 0.35,
    seed: int = 0,
    bias: float = _CONCAVE_BIAS,
    max_extra_cuts: int = _CONCAVE_MAX_EXTRA_CUTS,
    topo: EdgeTopology | None = None,
) -> tuple[np.ndarray, ConcaveReport]:
    """Cut ``sample`` the way an artist would: down the valleys, and no further.

    Why concavity, and why it is not merely a stylistic preference: a seam in a
    concave crease is both **hidden** -- the geometry folds away from the
    camera there -- and **cheap**, because a valley is where the surface is
    least developable and a cut relieves the most distortion per unit length.
    Those two wants coincide, which is the whole idea.

    The algorithm, in order:

    1. **Weight.**  Every edge costs its 3D length scaled down in a valley and
       up on a ridge (:func:`concavity_weights`).
    2. **Cut the topological minimum.**  Measure each chart's Euler
       characteristic on the *cut* surface (:func:`cut_topology`) and add one
       cut per defect: a shortest weighted path joining two boundary loops, a
       tree-cotree loop to remove a handle, or -- for a closed shell -- a path
       between two curvature extremes.  Repeat until ``chi == 1`` everywhere.
       Nothing here is hard-coded per shape: Suzanne's head is discovered to
       need exactly one join because it is measured at ``chi = 0`` with two
       boundary loops, and her eyeballs are discovered to need none.
    3. **Escalate only on evidence.**  Add further cuts only where the measured
       interior Gaussian curvature says a chart cannot be flattened without a
       cone singularity.

    Args:
        sample: The mesh to cut.
        curvature_threshold: Escalation sensitivity, and the only "how many
            islands" dial this strategy has.  **Lower cuts more.**  It sets
            three things at once: the per-vertex cone threshold
            (``threshold * pi`` radians, so the default 0.35 fires at about 63
            degrees of angle defect -- a cube corner at 90 escalates, a smooth
            sphere vertex at 1.5 does not), the per-chart curvature budget
            (``threshold * 4 pi``), and the share of ``max_extra_cuts`` the
            escalation may spend.  The last of those exists because the first
            two alone leave the dial inert on a high-curvature mesh; see
            :meth:`_ConcaveCutter.run`.
        seed: Rotates the source among a closed shell's sharpest few curvature
            extremes, so different grid candidates get different valid cuts.
        bias: Concavity preference strength; see :func:`concavity_weights`.
        max_extra_cuts: Ceiling on the distortion escalation.
        topo: Optional prebuilt edge topology.

    Returns:
        ``(mask, report)`` with ``mask`` an ``(E,)`` bool seam mask and
        ``report`` a :class:`ConcaveReport` of measured counts.  Never raises:
        a mesh this cannot analyse comes back with an empty cut and a report
        that says so.
    """
    if topo is None:
        topo = build_edge_topology(sample)
    if topo.n_edges == 0 or topo.n_tris == 0:
        return np.zeros(max(topo.n_edges, 0), dtype=bool), ConcaveReport()

    sides = _triangle_sides(sample, topo)
    cutter = _ConcaveCutter(sample, topo, sides, bias=bias)
    rounds = 0
    try:
        rounds = cutter.run(
            curvature_threshold=curvature_threshold,
            seed=_as_seed(seed),
            max_extra_cuts=max(0, int(max_extra_cuts)),
        )
    except Exception:
        # ARCHITECTURE.md section 1.4: a failure is contained, never thrown.
        # Whatever was already marked is still a legal -- if incomplete -- cut,
        # and the dual-graph strategy is the guaranteed-openable fallback.
        cutter.seam |= _dual_graph_seams(
            topo, curvature_threshold=_finite(curvature_threshold, 0.35), seed=_as_seed(seed)
        )
        cutter.fallbacks += 1

    seam = cutter.seam & topo.cuttable_mask
    ct = cut_topology(sample, seam, topo=topo, sides=sides)
    _, shells = _face_components(topo)

    length = np.where(np.isfinite(topo.edge_len), topo.edge_len, 0.0)
    man = topo.face_count == 2
    marked_man = seam & man
    concave = marked_man & (cutter.signed < 0.0)
    marked_len = float(length[marked_man].sum())
    mesh_len = float(length[man].sum())
    return seam, ConcaveReport(
        shells=shells,
        charts=ct.n_charts,
        disks=int((ct.chart_chi == 1).sum()),
        non_disks=int((ct.chart_chi != 1).sum()),
        cut_edges=int(seam.sum()),
        cut_length=float(length[seam].sum()),
        concave_edges=int(concave.sum()),
        concave_length=float(length[concave].sum()),
        concave_fraction=float(length[concave].sum() / marked_len) if marked_len else 0.0,
        baseline_fraction=(
            float(length[man & (cutter.signed < 0.0)].sum() / mesh_len) if mesh_len else 0.0
        ),
        joins=cutter.joins,
        handle_loops=cutter.handle_loops,
        arcs=cutter.arcs,
        extra_cuts=cutter.extra_cuts,
        splits=cutter.splits,
        fallbacks=cutter.fallbacks,
        rounds=rounds,
    )


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
            :func:`edge_curvature`), the dangling-cut protection level for
            ``DUAL_GRAPH``, and the escalation sensitivity for ``CONCAVE``
            (see :func:`concave_seams`).
        existing: Optional ``(E,)`` seam mask overriding ``sample.edge_seam``.
            Ignored when its length does not match the edge count.
        seed: Determinism seed.  ``DUAL_GRAPH`` and ``CONCAVE`` consume it,
            where it selects among equally-good cuts.

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

    if strat == "CONCAVE":
        return concave_seams(
            sample,
            curvature_threshold=_finite(curvature_threshold, 0.35),
            seed=_as_seed(seed),
            topo=topo,
        )[0]

    return _dual_graph_seams(
        topo,
        curvature_threshold=_finite(curvature_threshold, 0.35),
        seed=_as_seed(seed),
    )
