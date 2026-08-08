"""Blender-free mesh interchange used by every other core module.

`UVMeshSample` is a flat, numpy-backed snapshot of a triangulated mesh plus one
UV layer.  It is the *only* structure that crosses from :mod:`bpy` land into the
pure-Python core, which is what makes the evaluator, the parameter space and the
seam heuristics unit-testable without Blender.

Nothing in this module may import :mod:`bpy`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

__all__ = [
    "UVMeshSample",
    "tri_areas_3d",
    "tri_areas_uv",
    "degenerate_mask",
    "make_sample",
]


@dataclass(frozen=True)
class UVMeshSample:
    """An immutable, triangulated snapshot of a mesh and one of its UV layers.

    All arrays are plain numpy and index-aligned as described below.  ``V`` is
    the vertex count, ``T`` the triangle count, ``L`` the loop (face-corner)
    count and ``E`` the edge count.

    Attributes:
        verts: ``(V, 3)`` float64 object-space vertex positions.
        tri_verts: ``(T, 3)`` int32 vertex index per triangle corner.
        tri_loops: ``(T, 3)`` int32 loop index per triangle corner.  Used to
            look up per-corner UVs, which -- unlike vertices -- are split across
            seams.
        uvs: ``(L, 2)`` float64 UV coordinate per loop.
        edge_verts: ``(E, 2)`` int32 vertex indices per edge.
        edge_seam: ``(E,)`` bool, ``True`` where the edge is marked as a seam.
        loop_verts: ``(L,)`` int32 vertex index per loop.
        name: Optional label carried through for diagnostics.
    """

    verts: np.ndarray
    tri_verts: np.ndarray
    tri_loops: np.ndarray
    uvs: np.ndarray
    edge_verts: np.ndarray
    edge_seam: np.ndarray
    loop_verts: np.ndarray
    name: str = ""

    # -- sizes ---------------------------------------------------------------

    @property
    def n_verts(self) -> int:
        """Number of vertices."""
        return int(self.verts.shape[0])

    @property
    def n_tris(self) -> int:
        """Number of triangles."""
        return int(self.tri_verts.shape[0])

    @property
    def n_loops(self) -> int:
        """Number of loops (face corners)."""
        return int(self.uvs.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return int(self.edge_verts.shape[0])

    # -- derived views -------------------------------------------------------

    def tri_positions(self) -> np.ndarray:
        """Return ``(T, 3, 3)`` float64 world-space triangle corner positions."""
        if self.n_tris == 0:
            return np.zeros((0, 3, 3), dtype=np.float64)
        return self.verts[self.tri_verts]

    def tri_uvs(self) -> np.ndarray:
        """Return ``(T, 3, 2)`` float64 UV coordinates of each triangle corner."""
        if self.n_tris == 0:
            return np.zeros((0, 3, 2), dtype=np.float64)
        return self.uvs[self.tri_loops]

    def with_uvs(self, uvs: np.ndarray) -> "UVMeshSample":
        """Return a copy of this sample carrying a different UV layer.

        Args:
            uvs: ``(L, 2)`` array of replacement UVs.

        Returns:
            A new :class:`UVMeshSample` sharing every other array.

        Raises:
            ValueError: If ``uvs`` does not have shape ``(L, 2)``.
        """
        arr = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] != self.n_loops:
            raise ValueError(
                "uv count %d does not match loop count %d" % (arr.shape[0], self.n_loops)
            )
        return replace(self, uvs=arr)

    def with_seams(self, edge_seam: np.ndarray) -> "UVMeshSample":
        """Return a copy of this sample carrying a different seam mask."""
        arr = np.asarray(edge_seam, dtype=bool).reshape(-1)
        if arr.shape[0] != self.n_edges:
            raise ValueError(
                "seam count %d does not match edge count %d" % (arr.shape[0], self.n_edges)
            )
        return replace(self, edge_seam=arr)

    # -- diagnostics ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of human-readable structural problems.

        This is deliberately non-fatal: bad geometry is reported, never raised,
        so that a single broken mesh cannot take Blender down.  An empty list
        means the sample is internally consistent.
        """
        problems: list[str] = []
        if self.verts.ndim != 2 or self.verts.shape[1] != 3:
            problems.append("verts must be (V, 3)")
        if self.tri_verts.shape != self.tri_loops.shape:
            problems.append("tri_verts and tri_loops must have identical shape")
        if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
            problems.append("uvs must be (L, 2)")
        if self.n_tris and self.n_verts:
            if int(self.tri_verts.max()) >= self.n_verts:
                problems.append("tri_verts references a vertex index out of range")
            if int(self.tri_loops.max()) >= self.n_loops:
                problems.append("tri_loops references a loop index out of range")
        if self.edge_seam.shape[0] != self.n_edges:
            problems.append("edge_seam length does not match edge count")
        if self.n_verts and not np.isfinite(self.verts).all():
            problems.append("vertex positions contain NaN or Inf")
        if self.n_loops and not np.isfinite(self.uvs).all():
            problems.append("UV coordinates contain NaN or Inf")
        if self.n_tris == 0:
            problems.append("mesh has no triangles")
        return problems

    def summary(self) -> str:
        """Return a compact one-line description, for logs and reports."""
        return "%s V=%d T=%d L=%d E=%d seams=%d" % (
            self.name or "<mesh>",
            self.n_verts,
            self.n_tris,
            self.n_loops,
            self.n_edges,
            int(self.edge_seam.sum()) if self.n_edges else 0,
        )


def make_sample(
    verts: Sequence[Sequence[float]],
    tri_verts: Sequence[Sequence[int]],
    *,
    tri_loops: Sequence[Sequence[int]] | None = None,
    uvs: Sequence[Sequence[float]] | None = None,
    edge_verts: Sequence[Sequence[int]] | None = None,
    edge_seam: Sequence[bool] | None = None,
    loop_verts: Sequence[int] | None = None,
    name: str = "",
) -> UVMeshSample:
    """Build a :class:`UVMeshSample` from loose Python sequences.

    Convenience constructor for tests and for callers that do not have the
    Blender fast paths available.  Missing arrays are derived where possible:
    ``tri_loops`` defaults to one loop per triangle corner, ``uvs`` to zeros,
    and ``edge_verts`` to the unique undirected edges of ``tri_verts``.

    Args:
        verts: ``(V, 3)`` vertex positions.
        tri_verts: ``(T, 3)`` vertex indices per triangle.
        tri_loops: Optional ``(T, 3)`` loop indices per triangle.
        uvs: Optional ``(L, 2)`` UVs.
        edge_verts: Optional ``(E, 2)`` edge vertex pairs.
        edge_seam: Optional ``(E,)`` seam flags.
        loop_verts: Optional ``(L,)`` vertex index per loop.
        name: Optional label.

    Returns:
        A fully populated :class:`UVMeshSample`.
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tv = np.asarray(tri_verts, dtype=np.int32).reshape(-1, 3)

    if tri_loops is None:
        tl = np.arange(tv.size, dtype=np.int32).reshape(-1, 3)
    else:
        tl = np.asarray(tri_loops, dtype=np.int32).reshape(-1, 3)

    n_loops = int(tl.max()) + 1 if tl.size else 0

    if uvs is None:
        uv = np.zeros((n_loops, 2), dtype=np.float64)
    else:
        uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)

    if loop_verts is None:
        lv = np.zeros(max(n_loops, uv.shape[0]), dtype=np.int32)
        if tl.size:
            lv[tl.reshape(-1)] = tv.reshape(-1)
    else:
        lv = np.asarray(loop_verts, dtype=np.int32).reshape(-1)

    if edge_verts is None:
        if tv.size:
            pairs = np.concatenate(
                [tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], axis=0
            )
            pairs = np.sort(pairs, axis=1)
            ev = np.unique(pairs, axis=0).astype(np.int32)
        else:
            ev = np.zeros((0, 2), dtype=np.int32)
    else:
        ev = np.asarray(edge_verts, dtype=np.int32).reshape(-1, 2)

    if edge_seam is None:
        es = np.zeros(ev.shape[0], dtype=bool)
    else:
        es = np.asarray(edge_seam, dtype=bool).reshape(-1)

    return UVMeshSample(
        verts=v,
        tri_verts=tv,
        tri_loops=tl,
        uvs=uv,
        edge_verts=ev,
        edge_seam=es,
        loop_verts=lv,
        name=name,
    )


# ---------------------------------------------------------------------------
# Vectorised geometric helpers shared by the evaluator and the seam heuristics.
# ---------------------------------------------------------------------------


def tri_areas_3d(sample: UVMeshSample) -> np.ndarray:
    """Return ``(T,)`` float64 triangle areas in 3D object space."""
    if sample.n_tris == 0:
        return np.zeros(0, dtype=np.float64)
    p = sample.tri_positions()
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def tri_areas_uv(sample: UVMeshSample) -> np.ndarray:
    """Return ``(T,)`` float64 **signed** triangle areas in UV space.

    A negative area means the triangle is mirrored (flipped) in UV space, which
    the evaluator counts separately -- flipped faces break normal maps.
    """
    if sample.n_tris == 0:
        return np.zeros(0, dtype=np.float64)
    q = sample.tri_uvs()
    e1 = q[:, 1] - q[:, 0]
    e2 = q[:, 2] - q[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def degenerate_mask(sample: UVMeshSample, eps: float = 1e-12) -> np.ndarray:
    """Return ``(T,)`` bool marking triangles unusable for distortion metrics.

    A triangle is degenerate when its 3D area or its absolute UV area falls
    below ``eps``, or when either carries a non-finite coordinate.  These are
    excluded from every mean so that a single pinched face cannot poison the
    score of an otherwise good unwrap.
    """
    if sample.n_tris == 0:
        return np.zeros(0, dtype=bool)
    a3 = tri_areas_3d(sample)
    auv = np.abs(tri_areas_uv(sample))
    bad = (a3 <= eps) | (auv <= eps)
    p = sample.tri_positions()
    q = sample.tri_uvs()
    bad |= ~np.isfinite(p).all(axis=(1, 2))
    bad |= ~np.isfinite(q).all(axis=(1, 2))
    return bad
