"""Chart topology and distortion-bounded chart merging.

Merging adjacent charts is the largest single lever this add-on has on an
organic mesh, and ``docs/ORGANIC.md`` section 0.4 is the measurement that says
so: on Suzanne's winning layout **100.0% of the 31.613 units of cut length is
boundary between two islands and none of it is an intra-island slit**, so the
whole seam budget is reachable by merging, subject only to keeping each merged
chart a disk and its distortion acceptable.  Seam is 44.1% of the total penalty
(section 0.2), which makes this the highest-value refinement stage.

What this module is careful about
---------------------------------

**Disk-ness is decided exactly, and by more than the Euler number.**
ORGANIC.md section 5 specifies "the exact Euler test ``chi = F - E + V == 1``
over the triangulated submesh".  That is necessary but *not sufficient*: two
disks joined at a single vertex -- a bowtie -- have ``F=2, E=6, V=5`` and
therefore ``chi = 1`` while being nothing a flattener can handle.  So
:func:`is_disk` computes the Euler characteristic exactly as specified and then
also requires the submesh to be connected, edge-manifold, vertex-manifold
(every vertex link a single fan), **bounded** (at least one edge carrying a
single triangle) and free of triangles that repeat a corner.  Those are the
remaining conditions that make ``chi == 1`` mean "disk"; adding them can only
ever reject more, never accept more, so the specified test is still the one
doing the work.

The boundary condition is not decoration.  Connected + manifold + ``chi == 1``
still admits the real projective plane, which triangulates on six vertices and
is closed, and the usual justification -- "a connected, *orientable*, manifold
surface with ``chi == 1`` must have a boundary" -- assumes orientability rather
than testing it.  Testing for a boundary instead rules RP^2 out exactly, since
``chi = 2 - k - b`` has no solution at ``chi == 1`` with ``k >= 1, b >= 1``.

**Distortion is measured, never estimated.**  Every candidate merge is actually
flattened and the result is run through the frozen evaluator, because the
topological argument in ORGANIC.md section 0.1 permits merging almost the whole
head and only a real measurement stops it.  The flattener is injected
(:mod:`unwrap_variations.core.flatten` when it is importable), so this module is
testable against a stub and is developed in parallel with it.

**The seam length is monotonically non-increasing.**  A merge deletes exactly
the mesh edges lying between its two charts and creates none, so the layout's
cut length can only fall.  That is asserted, per step, in the report.

**Nothing here imports :mod:`bpy`.**  Repacking a merged layout does need
Blender, and lives in :mod:`unwrap_variations.backends.blender_native` where the
rest of the operator handling already is.

Measured on Suzanne
-------------------

Blender 5.1.2 and 4.5.4, ``--factory-startup``, ``generate(count=10, seed=1)``,
``balanced`` scorer, ``max_stretch=1.25``, taken against the layout the backend
produced at the time (score 23.189).  Identical to four decimal places on both
versions:

===========================  ======  =========  ==========  =======  =======
layout                       score   seam_ratio  stretch_l2  overlap  islands
===========================  ======  =========  ==========  =======  =======
backend, unrefined           23.189     8.9556      1.0164  1.34e-3       14
merged, **not** repacked     16.130     6.5274      1.0731  5.12e-2       10
merged and repacked          43.162     6.5274      1.0731        0       10
===========================  ======  =========  ==========  =======  =======

Three things in that table are worth carrying forward.

**A merged layout that is not repacked is worse than no merge at all.**  Pasting
a merged chart where its parents used to sit puts it on top of its neighbours:
``overlap_area`` goes up 38x and costs more than the seam reduction gains.  The
repack is not an optimisation on top of merging, it is part of it -- which is
why the caller must repack *before* judging the stage against rule 2.

**Merging removed the overlap entirely.**  ``overlap_area`` reaches exactly
``0.0``, not merely a smaller number.  ORGANIC.md section 0.3 located the defect
in one chart lapping over a distant part of itself at the brow ridge; the charts
containing it were merged and re-flattened by a solver that certifies its own
output, so the fold had nowhere to survive.  That is worth more than the seam
reduction it was aimed at.

**More merging is not monotonically better.**  Pushed harder the seam keeps
falling and the score does not: 7 islands reads ``seam_ratio`` 5.72 and scores
37.5, 5 islands reads 4.27 and scores 40.3, against 43.2 for 10 islands.  Fewer,
larger, more awkwardly shaped islands pack worse, and past a point the packing
term takes back more than the seam term gives.  ``max_stretch`` is therefore a
genuine trade-off dial and not a "higher is better" one.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..backends import seams as seam_mod
from .evaluator import evaluate
from .meshdata import UVMeshSample, make_sample, tri_areas_3d

__all__ = [
    "Chart",
    "ChartGraph",
    "ChartTopology",
    "MergeStep",
    "MergeReport",
    "chart_graph",
    "chart_from_mask",
    "is_disk",
    "merge_charts",
    "tri_islands",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Coordinate tolerance below which two UVs count as the same UV vertex when
#: deciding whether a mesh edge is split.  Matches
#: :func:`unwrap_variations.core.engine.derive_uv_seams`, so the islands this
#: module sees are exactly the islands the seam metric is measured over.
_UV_EPS = 1e-7

#: Floor on the denominator of the benefit ratio.  A merge that costs no
#: measurable extra distortion has an unbounded benefit, which is the correct
#: reading -- it is free seam reduction -- so the floor only keeps the arithmetic
#: finite and the ordering deterministic.
_DISTORTION_EPS = 1e-6

#: ``stretch_l2`` is provably at least 1.0 (see the evaluator's derivation), so a
#: bound below it can never be satisfied and merging is off.  This is what keeps
#: the *unmerged* layout reachable from the parameter grid, which ORGANIC.md
#: section 5 requires so the benchmark can attribute the gain.
_MERGE_OFF_BELOW = 1.0

#: Guard on the merge loop.  Each accepted merge removes one chart, so the loop
#: is bounded by the chart count regardless; this only stops a pathological
#: re-evaluation cycle from spinning.
_MAX_QUEUE_POPS = 1_000_000

#: Fraction of the flattener's own descent budget used to *screen* a candidate.
#:
#: Measured on Suzanne: the merge stage's cost is >99% inside ``flatten_chart``.
#: Nothing else registers -- the exact disk test is 0.02 s across 57 calls,
#: certification 0.03 s, the evaluator 0.01 s -- and the flattener's cost is very
#: close to linear in its iteration count (on a 166-triangle union: 19 ms at 16
#: iterations, 41 ms at 32, 128 ms at 128, 255 ms at 256).  So the only lever on
#: the cost is how much descent happens per candidate.
#:
#: Screening cheaply is *usually* conservative but **is not a proof**, and the
#: earlier claim that it was has been measured false.  The argument was: fewer
#: descent iterations leave a chart more distorted, so a screened candidate's
#: ``stretch_l2`` is an upper bound on the full-quality figure, and the cheap
#: number can therefore only ever reject a merge that a full flatten would have
#: accepted.  Two things break it.
#:
#: **The descent minimises a different functional from the one the gate reads.**
#: ``minimize_distortion`` descends the symmetric Dirichlet energy; the gate
#: measures Sander's L2 stretch.  The energy is monotone in the budget -- the
#: checkpoint ladder keeps the lowest-energy certifying iterate, and a larger
#: budget checks a superset of checkpoints -- but stretch is not.  Measured over
#: the adjacent pairs of the four corpus layouts, 18 of 35 have some budget at
#: which ``stretch_l2`` rises, worst case 1.2630 at 24 iterations against 1.4755
#: at 32 and above.
#:
#: **"Full quality" is not this constant's denominator.**  ``flatten_chart``
#: with no ``iters`` uses ``_iteration_budget``, which is
#: ``min(max(32000 // tris, 16), 256)`` -- not ``_DEFAULT_ITERS``.  Above 1000
#: triangles that is *below* the screening budget of 32, so screening is the
#: finer flatten and reports the *lower* stretch.  Measured on the corpus
#: sphere's 1578-triangle union: screening reads 1.0515, full quality reads
#: 1.0695, so every bound in ``(1.0515, 1.0695]`` -- inside the grid's own
#: ``[0, 3]`` range -- is admitted by screening and refused at full quality.
#: End to end at 1.060497 the two runs take *different* merges: ``(0, 1)``
#: screened, ``(0, 2)`` at full quality.
#:
#: What does still hold: the chart that ships was itself measured against the
#: bound and passed, because ``finalise`` re-flattens only to *replace* the
#: screened layout and keeps the screened one when the replacement fails the
#: gate.  So the bound is honoured by every shipped chart; what is not sound is
#: the *choice of which merges to take*.  The fix is to clamp the screening
#: budget by the chart's own full-quality budget rather than pin it globally;
#: that changes the accepted merge set on the corpus and needs a re-benchmark,
#: so it is recorded here rather than applied.
_SCREEN_FRACTION = 8

#: Screening budget when the flattener's own default cannot be read.
#:
#: This is a *fraction* rather than a constant because a constant already went
#: stale once and did so silently.  It was set to 8 against a flattener whose
#: default was 32; the flattener later moved its default to 256 and got faster
#: per iteration, at which point 8 iterations left a 166-triangle union at
#: ``stretch_l2 = 1.486`` against 1.116 at 32 -- so every candidate blew the 1.25
#: bound and Suzanne merged **nothing** while still spending the time to find
#: that out.  A screening budget too coarse for its flattener does not fail
#: loudly; it just quietly stops merging.  Tracking the ratio instead means the
#: two move together.
_SCREEN_ITERS_FALLBACK = 32

#: Maximum number of grid insertions the fallback boundary-crossing test will
#: make before giving up and reporting "not certified".
_CROSS_MAX_INSERTS = 4_000_000

#: Cap on the uniform grid used by the fallback boundary-crossing test.
_CROSS_MAX_GRID_DIM = 2048


# ---------------------------------------------------------------------------
# Small vectorised helpers
# ---------------------------------------------------------------------------


def _label_components(n_nodes: int, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Label the connected components of an undirected graph, vectorised.

    Shiloach-Vishkin: hook every node to the smallest label in its
    neighbourhood, flatten the pointer forest by doubling, contract the edge
    list, repeat.  A pure-Python union-find needs one interpreted step per edge,
    which is two orders of magnitude too slow at mesh scale.

    Args:
        n_nodes: Number of nodes; ids must lie in ``[0, n_nodes)``.
        ea: ``(E,)`` first endpoint of each edge.
        eb: ``(E,)`` second endpoint of each edge.

    Returns:
        ``(n_nodes,)`` int64 labels; the label is the smallest node id in the
        component, so the result is deterministic.
    """
    labels = np.arange(max(n_nodes, 0), dtype=np.int64)
    ea = np.asarray(ea, dtype=np.int64).reshape(-1)
    eb = np.asarray(eb, dtype=np.int64).reshape(-1)
    if labels.size == 0 or ea.size == 0:
        return labels

    for _ in range(128):
        la = labels[ea]
        lb = labels[eb]
        if np.array_equal(la, lb):
            break
        smallest = np.minimum(la, lb)
        np.minimum.at(labels, ea, smallest)
        np.minimum.at(labels, eb, smallest)
        for _ in range(64):
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
        ra = labels[ea]
        rb = labels[eb]
        keep = ra != rb
        if not keep.any():
            break
        ra = ra[keep]
        rb = rb[keep]
        packed = np.unique(np.minimum(ra, rb) * n_nodes + np.maximum(ra, rb))
        ea = packed // n_nodes
        eb = packed - ea * n_nodes
    return labels


def _compound_key(major: np.ndarray, minor: np.ndarray, minor_base: int) -> np.ndarray | None:
    """Return ``major * minor_base + minor`` as int64, or ``None`` on overflow.

    Packing two indices into one integer turns a slow ``np.unique(..., axis=0)``
    into a plain 1-D unique.  Returning ``None`` rather than wrapping silently is
    the point: a wrapped key would merge two distinct charts' edges and report a
    wrong Euler characteristic, which is exactly the class of bug that makes a
    disk test look like it works.
    """
    if minor_base <= 0:
        minor_base = 1
    top = int(major.max()) if major.size else 0
    tail = int(minor.max()) if minor.size else 0
    if top > 0 and minor_base > (np.iinfo(np.int64).max - tail) // top:
        return None
    return major.astype(np.int64) * np.int64(minor_base) + minor.astype(np.int64)


def _dense_labels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dense, originals)`` renumbering ``values`` to ``0..K-1``.

    ``originals[i]`` is the input label that became ``i``, in ascending order, so
    the renumbering is deterministic and reversible.
    """
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr, np.zeros(0, dtype=np.int64)
    originals, dense = np.unique(arr, return_inverse=True)
    return dense.reshape(-1).astype(np.int64), originals.astype(np.int64)


def _as_tri_mask(tri_mask: Any, n_tris: int) -> np.ndarray:
    """Coerce a boolean mask or an index array into a ``(T,)`` bool mask."""
    arr = np.asarray(tri_mask)
    if arr.dtype == bool:
        out = np.zeros(n_tris, dtype=bool)
        take = min(arr.shape[0], n_tris) if arr.ndim else 0
        if take:
            out[:take] = arr.reshape(-1)[:take]
        return out
    out = np.zeros(n_tris, dtype=bool)
    idx = arr.reshape(-1).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_tris)]
    out[idx] = True
    return out


# ---------------------------------------------------------------------------
# Exact chart topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartTopology:
    """Exact per-chart topology of a triangulated submesh.

    Every field is indexed by dense chart id.  ``chi`` is the Euler
    characteristic ``F - E + V`` over the **triangulated** edge set, which is the
    only edge set that can answer the question: ``sample.n_edges`` is the quad
    mesh's edge count and excludes the tessellation diagonals (1005 against 1473
    on Suzanne), so computing ``chi`` from it silently reports the wrong number.

    Attributes:
        n_charts: Number of charts described.
        faces: ``(K,)`` triangle count.
        edges: ``(K,)`` distinct triangulated edge count.
        verts: ``(K,)`` distinct vertex count.
        chi: ``(K,)`` Euler characteristic ``F - E + V``.
        components: ``(K,)`` number of connected components.
        edge_manifold: ``(K,)`` bool, no edge carries three or more triangles.
        vertex_manifold: ``(K,)`` bool, every vertex link is a single fan.
        boundary_edges: ``(K,)`` edges carrying exactly one triangle.
        degenerate_faces: ``(K,)`` triangles with a repeated corner index.
    """

    n_charts: int
    faces: np.ndarray
    edges: np.ndarray
    verts: np.ndarray
    chi: np.ndarray
    components: np.ndarray
    edge_manifold: np.ndarray
    vertex_manifold: np.ndarray
    boundary_edges: np.ndarray
    degenerate_faces: np.ndarray

    @property
    def disk(self) -> np.ndarray:
        """``(K,)`` bool: charts that are genuinely a topological disk.

        ``chi == 1`` on its own admits a bowtie (two disks sharing one vertex:
        ``F=2, E=6, V=5``) and a disk-plus-torus union (``1 + 0``), so
        connectedness and manifoldness are required alongside it.

        Those four are still not enough, and the reason is worth stating because
        the obvious justification quietly assumes its way past it.  "A
        connected, **orientable**, manifold surface with ``chi == 1`` must have
        a boundary and is therefore a disk" is true, but orientability is a
        *hypothesis* of that sentence, and none of the four conditions tests it.
        The classification of surfaces gives ``chi = 2 - 2g - b`` orientable and
        ``chi = 2 - k - b`` non-orientable, so ``chi == 1`` forces either
        ``(g=0, b=1)`` -- the disk -- or ``(k=1, b=0)``, the real projective
        plane.  RP^2 triangulates on six vertices (``F=10, E=15, V=6``), is
        connected, is edge- and vertex-manifold, and is not a disk.

        Rather than test orientability, which needs a parity union-find over the
        dual graph, test the conclusion: **a disk has a boundary and RP^2 does
        not**.  ``2 - k - b == 1`` with ``k >= 1`` and ``b >= 1`` has no
        solution, so requiring one boundary edge rules the non-orientable case
        out exactly, for the cost of one more ``bincount``.

        ``degenerate_faces`` covers the last gap, which is not a surface
        question at all: a triangle written ``(a, a, b)`` has ``F=1, E=2, V=2``
        and so ``chi == 1``, one component, one boundary edge and manifold links.
        It is a sliver, not a chart, and no flattener should be handed one.
        """
        return (
            (self.faces > 0)
            & (self.chi == 1)
            & (self.components == 1)
            & self.edge_manifold
            & self.vertex_manifold
            & (self.boundary_edges > 0)
            & (self.degenerate_faces == 0)
        )


def chart_topology(
    tri_verts: np.ndarray, labels: np.ndarray, n_charts: int
) -> ChartTopology:
    """Compute exact topology for every chart of a labelled triangle soup.

    Args:
        tri_verts: ``(m, 3)`` vertex index per triangle corner.
        labels: ``(m,)`` dense chart id per triangle, in ``[0, n_charts)``.
        n_charts: Number of charts.

    Returns:
        A :class:`ChartTopology`.  Never raises; a malformed input yields zeroed
        counts, which read as "not a disk" everywhere.

    Notes:
        The vertex-manifold test works on **corners** rather than on vertices:
        two triangles meeting at a vertex are linked when they share an edge
        through it, so the number of connected components of the corner graph
        equals the total number of vertex-link components.  A chart's links are
        all single fans exactly when that count equals its vertex count -- one
        component labelling instead of a per-vertex loop.
    """
    tv = np.asarray(tri_verts, dtype=np.int64).reshape(-1, 3)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    k = max(int(n_charts), 0)
    m = int(min(tv.shape[0], lab.shape[0]))
    zeros_i = np.zeros(k, dtype=np.int64)
    if m == 0 or k == 0:
        return ChartTopology(
            n_charts=k,
            faces=zeros_i,
            edges=zeros_i.copy(),
            verts=zeros_i.copy(),
            chi=zeros_i.copy(),
            components=zeros_i.copy(),
            edge_manifold=np.zeros(k, dtype=bool),
            vertex_manifold=np.zeros(k, dtype=bool),
            boundary_edges=zeros_i.copy(),
            degenerate_faces=zeros_i.copy(),
        )
    tv = tv[:m]
    lab = np.clip(lab[:m], 0, k - 1)

    faces = np.bincount(lab, minlength=k).astype(np.int64)
    base = int(tv.max()) + 1 if tv.size else 1

    # -- vertices ------------------------------------------------------------
    flat_vert = tv.reshape(-1)
    corner_label = np.repeat(lab, 3)
    vkey = _compound_key(corner_label, flat_vert, base)
    if vkey is None:  # pragma: no cover - needs a >1e9-vertex mesh
        stacked = np.stack([corner_label, flat_vert], axis=1)
        uniq_v = np.unique(stacked, axis=0)
        verts = np.bincount(uniq_v[:, 0], minlength=k).astype(np.int64)
    else:
        uniq_v = np.unique(vkey)
        verts = np.bincount((uniq_v // base).astype(np.int64), minlength=k).astype(np.int64)

    # -- edges ---------------------------------------------------------------
    # Block j holds the edge between corners j and (j+1)%3, so flat edge index
    # ``j * m + t`` recovers both corner ids of that edge without a lookup.
    pairs = np.concatenate([tv[:, [0, 1]], tv[:, [1, 2]], tv[:, [2, 0]]], axis=0)
    pairs = np.sort(pairs, axis=1)
    pair_key = pairs[:, 0] * np.int64(base) + pairs[:, 1]
    edge_label = np.tile(lab, 3)
    ekey = _compound_key(edge_label, pair_key, base * base)
    if ekey is None:  # pragma: no cover - needs a >1e6-vertex mesh
        order = np.lexsort((pair_key, edge_label))
        sorted_lab = edge_label[order]
        sorted_pair = pair_key[order]
        is_new = np.empty(order.shape[0], dtype=bool)
        is_new[0] = True
        is_new[1:] = (sorted_lab[1:] != sorted_lab[:-1]) | (sorted_pair[1:] != sorted_pair[:-1])
        inverse = np.cumsum(is_new) - 1
        inv = np.empty_like(inverse)
        inv[order] = inverse
        counts = np.bincount(inv).astype(np.int64)
        group_label = sorted_lab[is_new]
    else:
        uniq_e, inv, counts = np.unique(ekey, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1).astype(np.int64)
        counts = counts.astype(np.int64)
        group_label = (uniq_e // (base * base)).astype(np.int64)

    edges = np.bincount(group_label, minlength=k).astype(np.int64)
    bad_edge = counts > 2
    edge_manifold = np.ones(k, dtype=bool)
    if bad_edge.any():
        edge_manifold[np.unique(group_label[bad_edge])] = False

    # Edges carrying a single triangle.  A disk has some; RP^2 -- the one
    # non-disk that clears every other condition -- has none.  See
    # :meth:`ChartTopology.disk`.
    boundary_edges = np.bincount(
        group_label[counts == 1], minlength=k
    ).astype(np.int64)

    degenerate = (
        (tv[:, 0] == tv[:, 1]) | (tv[:, 1] == tv[:, 2]) | (tv[:, 2] == tv[:, 0])
    )
    degenerate_faces = np.bincount(lab[degenerate], minlength=k).astype(np.int64)

    chi = faces - edges + verts

    # -- adjacency across the shared (multiplicity 2) edges -------------------
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]])
    twin = counts == 2
    if twin.any():
        left = order[starts[twin]]
        right = order[starts[twin] + 1]
        tri_left = left % m
        tri_right = right % m
        comp_tri = _label_components(m, tri_left, tri_right)
    else:
        left = np.zeros(0, dtype=np.int64)
        right = left
        tri_left = left
        tri_right = left
        comp_tri = np.arange(m, dtype=np.int64)

    ckey = _compound_key(lab, comp_tri, m)
    if ckey is None:  # pragma: no cover - defensive
        components = np.bincount(lab, minlength=k).astype(np.int64)
    else:
        comp_ids = np.unique(ckey)
        components = np.bincount((comp_ids // m).astype(np.int64), minlength=k).astype(np.int64)

    # -- vertex links ---------------------------------------------------------
    if left.size:
        # Corner ids of each side of a shared edge, then matched by vertex so the
        # two fans are stitched at the right corners regardless of orientation.
        slot_l = left // m
        slot_r = right // m
        la0 = tri_left * 3 + slot_l
        la1 = tri_left * 3 + (slot_l + 1) % 3
        lb0 = tri_right * 3 + slot_r
        lb1 = tri_right * 3 + (slot_r + 1) % 3
        same = flat_vert[la0] == flat_vert[lb0]
        pa0 = np.where(same, lb0, lb1)
        pa1 = np.where(same, lb1, lb0)
        corner_comp = _label_components(
            3 * m,
            np.concatenate([la0, la1]),
            np.concatenate([pa0, pa1]),
        )
    else:
        corner_comp = np.arange(3 * m, dtype=np.int64)

    lkey = _compound_key(corner_label, corner_comp, 3 * m)
    if lkey is None:  # pragma: no cover - defensive
        vertex_manifold = np.zeros(k, dtype=bool)
    else:
        link_ids = np.unique(lkey)
        link_counts = np.bincount(
            (link_ids // (3 * m)).astype(np.int64), minlength=k
        ).astype(np.int64)
        vertex_manifold = link_counts == verts

    return ChartTopology(
        n_charts=k,
        faces=faces,
        edges=edges,
        verts=verts,
        chi=chi,
        components=components,
        edge_manifold=edge_manifold,
        vertex_manifold=vertex_manifold,
        boundary_edges=boundary_edges,
        degenerate_faces=degenerate_faces,
    )


def is_disk(sample: UVMeshSample, tri_mask: Any) -> bool:
    """Return whether the selected triangles form a single topological disk.

    The test is the exact Euler characteristic ``chi = F - E + V`` over the
    **triangulated** submesh -- ``F`` its triangles, ``E`` its distinct triangle
    edges *including the tessellation diagonals*, ``V`` its distinct vertices --
    combined with the conditions that make ``chi == 1`` actually mean "disk":
    the submesh must be connected, no edge may carry three or more of its
    triangles, every vertex link must be a single fan, it must have at least one
    boundary edge, and no triangle may repeat a corner.  See
    :meth:`ChartTopology.disk` for why each is needed; the bowtie has ``chi == 1``
    and is not a disk, and so does the real projective plane, which clears every
    condition except the boundary one.

    Args:
        sample: The mesh the triangles are indexed into.
        tri_mask: ``(T,)`` boolean mask, or an array of triangle indices.

    Returns:
        ``True`` only when the selection is a disk.  An empty selection is not a
        disk and returns ``False``.  Never raises.
    """
    try:
        n_tris = int(sample.n_tris)
        if n_tris == 0:
            return False
        mask = _as_tri_mask(tri_mask, n_tris)
        picked = np.flatnonzero(mask)
        if picked.size == 0:
            return False
        tv = np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3)[picked]
        topo = chart_topology(tv, np.zeros(picked.size, dtype=np.int64), 1)
        return bool(topo.disk[0])
    except Exception:  # noqa: BLE001 - a topology question must never raise
        return False


# ---------------------------------------------------------------------------
# Islands implied by a UV layout
# ---------------------------------------------------------------------------


def _split_edge_mask(
    sample: UVMeshSample,
    topo: "seam_mod.EdgeTopology",
    uvs: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Return ``(E,)`` bool: interior mesh edges whose two faces disagree in UV.

    This is the same measurement
    :func:`unwrap_variations.core.engine.derive_uv_seams` makes, reproduced here
    rather than imported so that this module stays free of the engine's Blender
    imports and keeps its own definition of "island" pinned to the definition the
    seam metric uses.
    """
    n_edges = int(topo.n_edges)
    out = np.zeros(max(n_edges, 0), dtype=bool)
    if n_edges == 0 or sample.n_tris == 0:
        return out

    interior = np.asarray(topo.face_count, dtype=np.int64) == 2
    idx = np.flatnonzero(interior)
    if idx.size == 0:
        return out

    fa = np.asarray(topo.face_a, dtype=np.int64)[idx]
    fb = np.asarray(topo.face_b, dtype=np.int64)[idx]
    n_tris = int(sample.n_tris)
    valid = (fa >= 0) & (fa < n_tris) & (fb >= 0) & (fb < n_tris)
    if not valid.all():
        idx, fa, fb = idx[valid], fa[valid], fb[valid]
    if idx.size == 0:
        return out

    ev = np.asarray(sample.edge_verts, dtype=np.int64).reshape(-1, 2)[idx]
    tv_a = np.asarray(sample.tri_verts, dtype=np.int64)[fa]
    tv_b = np.asarray(sample.tri_verts, dtype=np.int64)[fb]
    tl_a = np.asarray(sample.tri_loops, dtype=np.int64)[fa]
    tl_b = np.asarray(sample.tri_loops, dtype=np.int64)[fb]
    rows = np.arange(idx.size)

    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] < sample.n_loops:
        return out

    split = np.zeros(idx.size, dtype=bool)
    for corner in (0, 1):
        vert = ev[:, corner]
        eq_a = tv_a == vert[:, None]
        eq_b = tv_b == vert[:, None]
        both = eq_a.any(axis=1) & eq_b.any(axis=1)
        la = tl_a[rows, eq_a.argmax(axis=1)]
        lb = tl_b[rows, eq_b.argmax(axis=1)]
        delta = np.abs(uv[la] - uv[lb]).max(axis=1)
        split |= both & (delta > eps)

    out[idx] = split
    return out


def tri_islands(
    sample: UVMeshSample,
    *,
    uvs: np.ndarray | None = None,
    topo: "seam_mod.EdgeTopology | None" = None,
    eps: float = _UV_EPS,
) -> np.ndarray:
    """Return ``(T,)`` int64 island id per triangle, implied by a UV layout.

    Two triangles share an island when a chain of mesh edges joins them without
    ever crossing a UV split.  Tessellation diagonals always join, because
    Blender cannot cut a quad down its diagonal, so the two triangles of one
    n-gon are necessarily in the same island.

    Args:
        sample: The mesh.  Its own ``uvs`` are used unless ``uvs`` is given.
        uvs: Optional ``(L, 2)`` layout to measure instead.
        topo: Optional prebuilt edge topology, to avoid rebuilding it per
            candidate.
        eps: UV coordinate tolerance for "the two faces agree".

    Returns:
        ``(T,)`` int64 dense island ids starting at ``0``.  Never raises; an
        empty mesh returns an empty array.
    """
    n_tris = int(sample.n_tris)
    if n_tris == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        edge_topo = topo if topo is not None else seam_mod.build_edge_topology(sample)
        layout = sample.uvs if uvs is None else uvs
        split = _split_edge_mask(sample, edge_topo, layout, float(eps))

        manifold = np.asarray(edge_topo.face_count, dtype=np.int64) == 2
        joined = np.flatnonzero(manifold & ~split)
        ea = np.concatenate(
            [np.asarray(edge_topo.face_a, dtype=np.int64)[joined], edge_topo.diag_a]
        )
        eb = np.concatenate(
            [np.asarray(edge_topo.face_b, dtype=np.int64)[joined], edge_topo.diag_b]
        )
        labels = _label_components(n_tris, ea, eb)
    except Exception:  # noqa: BLE001 - degrade to "one island" rather than raise
        return np.zeros(n_tris, dtype=np.int64)
    dense, _ = _dense_labels(labels)
    return dense


# ---------------------------------------------------------------------------
# The chart graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartGraph:
    """Adjacency, shared boundary length and per-chart area/topology.

    Attributes:
        n_charts: Number of charts.
        labels: ``(T,)`` int64 dense chart id per triangle.
        originals: ``(K,)`` int64 chart id each dense id came from.
        area: ``(K,)`` float64 3D surface area per chart.
        tri_count: ``(K,)`` int64 triangle count per chart.
        pair_a: ``(P,)`` int64 lower chart id of each adjacent pair.
        pair_b: ``(P,)`` int64 higher chart id of each adjacent pair.
        pair_length: ``(P,)`` float64 shared boundary length, in world units.
        seam_length: Total cut length attributable to chart boundaries.  On a
            layout whose islands are its charts this equals the evaluator's
            ``seam_length`` exactly.
        topology: Exact per-chart :class:`ChartTopology`.
    """

    n_charts: int
    labels: np.ndarray
    originals: np.ndarray
    area: np.ndarray
    tri_count: np.ndarray
    pair_a: np.ndarray
    pair_b: np.ndarray
    pair_length: np.ndarray
    seam_length: float
    topology: ChartTopology

    def neighbours(self, chart: int) -> list[tuple[int, float]]:
        """Return ``(other_chart, shared_length)`` for one chart, ascending."""
        hit_a = np.flatnonzero(self.pair_a == chart)
        hit_b = np.flatnonzero(self.pair_b == chart)
        out = [(int(self.pair_b[i]), float(self.pair_length[i])) for i in hit_a]
        out.extend((int(self.pair_a[i]), float(self.pair_length[i])) for i in hit_b)
        out.sort()
        return out

    def summary(self) -> str:
        """Return a compact one-line description, for logs and reports."""
        return "charts=%d pairs=%d seam=%.3f disks=%d" % (
            self.n_charts,
            int(self.pair_a.shape[0]),
            self.seam_length,
            int(self.topology.disk.sum()) if self.n_charts else 0,
        )


def chart_graph(
    sample: UVMeshSample,
    tri_chart: np.ndarray,
    *,
    topo: "seam_mod.EdgeTopology | None" = None,
) -> ChartGraph:
    """Build the adjacency graph of a chart partition.

    Two charts are adjacent when a manifold mesh edge separates them, and the
    weight of that adjacency is the summed 3D length of every such edge -- which
    is exactly the cut length a merge of the two would delete.  Non-manifold
    edges are excluded because they can never be interior to a 2-manifold chart
    and are not counted by the seam metric either.  Tessellation diagonals are
    recorded as zero-length adjacencies: a merge across one buys nothing, and
    charts split down a quad's diagonal are not something Blender can express.

    Args:
        sample: The mesh.
        tri_chart: ``(T,)`` chart id per triangle.  Any integer labelling is
            accepted and renumbered densely in ascending order.
        topo: Optional prebuilt edge topology.

    Returns:
        A :class:`ChartGraph`.  Never raises.
    """
    n_tris = int(sample.n_tris)
    labels, originals = _dense_labels(
        np.asarray(tri_chart, dtype=np.int64).reshape(-1)[:n_tris]
    )
    if labels.shape[0] < n_tris:
        labels = np.concatenate(
            [labels, np.zeros(n_tris - labels.shape[0], dtype=np.int64)]
        )
    k = int(originals.shape[0]) if originals.size else 0

    empty_f = np.zeros(0, dtype=np.float64)
    empty_i = np.zeros(0, dtype=np.int64)
    if n_tris == 0 or k == 0:
        return ChartGraph(
            n_charts=0,
            labels=np.zeros(n_tris, dtype=np.int64),
            originals=empty_i,
            area=empty_f,
            tri_count=empty_i.copy(),
            pair_a=empty_i.copy(),
            pair_b=empty_i.copy(),
            pair_length=empty_f.copy(),
            seam_length=0.0,
            topology=chart_topology(np.zeros((0, 3), dtype=np.int64), empty_i, 0),
        )

    areas = tri_areas_3d(sample)
    area = np.bincount(labels, weights=areas, minlength=k).astype(np.float64)
    tri_count = np.bincount(labels, minlength=k).astype(np.int64)

    edge_topo = topo if topo is not None else seam_mod.build_edge_topology(sample)
    face_a = np.asarray(edge_topo.face_a, dtype=np.int64)
    face_b = np.asarray(edge_topo.face_b, dtype=np.int64)
    manifold = np.flatnonzero(np.asarray(edge_topo.face_count, dtype=np.int64) == 2)
    ok = (face_a[manifold] >= 0) & (face_a[manifold] < n_tris)
    ok &= (face_b[manifold] >= 0) & (face_b[manifold] < n_tris)
    manifold = manifold[ok]

    la = labels[face_a[manifold]]
    lb = labels[face_b[manifold]]
    cross = la != lb
    src_a = np.minimum(la[cross], lb[cross])
    src_b = np.maximum(la[cross], lb[cross])
    weights = np.asarray(edge_topo.edge_len, dtype=np.float64)[manifold[cross]]
    weights = np.where(np.isfinite(weights), weights, 0.0)

    # Tessellation diagonals join two triangles of one n-gon; a partition that
    # splits them is not realisable, so record the adjacency at zero length.
    diag_a = np.asarray(edge_topo.diag_a, dtype=np.int64)
    diag_b = np.asarray(edge_topo.diag_b, dtype=np.int64)
    if diag_a.size:
        good = (diag_a >= 0) & (diag_a < n_tris) & (diag_b >= 0) & (diag_b < n_tris)
        da = labels[diag_a[good]]
        db = labels[diag_b[good]]
        dcross = da != db
        if dcross.any():
            src_a = np.concatenate([src_a, np.minimum(da[dcross], db[dcross])])
            src_b = np.concatenate([src_b, np.maximum(da[dcross], db[dcross])])
            weights = np.concatenate([weights, np.zeros(int(dcross.sum()), dtype=np.float64)])

    if src_a.size:
        key = src_a * np.int64(k) + src_b
        uniq, inverse = np.unique(key, return_inverse=True)
        pair_length = np.bincount(
            inverse.reshape(-1), weights=weights, minlength=uniq.size
        ).astype(np.float64)
        pair_a = (uniq // k).astype(np.int64)
        pair_b = (uniq - pair_a * k).astype(np.int64)
    else:
        pair_a = empty_i.copy()
        pair_b = empty_i.copy()
        pair_length = empty_f.copy()

    tv = np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3)
    topology = chart_topology(tv, labels, k)

    return ChartGraph(
        n_charts=k,
        labels=labels,
        originals=originals,
        area=area,
        tri_count=tri_count,
        pair_a=pair_a,
        pair_b=pair_b,
        pair_length=pair_length,
        seam_length=float(pair_length.sum()),
        topology=topology,
    )


# ---------------------------------------------------------------------------
# The unit handed to the flattener
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chart:
    """One chart, re-indexed locally, ready to hand to a flattener.

    Field-for-field compatible with
    :class:`unwrap_variations.core.flatten.Chart` -- ``verts``, ``tris``,
    ``loops``, ``name`` -- so either type can be handed to either flattener.
    ``docs/ORGANIC.md`` section 4.2 names ``flatten_chart(chart)`` without saying
    what ``chart`` is; this is that exchange type, and it is deliberately the
    same shape as the one the flattener settled on.

    Vertices are welded on the **mesh vertex**, not on the UV vertex, which is
    what a merge needs: the seam between the two parents has to close, or the
    merged chart would still carry the cut it was supposed to remove.

    Attributes:
        verts: ``(n, 3)`` float64 positions, locally indexed.
        tris: ``(m, 3)`` int32 local vertex index per triangle corner.
        loops: ``(m, 3)`` int32 loop index of each corner in the originating
            mesh, which is where a flattened UV is written back to.
        name: Optional label carried through for diagnostics.
    """

    verts: np.ndarray
    tris: np.ndarray
    loops: np.ndarray
    name: str = ""

    @property
    def n_verts(self) -> int:
        """Number of vertices in the chart."""
        return int(self.verts.shape[0])

    @property
    def n_tris(self) -> int:
        """Number of triangles in the chart."""
        return int(self.tris.shape[0])

    def area_3d(self) -> float:
        """Return the chart's total 3D surface area."""
        if self.n_tris == 0:
            return 0.0
        p = self.verts[np.asarray(self.tris, dtype=np.int64)]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    def summary(self) -> str:
        """Return a compact one-line description, for logs."""
        return "%s V=%d T=%d" % (self.name or "<chart>", self.n_verts, self.n_tris)


def chart_from_mask(sample: UVMeshSample, tri_mask: Any, *, name: str = "") -> Chart:
    """Extract the selected triangles as a locally-indexed :class:`Chart`.

    Vertex-welded, so a chart spanning what used to be two islands has a single
    vertex where the two sides of the old seam met.  This mirrors
    :func:`unwrap_variations.core.flatten.chart_from_tris` with ``weld="vertex"``
    and is used when that module is not importable.
    """
    n_tris = int(sample.n_tris)
    mask = _as_tri_mask(tri_mask, n_tris)
    picked = np.flatnonzero(mask)
    tv = np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3)[picked]
    tl = np.asarray(sample.tri_loops, dtype=np.int64).reshape(-1, 3)[picked]
    if picked.size == 0:
        return Chart(
            verts=np.zeros((0, 3), dtype=np.float64),
            tris=np.zeros((0, 3), dtype=np.int32),
            loops=tl.astype(np.int32),
            name=name,
        )
    vert_index, local = np.unique(tv.reshape(-1), return_inverse=True)
    verts = np.asarray(sample.verts, dtype=np.float64).reshape(-1, 3)[vert_index]
    return Chart(
        verts=verts,
        tris=local.reshape(-1, 3).astype(np.int32),
        loops=tl.astype(np.int32),
        name=name,
    )


def _build_chart(sample: UVMeshSample, tri_mask: Any, name: str) -> Chart:
    """Build the chart to flatten, preferring the flattener's own constructor.

    :func:`unwrap_variations.core.flatten.chart_from_tris` with ``weld="vertex"``
    produces exactly the structure this module needs and is what that module's
    solver was written against, so use it when it is there and fall back to the
    local equivalent when it is not.
    """
    try:
        from .flatten import chart_from_tris  # noqa: PLC0415 - optional sibling
    except BaseException:  # noqa: BLE001 - the module may not exist yet
        return chart_from_mask(sample, tri_mask, name=name)
    try:
        return chart_from_tris(sample, _as_tri_mask(tri_mask, int(sample.n_tris)),
                               weld="vertex", name=name)
    except BaseException:  # noqa: BLE001 - never let a sibling's bug stop a merge
        return chart_from_mask(sample, tri_mask, name=name)


# ---------------------------------------------------------------------------
# Flattening and certification adapters
# ---------------------------------------------------------------------------


def _resolve_flatten(flatten: Callable[..., Any] | None) -> Callable[..., Any] | None:
    """Return the flattener to validate merges with, or ``None``.

    An explicit callable always wins; otherwise
    :func:`unwrap_variations.core.flatten.flatten_chart` is used when that module
    is importable.  With neither, :func:`merge_charts` performs no merges at all
    -- ORGANIC.md section 5 requires every candidate merge to be validated by
    actually flattening it, and an unvalidated merge is worse than none.
    """
    if flatten is not None:
        return flatten
    try:
        from .flatten import flatten_chart  # noqa: PLC0415 - optional sibling
    except BaseException:  # noqa: BLE001 - the module may not exist yet
        return None
    return flatten_chart


def _screen_iters() -> int:
    """Return the screening descent budget, tracking the flattener's own default.

    Read from :mod:`unwrap_variations.core.flatten` when it is importable so the
    two stay in proportion; see :data:`_SCREEN_ITERS_FALLBACK` for the regression
    that made tracking rather than pinning the right call.
    """
    default = None
    try:
        from . import flatten as flatten_mod  # noqa: PLC0415 - optional sibling

        default = getattr(flatten_mod, "_DEFAULT_ITERS", None)
    except BaseException:  # noqa: BLE001 - the module may not exist yet
        default = None
    try:
        budget = int(default) // _SCREEN_FRACTION
    except (TypeError, ValueError):
        return _SCREEN_ITERS_FALLBACK
    return max(16, budget)


def _resolve_certify(certify: Callable[..., Any] | None) -> Callable[..., Any] | None:
    """Return the injectivity certifier, or ``None`` to use the local fallback."""
    if certify is not None:
        return certify
    try:
        from .injectivity import certify as core_certify  # noqa: PLC0415
    except BaseException:  # noqa: BLE001 - the module may not exist yet
        return None
    return core_certify


def _flatten_uv(
    flatten: Callable[..., Any], chart: Chart, iters: int | None = None
) -> tuple[np.ndarray | None, bool | None]:
    """Call the flattener; return its ``(n, 2)`` UVs and its own verdict.

    The result is read leniently -- a bare array, or an object carrying ``uv``,
    ``uvs`` or ``uv_coords`` -- because ORGANIC.md section 4.2 fixes the function
    name but not the shape of ``FlattenResult``.  When the result does carry an
    ``ok`` flag it is returned alongside: the flattener has already certified its
    own output and disagreeing with it would be re-litigating a decision it made
    with more information (it knows which fallback stage produced the layout).

    Args:
        flatten: The flattener.
        chart: The chart to flatten.
        iters: Optional descent budget.  Passed as ``iters=`` when the flattener
            accepts it and dropped silently when it does not, so an injected
            stub never has to grow the argument.  ``None`` means "the
            flattener's own default" and passes no ``iters`` at all.

    Warning:
        A wrapper that fills the budget in with ``kwargs.setdefault("iters", n)``
        does **not** compose with this: ``setdefault`` cannot override the
        explicit value passed for screening, and it *does* fill in the omission
        that means full quality.  Both budgets then come out inverted, silently.
        Measured on Suzanne with such a wrapper pinned at 16: 24 screening calls
        ran at 32 (this module's value, not the wrapper's) and the 4 polish calls
        ran at 16 (the wrapper's, not the flattener's 256).  A caller that wants
        to choose the screening budget should set ``screen_iters`` on
        :func:`merge_charts`, which is the one knob that owns it.

    Returns:
        ``(uv, ok)``, with ``uv`` ``None`` when nothing usable came back and
        ``ok`` ``None`` when the flattener offered no verdict.
    """
    try:
        result = flatten(chart) if iters is None else flatten(chart, iters=int(iters))
    except TypeError:
        try:
            result = flatten(chart)
        except BaseException:  # noqa: BLE001 - a flattener failure rejects the merge
            return None, None
    except BaseException:  # noqa: BLE001 - a flattener failure rejects the merge
        return None, None
    candidate: Any = result
    verdict: bool | None = None
    if not isinstance(result, np.ndarray):
        if hasattr(result, "ok"):
            verdict = bool(result.ok)
        for attr in ("uv", "uvs", "uv_coords"):
            if hasattr(result, attr):
                candidate = getattr(result, attr)
                break
    try:
        arr = np.asarray(candidate, dtype=np.float64).reshape(-1, 2)
    except BaseException:  # noqa: BLE001
        return None, verdict
    if arr.shape[0] != chart.n_verts or not np.isfinite(arr).all():
        return None, verdict
    return arr, verdict


def _orient(uv: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Return ``uv`` mirrored if needed so signed triangle areas are positive.

    A flattener is free to return a mirrored embedding; pasted into the layout
    as-is it would report every triangle of the merged chart as flipped, which
    the scorer treats as a hard defect.  Mirroring in ``u`` costs nothing and is
    the difference between a valid merge and a rejected one.
    """
    if tris.shape[0] == 0:
        return uv
    q = uv[tris]
    e1 = q[:, 1] - q[:, 0]
    e2 = q[:, 2] - q[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if float(signed.sum()) >= 0.0:
        return uv
    out = uv.copy()
    out[:, 0] = -out[:, 0]
    return out


class _LayoutStretch:
    """Per-chart Sander L2 stretch of an existing layout, additive over charts.

    Exists to kill the other half of the merge cost.  The benefit ratio needs a
    *baseline* distortion to divide by, and flattening each parent chart to get
    one costs one solve per chart -- 14 of the 42 flattens on Suzanne, a third of
    the total, spent on a number that only orders the queue and never gates
    anything.

    The layout the merge is refining already has a measured stretch, and Sander's
    L2 is an area-weighted RMS, so its ingredients are plain sums over triangles:
    with ``H = sum(half_sum * area3)``, ``A = sum(area3)`` and ``U = sum|uv
    area|`` accumulated per chart, any union's stretch is
    ``sqrt(U/A) * sqrt(H/A)`` over the summed accumulators -- exact, and O(1) per
    query.  See the derivation in
    :mod:`unwrap_variations.core.evaluator`'s docstring; the arithmetic here is
    the same, restricted to one chart at a time.

    This is a *baseline*, never a gate.  What decides whether a merge is
    admissible is always the measured stretch of the actually-flattened union.
    """

    __slots__ = ("half", "area", "uv_area")

    def __init__(self, sample: UVMeshSample, uvs: np.ndarray, labels: np.ndarray, k: int):
        self.half = np.zeros(max(k, 0), dtype=np.float64)
        self.area = np.zeros(max(k, 0), dtype=np.float64)
        self.uv_area = np.zeros(max(k, 0), dtype=np.float64)
        if k == 0 or sample.n_tris == 0:
            return
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            self._accumulate(sample, uvs, labels, k)

    def _accumulate(
        self, sample: UVMeshSample, uvs: np.ndarray, labels: np.ndarray, k: int
    ) -> None:
        """Sum the three per-triangle ingredients into per-chart accumulators."""
        tv = np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3)
        tl = np.asarray(sample.tri_loops, dtype=np.int64).reshape(-1, 3)
        verts = np.asarray(sample.verts, dtype=np.float64).reshape(-1, 3)
        uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)

        d1 = verts[tv[:, 1]] - verts[tv[:, 0]]
        d2 = verts[tv[:, 2]] - verts[tv[:, 0]]
        e3 = np.einsum("ij,ij->i", d1, d1)
        f3 = np.einsum("ij,ij->i", d1, d2)
        g3 = np.einsum("ij,ij->i", d2, d2)
        area3 = 0.5 * np.sqrt(np.maximum(e3 * g3 - f3 * f3, 0.0))

        q = uv[tl]
        ds1 = q[:, 1, 0] - q[:, 0, 0]
        dt1 = q[:, 1, 1] - q[:, 0, 1]
        ds2 = q[:, 2, 0] - q[:, 0, 0]
        dt2 = q[:, 2, 1] - q[:, 0, 1]
        signed = 0.5 * (ds1 * dt2 - dt1 * ds2)
        abs_uv = np.abs(signed)

        good = np.isfinite(area3) & np.isfinite(signed) & (area3 > 0.0) & (abs_uv > 0.0)
        if not good.any():
            return
        inv_d = np.where(good, 0.5 / np.where(good, signed, 1.0), 0.0)
        x1, y1 = ds1 * inv_d, dt1 * inv_d
        x2, y2 = ds2 * inv_d, dt2 * inv_d
        a = y2 * y2 * e3 - 2.0 * y1 * y2 * f3 + y1 * y1 * g3
        c = x2 * x2 * e3 - 2.0 * x1 * x2 * f3 + x1 * x1 * g3
        half = 0.5 * (a + c)
        half = np.where(good & np.isfinite(half), half, 0.0)

        lab = np.clip(np.asarray(labels, dtype=np.int64).reshape(-1), 0, k - 1)
        self.half = np.bincount(lab, weights=half * area3, minlength=k).astype(np.float64)
        self.area = np.bincount(
            lab, weights=np.where(good, area3, 0.0), minlength=k
        ).astype(np.float64)
        self.uv_area = np.bincount(
            lab, weights=np.where(good, abs_uv, 0.0), minlength=k
        ).astype(np.float64)

    def of(self, members: Sequence[int]) -> float:
        """Return the layout's L2 stretch over the union of ``members``."""
        idx = list(members)
        if not idx:
            return 1.0
        area = float(self.area[idx].sum())
        if not (area > 0.0):
            return 1.0
        uv_area = float(self.uv_area[idx].sum())
        half = float(self.half[idx].sum())
        if not (uv_area > 0.0) or not math.isfinite(half):
            return 1.0
        value = math.sqrt(uv_area / area) * math.sqrt(max(half / area, 0.0))
        # L2 >= 1 is a theorem; anything below it is float noise.
        return max(value, 1.0) if math.isfinite(value) else 1.0


def _mini_sample(chart: Chart, uv: np.ndarray, name: str = "") -> UVMeshSample:
    """Build a standalone :class:`UVMeshSample` for one flattened chart.

    Measuring through the real evaluator rather than through a private copy of
    the stretch formula is deliberate: the number that gates a merge is then the
    same number the frozen scorer will read back.
    """
    corner_uv = uv[chart.tris.reshape(-1)]
    return make_sample(
        chart.verts,
        chart.tris,
        uvs=corner_uv,
        name=name,
    )


# ---------------------------------------------------------------------------
# Fallback injectivity certification
# ---------------------------------------------------------------------------


def _boundary_loops(tris: np.ndarray, n_verts: int) -> list[np.ndarray]:
    """Return each boundary loop of a triangulated chart as a vertex cycle.

    Boundary edges are the triangle edges carried by exactly one triangle.  On a
    manifold chart every boundary vertex has exactly two of them, so the loops
    fall out of a walk.  A vertex with more than two boundary edges means the
    chart is pinched; the walk stops there and the caller treats the result as
    uncertified rather than guessing.
    """
    if tris.shape[0] == 0:
        return []
    pairs = np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0
    ).astype(np.int64)
    keys = np.sort(pairs, axis=1)
    base = max(int(n_verts), 1)
    flat = keys[:, 0] * base + keys[:, 1]
    uniq, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    lone = counts == 1
    if not lone.any():
        return []
    take = np.isin(inverse.reshape(-1), np.flatnonzero(lone))
    edges = keys[take]

    adjacency: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adjacency.values()):
        return []

    loops: list[np.ndarray] = []
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        previous = -1
        current = start
        while True:
            options = adjacency[current]
            nxt = options[0] if options[0] != previous else options[1]
            if nxt == start:
                break
            if nxt in seen:
                return []
            seen.add(nxt)
            loop.append(nxt)
            previous, current = current, nxt
            if len(loop) > edges.shape[0] + 1:
                return []
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _orient2d(ax, ay, bx, by, cx, cy) -> np.ndarray:
    """Return the signed area sign of triangle ``abc``, vectorised."""
    return np.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


def _bucket_pairs(
    cell_id: np.ndarray, member: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return every within-bucket pair of ``member``, for buckets sorted by key.

    A Python loop over buckets is the obvious implementation and is also the
    thing that makes a broad phase slower than the exact test it was meant to
    accelerate: a uniform grid over ``n`` segments produces ``O(n)`` buckets, so
    the loop costs one interpreted iteration per segment.  This decodes the
    triangular pair index arithmetically instead, in one pass.

    Args:
        cell_id: ``(n,)`` bucket key, already sorted ascending.
        member: ``(n,)`` payload aligned to ``cell_id``.

    Returns:
        ``(a, b)`` payload pairs, or ``(None, None)`` when the pair count would
        exceed the budget -- the caller reports "not certified" rather than
        silently testing a subset.
    """
    n = int(cell_id.shape[0])
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.flatnonzero(np.diff(cell_id)) + 1]
    )
    sizes = np.diff(np.concatenate([starts, np.asarray([n], dtype=np.int64)]))
    multi = sizes >= 2
    if not multi.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = starts[multi]
    sizes = sizes[multi].astype(np.int64)

    per_group = sizes * (sizes - 1) // 2
    total = int(per_group.sum())
    if total > _CROSS_MAX_INSERTS:
        return None, None

    group = np.repeat(np.arange(sizes.shape[0], dtype=np.int64), per_group)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(per_group)[:-1]])
    local = np.arange(total, dtype=np.int64) - offsets[group]
    s = sizes[group].astype(np.float64)
    # Inverse of the row-major upper-triangular numbering: for a group of s
    # items, entry `local` is the pair (i, j) with i < j.
    root = np.sqrt(np.maximum(-8.0 * local + 4.0 * s * (s - 1.0) - 7.0, 0.0))
    i = np.floor(s - 2.0 - (root / 2.0 - 0.5)).astype(np.int64)
    i = np.clip(i, 0, sizes[group] - 2)
    tri_before = sizes[group] * (sizes[group] - 1) // 2 - (sizes[group] - i) * (
        sizes[group] - i - 1
    ) // 2
    j = local - tri_before + i + 1
    j = np.clip(j, i + 1, sizes[group] - 1)
    return member[starts[group] + i], member[starts[group] + j]


def _boundary_crossings(uv: np.ndarray, loops: Sequence[np.ndarray]) -> int:
    """Count proper self-intersections of a chart's boundary polylines.

    A uniform grid keyed on the median segment length supplies the candidate
    pairs, so the cost is linear in the number of segments times the local
    density rather than quadratic.  Segments sharing an endpoint are excluded,
    and exactly-touching counts as *not* crossing, matching the evaluator's
    overlap convention (ORGANIC.md section 0.3).

    Returns ``-1`` when the test could not be run within its insert budget, so
    the caller can report "not certified" rather than "clean".
    """
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for loop in loops:
        if loop.shape[0] < 3:
            continue
        starts.append(loop)
        ends.append(np.roll(loop, -1))
    if not starts:
        return 0
    p0 = uv[np.concatenate(starts)]
    p1 = uv[np.concatenate(ends)]
    ident0 = np.concatenate(starts)
    ident1 = np.concatenate(ends)
    n = p0.shape[0]
    if n < 4:
        return 0

    lo = np.minimum(p0, p1)
    hi = np.maximum(p0, p1)
    span = float(max((hi - lo).max(), 0.0))
    extent = float(max(hi.max() - lo.min(), 0.0))
    if extent <= 0.0:
        return 0
    cell = float(np.median((hi - lo).max(axis=1)))
    if not (cell > 0.0):
        cell = extent / 64.0
    cell = max(cell, extent / _CROSS_MAX_GRID_DIM, span * 1e-12, 1e-15)

    origin = lo.min(axis=0)
    for _ in range(8):
        i0 = np.floor((lo[:, 0] - origin[0]) / cell).astype(np.int64)
        i1 = np.floor((hi[:, 0] - origin[0]) / cell).astype(np.int64)
        j0 = np.floor((lo[:, 1] - origin[1]) / cell).astype(np.int64)
        j1 = np.floor((hi[:, 1] - origin[1]) / cell).astype(np.int64)
        counts = (i1 - i0 + 1) * (j1 - j0 + 1)
        total = int(counts.sum())
        if total <= _CROSS_MAX_INSERTS:
            break
        cell *= 2.0
    else:
        return -1
    if total <= 1:
        return 0

    grid_w = int(i1.max()) + 2
    seg_of = np.repeat(np.arange(n, dtype=np.int64), counts)
    offset = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]]), counts
    )
    width = np.repeat(i1 - i0 + 1, counts)
    cell_id = (np.repeat(j0, counts) + offset // width) * grid_w + (
        np.repeat(i0, counts) + offset % width
    )

    order = np.argsort(cell_id, kind="stable")
    cell_id = cell_id[order]
    seg_of = seg_of[order]
    a, b = _bucket_pairs(cell_id, seg_of)
    if a is None:
        return -1
    if a.size == 0:
        return 0
    packed = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    a = packed // n
    b = packed - a * n

    share = (
        (ident0[a] == ident0[b])
        | (ident0[a] == ident1[b])
        | (ident1[a] == ident0[b])
        | (ident1[a] == ident1[b])
    )
    keep = ~share
    if not keep.any():
        return 0
    a = a[keep]
    b = b[keep]

    ax, ay = p0[a, 0], p0[a, 1]
    bx, by = p1[a, 0], p1[a, 1]
    cx, cy = p0[b, 0], p0[b, 1]
    dx, dy = p1[b, 0], p1[b, 1]
    d1 = _orient2d(ax, ay, bx, by, cx, cy)
    d2 = _orient2d(ax, ay, bx, by, dx, dy)
    d3 = _orient2d(cx, cy, dx, dy, ax, ay)
    d4 = _orient2d(cx, cy, dx, dy, bx, by)
    proper = (d1 * d2 < 0) & (d3 * d4 < 0)
    return int(np.count_nonzero(proper))


def _fallback_certify(chart: Chart, uv: np.ndarray) -> tuple[bool, str]:
    """Certify global injectivity without :mod:`unwrap_variations.core.injectivity`.

    A locally injective piecewise-linear map of a disk whose boundary image is a
    simple closed curve is globally injective (ORGANIC.md section 0.3), so this
    checks exactly those two things: every signed UV area strictly positive, and
    no proper crossing among the boundary segments.  ``flipped_tris == 0`` alone
    provably cannot see Suzanne's defect, which is why the boundary test is here
    and not optional.

    Returns:
        ``(ok, reason)``.  ``ok`` is ``False`` both when the map is genuinely not
        injective and when the test could not be completed, because an
        uncertified merge must be refused, not waved through.
    """
    tris = np.asarray(chart.tris, dtype=np.int64)
    if tris.shape[0] == 0:
        return False, "empty chart"
    q = uv[tris]
    e1 = q[:, 1] - q[:, 0]
    e2 = q[:, 2] - q[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if not np.isfinite(signed).all():
        return False, "non-finite UV"
    scale = float(np.abs(signed).mean())
    if not (scale > 0.0):
        return False, "degenerate flattening"
    if int(np.count_nonzero(signed <= 1e-12 * scale)):
        return False, "flipped or degenerate triangles"

    loops = _boundary_loops(tris, chart.n_verts)
    if not loops:
        return False, "no usable boundary loop"
    crossings = _boundary_crossings(uv, loops)
    if crossings < 0:
        return False, "boundary test exceeded its budget"
    if crossings:
        return False, "boundary self-intersects (%d crossings)" % crossings

    # Zero *transversal* crossings is not simplicity, and the gap is not
    # academic.  A degree-2 branched cover -- a fan whose rim traces the same
    # convex polygon twice, the second wrap reusing the first's coordinates
    # bit-for-bit -- has every triangle positively oriented, ``chi == 1``, one
    # boundary cycle, no non-manifold edge and no repeated *welded* vertex, so
    # every test above passes.  The two wraps are exactly collinear rather than
    # crossing, so the count is 0, and they enclose no extra area either, so
    # ``overlap_area`` cannot object.  Measured on the 12-triangle case: the map
    # is exactly two-to-one, 15100 of 15100 sampled interior points covered
    # twice, and this function returned ``True``.
    #
    # A simple closed curve cannot pass through a plane point twice.  Same rule
    # and same exact-equality convention as
    # :func:`unwrap_variations.core.injectivity._loops_revisit_a_point`, so the
    # fallback and the strict path agree by construction rather than by luck.
    for loop in loops:
        idx = np.asarray(loop, dtype=np.int64).reshape(-1)
        if idx.shape[0] < 2 or idx.min() < 0 or idx.max() >= uv.shape[0]:
            continue
        points = np.ascontiguousarray(uv[idx]).view(np.complex128).reshape(-1)
        if np.unique(points).shape[0] != idx.shape[0]:
            return False, "boundary revisits a UV point, so it is not simple"
    return True, ""


def _certify_chart(
    chart: Chart, uv: np.ndarray, certify: Callable[..., Any] | None
) -> tuple[bool, str]:
    """Certify a flattened chart, preferring ``core.injectivity`` when present.

    A certifier that **raises** has not certified anything, so its exception is a
    refusal rather than an invitation to try a weaker test.  Downgrading on an
    exception meant any crash in the strict path silently re-opened every hole
    the strict path exists to close -- and the local fallback is a genuinely
    different implementation, not a subset.  Measured over 279 certify calls on
    the whole corpus at seven ``max_stretch`` values: **zero raised and zero
    returned no verdict**, so refusing costs nothing observable and buys the
    guarantee that no merge is ever accepted on a certificate nobody produced.

    A stub with no ``globally_injective`` attribute is a different case -- it is
    an injected test double, not a failure -- and still gets the local test.
    """
    if certify is None:
        return _fallback_certify(chart, uv)
    try:
        report = certify(_mini_sample(chart, uv, name=chart.name))
    except BaseException as exc:  # noqa: BLE001 - a crash is a refusal
        return False, "certifier raised %s: %s" % (type(exc).__name__, exc)
    verdict = getattr(report, "globally_injective", None)
    if verdict is None:
        return _fallback_certify(chart, uv)
    if bool(verdict):
        return True, ""
    return False, "core.injectivity rejected the flattening"


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeStep:
    """One accepted merge, recorded for the report and for the tests."""

    kept: int
    absorbed: int
    members: tuple[int, ...]
    shared_length: float
    stretch_before: float
    stretch_after: float
    benefit: float
    seam_length_after: float

    def describe(self) -> str:
        """Return a one-line human description of this merge."""
        return "%s -> %d  seam -%.3f  stretch %.4f -> %.4f  seam_left %.3f" % (
            "+".join(str(m) for m in self.members),
            self.kept,
            self.shared_length,
            self.stretch_before,
            self.stretch_after,
            self.seam_length_after,
        )


@dataclass
class MergeReport:
    """What :func:`merge_charts` did, and what it refused to do.

    Attributes:
        steps: Accepted merges, in the order they were taken.
        chart_count_before: Chart count on entry.
        chart_count_after: Chart count on exit.
        seam_length_before: Cut length between charts on entry.
        seam_length_after: Cut length between charts on exit.  Provably no
            larger than ``seam_length_before``.
        max_stretch_seen: Largest ``stretch_l2`` of any accepted merged chart.
        uvs: ``(L, 2)`` layout implied by the accepted merges, or ``None`` when
            nothing was merged or no input layout was supplied.  Charts that
            were not merged keep their incoming UVs bit-identical.
        rejected_not_disk: Candidates refused because the union is not a disk.
        rejected_stretch: Candidates refused for exceeding ``max_stretch``.
        rejected_injectivity: Candidates refused by certification.
        rejected_flatten: Candidates the flattener could not solve.
        rejected_gain: Adjacent pairs never entered on the queue because their
            shared boundary did not exceed ``min_gain``.
        out_of_time: Candidate pairs left unscreened because ``time_budget``
            ran out.  Non-zero means the merge is a prefix of what an unbudgeted
            run would have found, never that it took a shortcut on a gate.
        flatten_calls: Number of flattener invocations.
        seconds: Wall-clock time.
        changed: Whether anything was merged at all.
        failed: Whether the zero merges are a *crash* rather than a decision.
            Merging never raises, so without this the two are indistinguishable
            downstream: both arrive as ``changed=False`` and get reported as
            "nothing was worth merging".  Observed live -- one Blender 4.5.4 run
            in five came back with ``chart_count_before=0`` and all-zero labels
            from the :func:`chart_graph` exception path, and the only trace was
            a note claiming no merge was admissible.  ORGANIC.md 0.7 names the
            same failure mode in another guise: ``bpy.ops.uv.unwrap`` returns
            ``{'FINISHED'}`` when it fails.
        message: Human-readable summary, ASCII only.
    """

    steps: list[MergeStep] = field(default_factory=list)
    chart_count_before: int = 0
    chart_count_after: int = 0
    seam_length_before: float = 0.0
    seam_length_after: float = 0.0
    max_stretch_seen: float = 1.0
    uvs: np.ndarray | None = None
    rejected_not_disk: int = 0
    rejected_stretch: int = 0
    rejected_injectivity: int = 0
    rejected_flatten: int = 0
    rejected_gain: int = 0
    out_of_time: int = 0
    flatten_calls: int = 0
    seconds: float = 0.0
    changed: bool = False
    failed: bool = False
    message: str = ""

    def summary(self) -> str:
        """Return a compact one-line description, for logs and reports.

        A crashed run appends its ``message``, because a summary that cannot
        say "I broke" is the status flag ORGANIC.md 0.7 warns about.
        """
        head = (
            "merges=%d charts %d->%d seam %.3f->%.3f stretch<=%.4f "
            "rejected(disk=%d stretch=%d inj=%d flat=%d) unscreened=%d "
            "flattens=%d %.2fs"
            % (
                len(self.steps),
                self.chart_count_before,
                self.chart_count_after,
                self.seam_length_before,
                self.seam_length_after,
                self.max_stretch_seen,
                self.rejected_not_disk,
                self.rejected_stretch,
                self.rejected_injectivity,
                self.rejected_flatten,
                self.out_of_time,
                self.flatten_calls,
                self.seconds,
            )
        )
        return head + (" FAILED: %s" % self.message if self.failed else "")


class _Candidate:
    """One evaluated merge candidate, cached by its member set.

    ``code`` is the machine-readable rejection reason -- ``""`` when accepted,
    else one of ``disk``, ``flatten``, ``stretch`` or ``injective``.  The report
    counts by code rather than by parsing ``reason``, which is for humans.
    """

    __slots__ = ("ok", "code", "reason", "stretch", "uv", "chart")

    def __init__(self, ok: bool, code: str, reason: str, stretch: float, uv, chart) -> None:
        self.ok = ok
        self.code = code
        self.reason = reason
        self.stretch = stretch
        self.uv = uv
        self.chart = chart


def _clamped_iters(iters: int | None, chart: "Chart") -> int | None:
    """Return a screening budget that can never exceed the full-quality one.

    The gate's whole safety argument is that screening cheaply can only ever
    *reject* more than screening at full quality would, so a cheap gate cannot
    admit a merge the expensive one would refuse.  That argument silently
    inverts above about a thousand triangles.

    ``flatten_chart``'s own budget is per-chart -- ``min(max(32000 // tris, 16),
    256)`` -- so it *falls* as the chart grows, while the screening budget is a
    flat constant.  Past the crossover the screen is the **finer** flatten,
    reports the **lower** stretch, and admits merges full quality refuses.
    Measured on the corpus sphere at ``max_stretch=1.060497``: screening at 32
    measured 1.0515 and admitted the merge, full-quality screening measured
    1.0695 and refused it.

    The descent also minimises symmetric Dirichlet while the gate reads Sander
    L2, and stretch is *not* monotone in the iteration budget even though energy
    is -- 18 of 35 corpus pair-unions have a budget at which stretch rises.  So
    "more iterations is always a tighter gate" was never true either; clamping
    is what restores the one-sided guarantee the docstring claims.
    """
    if iters is None:
        return None
    try:
        from .flatten import _iteration_budget  # noqa: PLC0415 - optional sibling
    except BaseException:  # noqa: BLE001 - flatten may be absent or renamed
        return iters
    try:
        return int(min(int(iters), int(_iteration_budget(chart))))
    except (TypeError, ValueError, AttributeError):
        return iters


def _measure_union(
    sample: UVMeshSample,
    tri_mask: np.ndarray,
    flatten: Callable[..., Any],
    certify: Callable[..., Any] | None,
    max_stretch: float,
    name: str,
    *,
    require_injective: bool = True,
    iters: int | None = None,
) -> _Candidate:
    """Flatten one candidate chart and decide whether it is acceptable.

    The order of the gates is the order of their cost: the exact disk test is a
    handful of bincounts, the flattening is a linear solve, and certification
    walks the boundary.  Nothing is estimated -- ORGANIC.md section 5 is explicit
    that predicting a union's distortion from its two parents is the shortcut
    that produces charts nobody would accept.

    Args:
        sample: The mesh.
        tri_mask: Triangles forming the candidate chart.
        flatten: The flattener.
        certify: Injectivity certifier, or ``None`` for the local fallback.
        max_stretch: Upper bound on the measured Sander L2 stretch.
        name: Diagnostic label.
        require_injective: When ``False`` the certification step is skipped.
            Used for the per-chart baseline flattenings, which are only there to
            supply a denominator and are never pasted into a layout.
        iters: Descent budget handed to the flattener.  See
            :data:`_SCREEN_FRACTION` for what screening with a small one does and
            does not guarantee -- it does *not* bound the full-quality figure.
    """
    if not is_disk(sample, tri_mask):
        return _Candidate(False, "disk", "not a disk", math.inf, None, None)

    chart = _build_chart(sample, tri_mask, name)
    uv, verdict = _flatten_uv(flatten, chart, _clamped_iters(iters, chart))
    if uv is None:
        return _Candidate(False, "flatten", "flatten failed", math.inf, None, chart)
    if require_injective and verdict is False:
        return _Candidate(
            False, "injective", "the flattener could not certify its own result",
            math.inf, None, chart,
        )
    uv = _orient(uv, np.asarray(chart.tris, dtype=np.int64))

    metrics = evaluate(_mini_sample(chart, uv, name=name), compute_overlap=False)
    stretch = float(metrics.stretch_l2)
    if not math.isfinite(stretch):
        return _Candidate(False, "flatten", "unmeasurable stretch", math.inf, None, chart)
    if metrics.degenerate_tris or metrics.flipped_tris:
        return _Candidate(
            False, "injective", "flipped or degenerate triangles", stretch, None, chart
        )
    if stretch > max_stretch:
        return _Candidate(
            False,
            "stretch",
            "stretch %.4f > %.4f" % (stretch, max_stretch),
            stretch,
            None,
            chart,
        )

    if require_injective:
        ok, why = _certify_chart(chart, uv, certify)
        if not ok:
            return _Candidate(False, "injective", why or "not injective", stretch, None, chart)
    return _Candidate(True, "", "", stretch, uv, chart)


def _paste_chart(uvs: np.ndarray, chart: Chart, uv: np.ndarray, target_area: float,
                 centre: np.ndarray) -> None:
    """Write a flattened chart into a whole-mesh UV layer, in place.

    The chart is scaled so that its UV area matches the area its parents already
    occupied and translated onto their shared centroid, which keeps texel density
    and rough placement intact; the repack in
    :mod:`unwrap_variations.backends.blender_native` then lays the whole layout
    out properly.
    """
    tris = np.asarray(chart.tris, dtype=np.int64)
    if tris.shape[0] == 0:
        return
    q = uv[tris]
    e1 = q[:, 1] - q[:, 0]
    e2 = q[:, 2] - q[:, 0]
    area = float(np.abs(0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])).sum())
    scale = math.sqrt(target_area / area) if area > 0.0 and target_area > 0.0 else 1.0
    placed = (uv - uv.mean(axis=0)) * scale + centre
    loops = np.asarray(chart.loops, dtype=np.int64).reshape(-1)
    good = (loops >= 0) & (loops < uvs.shape[0])
    uvs[loops[good]] = placed[tris.reshape(-1)[good]]


def merge_charts(
    sample: UVMeshSample,
    tri_chart: np.ndarray,
    *,
    max_stretch: float = 1.25,
    min_gain: float = 0.0,
    flatten: Callable[..., Any] | None = None,
    certify: Callable[..., Any] | None = None,
    uvs: np.ndarray | None = None,
    topo: "seam_mod.EdgeTopology | None" = None,
    max_merges: int | None = None,
    screen_iters: int | None = -1,
    time_budget: float = 0.0,
    max_screens: int | None = None,
) -> tuple[np.ndarray, MergeReport]:
    """Greedily merge adjacent charts while each merge stays a disk and in bounds.

    Charts are merged in descending order of the **benefit ratio**
    ``shared_boundary_length / max(added_distortion, eps)``, recomputed after
    every acceptance because a merge changes its neighbours' ratios.  The queue
    is a lazy-greedy heap: a pair is first entered at its *upper bound*
    (``shared_length / eps``, which no true ratio can exceed because added
    distortion is never negative in the denominator), and is only flattened when
    it reaches the top.  Entries whose endpoints have since changed are dropped
    on sight, so no full re-sort ever happens and most pairs are never flattened
    at all.

    Every surviving candidate is validated by **actually flattening the merged
    chart** and measuring it through the frozen evaluator.  A candidate is
    refused when the union is not a disk, when its measured ``stretch_l2``
    exceeds ``max_stretch``, or when the flattened union fails injectivity
    certification.

    The layout's cut length is **monotonically non-increasing**: a merge deletes
    exactly the mesh edges between its two charts and creates none.

    Args:
        sample: The mesh.  Nothing is mutated.
        tri_chart: ``(T,)`` chart id per triangle -- typically
            :func:`tri_islands` of the layout being refined.
        max_stretch: Upper bound on a merged chart's Sander L2 stretch.  Values
            below 1.0 disable merging outright, because ``stretch_l2 >= 1.0`` is
            a theorem; that is how the unmerged layout stays reachable from the
            parameter grid.
        min_gain: Shared boundary length a merge must strictly exceed to be
            worth taking, in world units.
        flatten: The flattener, as ``flatten(chart) -> uv`` or a result object
            carrying ``uv``.  Defaults to
            :func:`unwrap_variations.core.flatten.flatten_chart` when importable;
            with no flattener available nothing is merged.
        certify: Injectivity certifier, defaulting to
            :func:`unwrap_variations.core.injectivity.certify` when importable
            and to a local boundary-simplicity test otherwise.
        uvs: Optional ``(L, 2)`` incoming layout.  When given, the report carries
            a rebuilt layout in which merged charts hold their flattened UVs and
            every other chart is bit-identical.
        topo: Optional prebuilt edge topology.
        max_merges: Optional cap on the number of merges, for tests and for
            bounding the cost on a very large mesh.
        screen_iters: Descent budget for *screening* candidates, handed to the
            flattener as ``iters=`` when it accepts one.  The default ``-1``
            means "an eighth of the flattener's own budget" (see
            :func:`_screen_iters`); ``None`` screens at full quality, which on
            Suzanne is 3.3x slower for one extra merge.  Accepted merges are
            re-flattened at full quality before their UVs are used, and the
            replacement is kept only when it too clears the gate, so every chart
            that ships has been measured against ``max_stretch`` and passed.
            **This value does change which merges are taken, and not only in the
            conservative direction** -- see :data:`_SCREEN_FRACTION` for the
            measured counterexample on the corpus sphere, where screening admits
            a pair that full-quality screening refuses.
        max_screens: Cap on how many candidate pairs are screened, counted in
            flattener calls.  **Prefer this to ``time_budget``.** Both bound the
            same work and both stop on a prefix of the greedy sequence, but a
            wall-clock deadline makes the result depend on machine load, so the
            same mesh and seed can produce different layouts on a busy machine.
            Measured: with ``time_budget`` alone the seven-mesh benchmark's
            Suzanne cell went from a run-to-run spread of 0.0 to **3.4** on
            Blender 4.5.4.  A screen count is deterministic and bounds the same
            thing, because screening is where essentially all the time goes.
        time_budget: Seconds after which no *new* candidate pair is screened;
            ``0`` means no limit.  Merges already screened and admissible are
            still taken, and an in-flight flatten is never interrupted, so the
            result is always a prefix of the unbudgeted greedy sequence -- the
            most valuable merges, since greedy takes them first.  Nothing about
            the gates changes: a budget can only cost seam length, never let a
            bad chart through.

    Returns:
        ``(labels, report)``.  ``labels`` is ``(T,)`` int64 dense chart ids.  Per
        ORGANIC.md rule 2 a stage that cannot improve a layout returns it
        untouched: with nothing merged, ``labels`` is the densified input and
        ``report.changed`` is ``False``.

    Notes:
        This function never raises.  A flattener that throws, a malformed chart
        labelling and a degenerate mesh all come back as a report with zero
        merges.
    """
    started = time.perf_counter()
    report = MergeReport()
    try:
        graph = chart_graph(sample, tri_chart, topo=topo)
    except Exception as exc:  # noqa: BLE001 - containment, per the contract
        report.failed = True
        report.message = "chart graph failed: %s: %s" % (type(exc).__name__, exc)
        report.seconds = time.perf_counter() - started
        n_tris = int(sample.n_tris)
        return np.zeros(n_tris, dtype=np.int64), report

    labels = graph.labels
    report.chart_count_before = graph.n_charts
    report.chart_count_after = graph.n_charts
    report.seam_length_before = graph.seam_length
    report.seam_length_after = graph.seam_length

    # NaN is the only value with no defensible reading: it is neither "off" nor
    # "unbounded", so it takes the documented default.  The infinities do have
    # one and must keep it -- a blanket ``isfinite`` guard sent ``inf`` to 1.25,
    # so "no bound" quietly became the conservative bound while ``1e300``, being
    # finite, sailed through as unbounded; and it sent ``-inf`` to 1.25 as well,
    # turning merging *on* for a caller who asked for the least permissive
    # setting expressible.
    bound = float(max_stretch)
    if math.isnan(bound):
        bound = 1.25
    if bound < _MERGE_OFF_BELOW:
        report.message = "merging disabled (max_stretch %.4f < 1.0)" % bound
        report.seconds = time.perf_counter() - started
        return labels, report

    if screen_iters is not None and int(screen_iters) < 0:
        screen_iters = _screen_iters()

    solver = _resolve_flatten(flatten)
    if solver is None:
        report.message = "no flattener available; every merge would be unvalidated"
        report.seconds = time.perf_counter() - started
        return labels, report
    certifier = _resolve_certify(certify)

    if graph.n_charts < 2 or graph.pair_a.size == 0:
        report.message = "nothing adjacent to merge"
        report.seconds = time.perf_counter() - started
        return labels, report

    try:
        merged, steps, accepted = _greedy_merge(
            sample,
            graph,
            solver,
            certifier,
            bound,
            float(min_gain),
            max_merges,
            report,
            uvs=uvs,
            screen_iters=screen_iters,
            deadline=(started + time_budget) if time_budget > 0.0 else None,
            max_screens=max_screens,
        )
    except Exception as exc:  # noqa: BLE001 - containment, per the contract
        report.failed = True
        report.message = "merge aborted: %s: %s" % (type(exc).__name__, exc)
        report.seconds = time.perf_counter() - started
        return labels, report

    report.steps = steps
    if not steps:
        report.message = (
            "time budget spent before any merge was screened"
            if report.out_of_time and not report.flatten_calls
            else "no merge cleared the disk, stretch and injectivity gates"
        )
        report.seconds = time.perf_counter() - started
        return labels, report

    out, _ = _dense_labels(merged)
    report.changed = True
    report.chart_count_after = int(np.unique(out).size)
    report.message = "merged %d chart pairs; seam %.3f -> %.3f" % (
        len(steps),
        report.seam_length_before,
        report.seam_length_after,
    )
    if uvs is not None:
        report.uvs = _rebuild_uvs(sample, uvs, accepted)
    report.seconds = time.perf_counter() - started
    return out, report


def _greedy_merge(
    sample: UVMeshSample,
    graph: ChartGraph,
    solver: Callable[..., Any],
    certifier: Callable[..., Any] | None,
    max_stretch: float,
    min_gain: float,
    max_merges: int | None,
    report: MergeReport,
    *,
    uvs: np.ndarray | None = None,
    screen_iters: int | None = None,
    deadline: float | None = None,
    max_screens: int | None = None,
) -> tuple[np.ndarray, list[MergeStep], dict[int, _Candidate]]:
    """Run the lazy-greedy merge loop.  See :func:`merge_charts` for the contract."""
    k = graph.n_charts
    labels = graph.labels.copy()
    members: dict[int, tuple[int, ...]] = {c: (c,) for c in range(k)}
    tri_of: dict[int, np.ndarray] = {
        c: np.flatnonzero(labels == c) for c in range(k)
    }
    layout = (
        _LayoutStretch(sample, uvs, labels, k)
        if uvs is not None and np.asarray(uvs).size
        else None
    )
    shared: dict[int, dict[int, float]] = {c: {} for c in range(k)}
    for a, b, length in zip(
        graph.pair_a.tolist(), graph.pair_b.tolist(), graph.pair_length.tolist(), strict=False
    ):
        shared[a][b] = shared[a].get(b, 0.0) + float(length)
        shared[b][a] = shared[b].get(a, 0.0) + float(length)

    alive = np.ones(k, dtype=bool)
    version = np.zeros(k, dtype=np.int64)
    cache: dict[Any, _Candidate] = {}
    accepted: dict[int, _Candidate] = {}

    def solo(chart: int) -> _Candidate:
        """Flatten one chart on its own; a fallback baseline when no layout is given."""
        key = members[chart]
        hit = cache.get(key)
        if hit is None:
            mask = np.zeros(sample.n_tris, dtype=bool)
            mask[tri_of[chart]] = True
            report.flatten_calls += 1
            hit = _measure_union(
                sample,
                mask,
                solver,
                certifier,
                math.inf,
                "chart%d" % chart,
                require_injective=False,
                iters=screen_iters,
            )
            cache[key] = hit
        return hit

    def baseline(a: int, b: int) -> float:
        """L2 stretch the two parents already carry, as the ratio's denominator.

        Read straight off the incoming layout when there is one -- exact, free,
        and the honest answer to "what does merging cost the layout we have".
        Without a layout there is nothing to read, so each parent is flattened
        once and the two are combined in quadrature, which is how an
        area-weighted RMS composes.
        """
        if layout is not None:
            return layout.of(members[a] + members[b])
        ca, cb = solo(a), solo(b)
        sa = ca.stretch if math.isfinite(ca.stretch) else 1.0
        sb = cb.stretch if math.isfinite(cb.stretch) else 1.0
        wa = float(graph.area[list(members[a])].sum())
        wb = float(graph.area[list(members[b])].sum())
        total = wa + wb
        if total <= 0.0:
            return max(sa, sb)
        return math.sqrt((wa * sa * sa + wb * sb * sb) / total)

    def measure(a: int, b: int, iters: int | None) -> _Candidate:
        """Flatten the union of two charts at a given budget, cached."""
        key = (tuple(sorted(members[a] + members[b])), iters)
        hit = cache.get(key)
        if hit is None:
            mask = np.zeros(sample.n_tris, dtype=bool)
            mask[tri_of[a]] = True
            mask[tri_of[b]] = True
            report.flatten_calls += 1
            hit = _measure_union(
                sample,
                mask,
                solver,
                certifier,
                max_stretch,
                "merge%d_%d" % (a, b),
                iters=iters,
            )
            cache[key] = hit
        return hit

    def evaluate_pair(a: int, b: int) -> tuple[_Candidate, float]:
        """Screen the union of two charts and return it with its benefit ratio."""
        hit = measure(a, b, screen_iters)
        if not hit.ok:
            return hit, -math.inf
        added = max(hit.stretch - baseline(a, b), 0.0)
        return hit, shared[a].get(b, 0.0) / max(added, _DISTORTION_EPS)

    def finalise(chart: int, screened: _Candidate) -> _Candidate:
        """Re-flatten a *surviving* merged chart at full quality.

        Done once per chart at the end of the run rather than once per accepted
        merge, because a chart that merges again throws its predecessor's
        polish away: on a 121-chart Suzanne candidate that was 102 full-quality
        solves of which 6 survived, and at ~250 ms each it was most of the
        stage's cost.

        More descent can only lower the energy, so the full result is normally
        both less distorted and still certified.  If it is not -- or if the time
        budget has run out -- the screened layout is kept: it already passed
        every gate, and a certified more-distorted chart beats an uncertified
        less-distorted one.
        """
        if screen_iters is None:
            return screened
        if deadline is not None and time.perf_counter() >= deadline:
            return screened
        if max_screens is not None and report.flatten_calls >= max_screens:
            return screened
        key = (members[chart], None)
        final = cache.get(key)
        if final is None:
            mask = np.zeros(sample.n_tris, dtype=bool)
            mask[tri_of[chart]] = True
            report.flatten_calls += 1
            final = _measure_union(
                sample, mask, solver, certifier, max_stretch, "final%d" % chart
            )
            cache[key] = final
        if final.ok and final.uv is not None:
            return final
        return screened

    # Two heaps, not one, and this is the difference between the lazy queue
    # working and being decorative.
    #
    # ``pending`` holds pairs nobody has flattened yet, keyed by shared boundary
    # length; ``ready`` holds screened, admissible pairs keyed by their measured
    # benefit ratio.  A single heap has to enter an unflattened pair at *some*
    # numeric priority, and the only safe one -- ``shared / eps`` -- is about
    # 1e5 times any real ratio, so every unflattened pair outranks every
    # measured one and the queue degenerates into "flatten everything, then sort".
    # Measured on a 121-chart Suzanne candidate that was 703 flattens and 16.6 s.
    #
    # Split in two, the comparison is made at pop time against
    # ``shared_top / best_added``, where ``best_added`` is the smallest added
    # distortion any admissible merge has actually shown.  That divisor is a
    # running estimate, not a proof, so the *ordering* is a heuristic -- which is
    # all it ever was, and is a different thing from the gates: every merge that
    # is accepted has still been flattened and measured in full.
    pending: list[tuple[float, int, int, int, int]] = []
    ready: list[tuple[float, int, int, int, int]] = []
    best_added = math.inf

    def stamp(a: int, b: int) -> tuple[int, int, int, int]:
        """Return the pair, ordered, with both endpoints' current versions."""
        lo, hi = (a, b) if a < b else (b, a)
        return lo, hi, int(version[lo]), int(version[hi])

    def live(entry) -> bool:
        """Return whether a heap entry still describes two current charts."""
        _, a, b, va, vb = entry
        return (
            alive[a] and alive[b]
            and int(version[a]) == va and int(version[b]) == vb
            and shared[a].get(b, 0.0) > min_gain
        )

    def peel(heap) -> None:
        """Drop stale entries from the top of a heap."""
        while heap and not live(heap[0]):
            heapq.heappop(heap)

    for a in range(k):
        for b, length in shared[a].items():
            if b <= a:
                continue
            if length > min_gain:
                heapq.heappush(pending, (-length, *stamp(a, b)))
            else:
                report.rejected_gain += 1

    steps: list[MergeStep] = []
    seam_left = graph.seam_length
    pops = 0
    while pops < _MAX_QUEUE_POPS:
        pops += 1
        peel(pending)
        peel(ready)
        if not pending and not ready:
            break

        take_ready = bool(ready)
        if take_ready and pending:
            bound = -pending[0][0] / max(best_added, _DISTORTION_EPS)
            take_ready = -ready[0][0] >= bound

        if not take_ready:
            spent = (deadline is not None and time.perf_counter() >= deadline) or (
                max_screens is not None and report.flatten_calls >= max_screens
            )
            if spent:
                # Out of budget for *new* screening.  Pairs already screened
                # stay on ``ready`` and can still be accepted, so the run ends
                # on a prefix of the greedy sequence rather than on a
                # half-merge.
                report.out_of_time += len(pending)
                pending.clear()
                continue
            _, a, b, _, _ = heapq.heappop(pending)
            candidate, ratio = evaluate_pair(a, b)
            if not candidate.ok:
                counter = {
                    "disk": "rejected_not_disk",
                    "stretch": "rejected_stretch",
                    "flatten": "rejected_flatten",
                    "injective": "rejected_injectivity",
                }.get(candidate.code, "rejected_injectivity")
                setattr(report, counter, getattr(report, counter) + 1)
                continue
            added = shared[a].get(b, 0.0) / ratio if ratio > 0.0 else math.inf
            if added > 0.0:
                best_added = min(best_added, added)
            heapq.heappush(ready, (-ratio, *stamp(a, b)))
            continue

        neg_ratio, a, b, _, _ = heapq.heappop(ready)
        length = shared[a].get(b, 0.0)
        candidate = cache[(tuple(sorted(members[a] + members[b])), screen_iters)]
        if not candidate.ok:  # pragma: no cover - defensive
            continue

        before = baseline(a, b)
        seam_left -= length
        if seam_left < -1e-9:  # pragma: no cover - defensive
            seam_left = 0.0
        steps.append(
            MergeStep(
                kept=a,
                absorbed=b,
                members=tuple(sorted(members[a] + members[b])),
                shared_length=float(length),
                stretch_before=float(before),
                stretch_after=float(candidate.stretch),
                benefit=float(-neg_ratio),
                seam_length_after=float(max(seam_left, 0.0)),
            )
        )
        report.max_stretch_seen = max(report.max_stretch_seen, float(candidate.stretch))
        report.seam_length_after = float(max(seam_left, 0.0))
        accepted[a] = candidate

        # Absorb b into a.
        members[a] = tuple(sorted(members[a] + members[b]))
        tri_of[a] = np.concatenate([tri_of[a], tri_of[b]])
        labels[tri_of[b]] = a
        alive[b] = False
        accepted.pop(b, None)
        del shared[a][b]
        del shared[b][a]
        for other, extra in list(shared[b].items()):
            shared[other].pop(b, None)
            if other == a:
                continue
            shared[a][other] = shared[a].get(other, 0.0) + extra
            shared[other][a] = shared[other].get(a, 0.0) + extra
        shared[b] = {}
        version[a] += 1
        version[b] += 1

        for other, extra in shared[a].items():
            if alive[other] and extra > min_gain:
                heapq.heappush(pending, (-extra, *stamp(a, other)))

        if max_merges is not None and len(steps) >= int(max_merges):
            break

    # Polish, once per surviving chart.  See finalise() for why this is here and
    # not in the accept path.  ``max_stretch_seen`` is then restated over the
    # layouts that actually ship: the per-step figures stay as the screening
    # measurements the gate ruled on, which are upper bounds on these.
    for chart in list(accepted):
        accepted[chart] = finalise(chart, accepted[chart])
    if accepted:
        report.max_stretch_seen = max(
            1.0,
            max(
                float(c.stretch)
                for c in accepted.values()
                if math.isfinite(c.stretch)
            ),
        )

    return labels, steps, accepted


def _rebuild_uvs(
    sample: UVMeshSample,
    uvs: np.ndarray,
    accepted: dict[int, _Candidate],
) -> np.ndarray | None:
    """Paste every merged chart's flattened UVs into a copy of the input layout.

    Charts that were not merged are left **bit-identical**, which is what makes
    the stage safe to run unconditionally: the only thing that can change is the
    part that was measured and accepted.  Each merged chart is scaled to the UV
    area its parents already held and centred on their area-weighted centroid, so
    texel density survives and the repack has something sane to arrange.
    """
    if not accepted:
        return None
    base = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if base.shape[0] < sample.n_loops:
        return None
    out = base.copy()

    for candidate in accepted.values():
        if candidate.uv is None or candidate.chart is None:
            continue
        loops = np.asarray(candidate.chart.loops, dtype=np.int64).reshape(-1, 3)
        if loops.shape[0] == 0 or int(loops.max()) >= base.shape[0] or int(loops.min()) < 0:
            continue
        # Read the parents' footprint straight off the chart's own corners.
        # Reverse-mapping loops back to triangle indices would be wrong here:
        # ``tri_loops`` comes from Blender's ``loop_triangles``, so the two
        # triangles of a quad *share* two of their loops, and a loop-to-triangle
        # table silently keeps only one of them -- on a quad mesh like Suzanne
        # that is half the chart's area.
        q = base[loops]
        e1 = q[:, 1] - q[:, 0]
        e2 = q[:, 2] - q[:, 0]
        weight = np.abs(0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]))
        total = float(weight.sum())
        if total > 0.0:
            centre = (q.mean(axis=1) * weight[:, None]).sum(axis=0) / total
        else:
            centre = q.reshape(-1, 2).mean(axis=0)
        _paste_chart(out, candidate.chart, candidate.uv, total, centre)
    return out
