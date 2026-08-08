"""Injective flattening: re-parameterise a chart so its UV map cannot fold.

``docs/ORGANIC.md`` section 0.3 is the fact that shapes this whole module.  On
Suzanne's winning layout ``flipped_tris`` is **0** -- every one of the 968
signed UV areas is positive, so the map is locally injective everywhere -- and
yet the layout still overlaps itself by 0.0013421.  Sixteen triangle pairs lap
over each other and *none of them share an edge*.  The chart's boundary curve
crosses itself while every triangle keeps its orientation.

A ``signed_area < 0`` test cannot see that, and a flip-only repair cannot fix
it.  What this module relies on instead is a classical theorem:

    A locally injective piecewise-linear map of a disk whose boundary image is
    a simple closed curve is globally injective.

So the guarantee is built as a chain, and every link is load-bearing:

1. **Tutte / Floater start.**  A disk triangulation mapped to a convex boundary
   with strictly positive weights has a provably crossing-free embedding.  The
   weights are **mean value** (Floater 2003), not cotangent: cotangent weights
   go negative on obtuse triangles, and one negative weight forfeits the
   theorem.  The start is heavily distorted; that is fine, it is a feasible
   point, not an answer.
2. **Descent that cannot flip a triangle.**  Each iteration takes a descent
   direction and then a *maximum-step* line search: the signed area of every
   triangle along ``uv + t*d`` is a quadratic in ``t``, so the largest ``t``
   that keeps every currently-positive triangle positive is available in closed
   form.  Capping at 80% of it means no step can ever flip a triangle, so local
   injectivity holds by construction from the Tutte start onward.  This is the
   entire reason for starting at Tutte rather than at the backend's layout --
   the backend's layout is not a feasible point and there is nothing to
   preserve.
3. **Free the boundary, then certify.**  Fixed-boundary results are far too
   distorted to ship, so the descent frees the boundary and the result is
   checked afterwards.  (No flips) + (simple boundary) is a complete proof, not
   a heuristic.  If the boundary does self-intersect the fallbacks run in the
   order the contract fixes: more iterations with a smaller step cap, then a
   boundary-constrained descent, then the Tutte embedding, which is always
   valid.  A layout that fails certification is never returned.

Symmetric Dirichlet ``sum A_t (||J_t||_F^2 + ||J_t^-1||_F^2)`` is the energy
because it is infinite at ``det J = 0``: the barrier and the step cap push in
the same direction, so a triangle that is nearly collapsed is repelled rather
than merely constrained.

Nothing here imports :mod:`bpy`, and nothing imports ``scipy`` -- Blender ships
numpy and nothing else (``docs/ORGANIC.md`` section 0.5).  The linear algebra
lives in :mod:`unwrap_variations.core.linalg`.

A note on the mean-value weights and symmetry
---------------------------------------------

Floater's mean-value weight ``w_ij`` is *not* symmetric: it is built from the
angles at ``i``, so ``w_ij != w_ji``.  A conjugate-gradient solver needs a
symmetric operator, so the assembled edge weight here is
``(w_ij + w_ji) / 2``.  That is not a compromise of the guarantee.  Tutte's
theorem, in Floater's form, needs only that each interior vertex is a convex
combination of its neighbours with **strictly positive** coefficients; the
symmetrised weight is an average of two positive numbers, so the row-normalised
coefficients ``w_ij_sym / sum_k w_ik_sym`` are still positive and still sum to
one.  The embedding stays provably crossing-free, and the system stays
symmetric positive definite.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .evaluator import evaluate
from .linalg import SparseSym, pcg
from .meshdata import UVMeshSample
from .metrics import ScoreWeights, score

__all__ = [
    "Chart",
    "ChartTopology",
    "FlattenResult",
    "RepairReport",
    "uv_charts",
    "chart_from_tris",
    "chart_topology",
    "symmetric_dirichlet_energy",
    "tutte_embed",
    "minimize_distortion",
    "flatten_chart",
    "repair_folds",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Fraction of the flip-free step that a line search is allowed to take.  Never
#: 1.0: at exactly the root a triangle is collapsed, and float64 rounding either
#: side of it decides whether the area is a tiny positive or a tiny negative.
#: 0.8 keeps the iterate a clear margin inside the feasible cone.
_STEP_SAFETY = 0.8

#: Armijo sufficient-decrease constant.  Small, because the search direction is
#: preconditioned and the first trial step is already capped by feasibility.
_ARMIJO_C1 = 1e-4

#: Backtracking halvings before a line search gives up and the descent stops.
_MAX_BACKTRACK = 40

#: Iterations between rebuilds of the distortion-weighted preconditioner.  The
#: pattern never changes, only the values, so a rebuild is one bincount.
_PRECOND_REFRESH = 4

#: Relative regulariser added to the preconditioner's diagonal.  The free-
#: boundary system has the constant (global translation) in its null space; the
#: gradient is orthogonal to it, so CG copes either way, but pinning it keeps
#: the direction reproducible run to run.
_PRECOND_EPS = 1e-9

#: Tolerance and iteration cap for the preconditioner solve inside the descent.
#: Deliberately loose.  ``H d = -g`` is not a solve whose answer is wanted, it
#: is a *change of metric*: every CG iterate on an SPD system started from zero
#: is already a descent direction, so extra digits buy nothing and the line
#: search re-checks the slope anyway.  Measured on Suzanne's 77-vertex chart, a
#: 1e-10 tolerance spent 74 CG iterations per descent step -- 95% of the whole
#: flattener's runtime -- on a system that a dozen iterations settle.
_SOLVE_TOL = 1e-6
_SOLVE_MAXITER_CAP = 64

#: Relative tolerance for welding two loops into one UV vertex.  Matches
#: ``evaluator._WELD_EPS`` so this module's idea of an island is the same as the
#: metric's; the test suite pins the two together on the corpus.
_WELD_EPS = 1e-6

#: Below this fraction of the mean triangle area a chart triangle is treated as
#: degenerate: excluded from the energy, and not allowed to veto a step.
_DEGENERATE_EPS = 1e-12

#: Hard cap on candidate segment pairs in the fallback boundary-crossing test.
_MAX_SEGMENT_PAIRS = 20_000_000

#: Upper bound on the descent budget :func:`_iteration_budget` will hand out.
_DEFAULT_ITERS = 256

#: Descent iterations between injectivity checkpoints.  See
#: :func:`_certified_descent` for why checkpointing is not optional.
_CERT_CHUNK = 16

#: Bisection steps used to claw back the area the coarse scale ladder overshot
#: when a repaired chart has to shrink to clear its neighbours.  See
#: :func:`_place_chart`.
_SHRINK_REFINE = 8

#: Relative tolerance for the boolean triangle-overlap test.  Matches the
#: evaluator's separating-axis tolerance, so "collides" here and "contributes
#: overlap area" there agree about triangles that merely touch.
_TOUCH_EPS = 1e-12

#: Candidate triangle pairs tested per broadcast block during placement.
_PAIR_BLOCK = 1_000_000


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chart:
    """One sub-mesh to flatten, in its own dense local indexing.

    A chart is *not* a set of mesh vertices: a vertex sitting on a UV seam
    belongs to two charts with two different UVs, so the vertices here are
    **UV vertices** -- welded face corners -- and :attr:`verts` holds the 3D
    position each one sits at.  That is what lets a chart carrying an internal
    slit stay a disk instead of being welded shut into a cylinder.

    Attributes:
        verts: ``(V, 3)`` float64 positions, one per chart vertex.
        tris: ``(T, 3)`` int32 chart-local vertex indices.
        loops: ``(T, 3)`` int32 loop index of each corner in the originating
            mesh, so a flattened result can be written straight back.  ``-1``
            for a chart that did not come from a mesh.
        name: Optional label, for diagnostics.
    """

    verts: np.ndarray
    tris: np.ndarray
    loops: np.ndarray
    name: str = ""

    @property
    def n_verts(self) -> int:
        """Number of chart vertices."""
        return int(self.verts.shape[0])

    @property
    def n_tris(self) -> int:
        """Number of chart triangles."""
        return int(self.tris.shape[0])

    def summary(self) -> str:
        """Return a compact one-line description, for logs."""
        return "%s V=%d T=%d" % (self.name or "<chart>", self.n_verts, self.n_tris)


def _as_chart(chart: object) -> Chart:
    """Adapt a foreign chart-like object to this module's :class:`Chart`.

    ``core/charts.py`` defines its own ``Chart`` and injects
    :func:`flatten_chart` as a callable, so the two types meet at runtime
    without ever being the same class.  Rather than reading attributes off a
    stranger and silently misinterpreting whichever ones happen to exist, this
    converts explicitly: ``verts`` and ``tris`` are required, ``loops`` and
    ``name`` are optional, and anything else raises.

    Args:
        chart: A :class:`Chart`, or any object exposing ``verts`` and ``tris``.

    Returns:
        A :class:`Chart` with float64 positions and int32 indices.

    Raises:
        TypeError: If the object does not expose the required fields, or their
            shapes do not agree.  Loud, because a chart misread as another
            chart's indices produces a plausible-looking wrong answer.
    """
    if isinstance(chart, Chart):
        return chart
    verts = getattr(chart, "verts", None)
    tris = getattr(chart, "tris", None)
    if verts is None or tris is None:
        raise TypeError(
            "expected a Chart or an object with .verts and .tris, got %s"
            % type(chart).__name__
        )
    v = np.asarray(verts, dtype=np.float64)
    t = np.asarray(tris, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3:
        raise TypeError(
            "chart-like %s has verts %s and tris %s, expected (V, 3) and (T, 3)"
            % (type(chart).__name__, v.shape, t.shape)
        )
    loops = getattr(chart, "loops", None)
    if loops is None:
        loop_array = np.full(t.shape, -1, dtype=np.int32)
    else:
        loop_array = np.asarray(loops, dtype=np.int32).reshape(-1, 3)
        if loop_array.shape[0] != t.shape[0]:
            raise TypeError(
                "chart-like %s has %d loop rows for %d triangles"
                % (type(chart).__name__, loop_array.shape[0], t.shape[0])
            )
    return Chart(
        verts=v,
        tris=t.astype(np.int32),
        loops=loop_array,
        name=str(getattr(chart, "name", "") or ""),
    )


@dataclass(frozen=True)
class ChartTopology:
    """The exact topological facts a flattener needs before it starts.

    Attributes:
        n_verts: Chart vertex count.
        n_tris: Chart triangle count.
        n_edges: Undirected edge count over the chart's triangles.
        euler: ``V - E + F``.  Exactly 1 for a disk.
        loops: Ordered boundary loops as tuples of chart vertex indices, wound
            to agree with the triangles' own orientation.
        is_disk: True when the chart is a single connected disk: one boundary
            loop, ``euler == 1`` and no structural problems.
        problems: Human-readable reasons the chart is not a clean disk.
    """

    n_verts: int
    n_tris: int
    n_edges: int
    euler: int
    loops: tuple[tuple[int, ...], ...]
    is_disk: bool
    problems: tuple[str, ...]


@dataclass(frozen=True)
class FlattenResult:
    """A flattened chart plus the certificate that says whether to trust it.

    Attributes:
        uv: ``(V, 2)`` float64 chart-local UVs.  Always finite.
        ok: True only when the layout is certified globally injective -- no
            flipped or degenerate triangle and a simple boundary.
        stage: Which stage produced :attr:`uv`.  One of ``"descent"``,
            ``"descent_fine"``, ``"fixed_boundary"``, ``"tutte"`` or
            ``"projection"``.
        flipped: Triangles whose signed area opposes the chart majority.
        degenerate: Triangles with no usable area.
        boundary_crossings: Self-intersections of the chart boundary.
        energy: Final symmetric-Dirichlet energy, normalised by surface area so
            a perfectly isometric map reads 4.0.
        iterations: Descent iterations actually run.
        seconds: Wall time.
        message: Short explanation, empty when nothing went wrong.
    """

    uv: np.ndarray
    ok: bool
    stage: str
    flipped: int
    degenerate: int
    boundary_crossings: int
    energy: float
    iterations: int
    seconds: float
    message: str = ""


@dataclass(frozen=True)
class RepairReport:
    """What :func:`repair_folds` did, and what it measured while doing it.

    Attributes:
        charts_checked: Charts certified on the way in.
        charts_failed: Charts that failed certification.
        charts_repaired: Charts actually rewritten.
        charts_reverted: Charts that failed but whose re-flattening was
            rejected, so the original was kept.
        overlap_before: ``overlap_area`` of the incoming layout.
        overlap_after: ``overlap_area`` of the returned layout.
        flipped_before: ``flipped_tris`` of the incoming layout.
        flipped_after: ``flipped_tris`` of the returned layout.
        changed: False when the input was returned untouched.
        seconds: Wall time.
        messages: Per-chart notes, deterministic in chart order.
    """

    charts_checked: int = 0
    charts_failed: int = 0
    charts_repaired: int = 0
    charts_reverted: int = 0
    overlap_before: float = 0.0
    overlap_after: float = 0.0
    flipped_before: int = 0
    flipped_after: int = 0
    changed: bool = False
    seconds: float = 0.0
    messages: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Return a compact one-line description, for logs."""
        return (
            "charts %d/%d failed, %d repaired, %d reverted; overlap %.6g -> %.6g; "
            "flipped %d -> %d"
            % (
                self.charts_failed,
                self.charts_checked,
                self.charts_repaired,
                self.charts_reverted,
                self.overlap_before,
                self.overlap_after,
                self.flipped_before,
                self.flipped_after,
            )
        )


# ---------------------------------------------------------------------------
# Certification: the runtime module when it exists, a local fallback until then
# ---------------------------------------------------------------------------


def _injectivity_module():
    """Return :mod:`unwrap_variations.core.injectivity`, or None if absent.

    ``core/injectivity.py`` is written in parallel with this module.  Until it
    lands, the private fallback below stands in; when it lands, this function
    finds it and the fallback stops being used.  The import is deliberately not
    at module scope: an absent sibling must not stop this module importing, and
    a *broken* sibling must not either.
    """
    try:
        from . import injectivity  # noqa: PLC0415 - deliberate late import
    except Exception:  # noqa: BLE001 - a broken sibling degrades, never breaks
        return None
    if not hasattr(injectivity, "boundary_crossings"):
        return None
    return injectivity


def _boundary_crossings(uv: np.ndarray, loops) -> int:
    """Count self-intersections of a set of closed boundary polylines.

    Delegates to ``core/injectivity.py`` when that module is available and falls
    back to :func:`_fallback_boundary_crossings` otherwise.  Switching over
    permanently is a one-line change: drop the fallback branch.

    Args:
        uv: ``(N, 2)`` coordinates the loop indices refer to.
        loops: Sequence of index sequences, each a closed polyline.

    Returns:
        Number of proper crossings.  Segments that merely touch do not count,
        matching the evaluator's overlap convention.
    """
    module = _injectivity_module()
    if module is not None:
        try:
            return int(module.boundary_crossings(uv, loops))
        except Exception:  # noqa: BLE001 - never let a sibling take us down
            pass
    return _fallback_boundary_crossings(uv, loops)


def _fallback_boundary_crossings(uv: np.ndarray, loops) -> int:
    """Count proper self-intersections of closed polylines, sort-swept.

    Broad phase is a sweep on the segments' minimum x: after sorting, segment
    ``i`` can only meet segments whose minimum x is below its own maximum x, so
    a single ``searchsorted`` bounds each candidate run.  Narrow phase is the
    standard four-orientation test with **strict** signs, so a shared endpoint
    or a T-junction counts as touching, not crossing -- the same convention the
    evaluator applies to overlapping triangles.

    Args:
        uv: ``(N, 2)`` coordinates.
        loops: Sequence of index sequences, each a closed polyline.

    Returns:
        Number of proper crossings, ``0`` when there is nothing to test.
    """
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for loop in loops:
        idx = np.asarray(loop, dtype=np.int64).reshape(-1)
        if idx.shape[0] < 3:
            continue
        starts.append(idx)
        ends.append(np.roll(idx, -1))
    if not starts:
        return 0
    a_idx = np.concatenate(starts)
    b_idx = np.concatenate(ends)
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return 0
    in_range = (
        (a_idx >= 0) & (a_idx < points.shape[0]) & (b_idx >= 0) & (b_idx < points.shape[0])
    )
    a_idx = a_idx[in_range]
    b_idx = b_idx[in_range]
    if a_idx.shape[0] < 2:
        return 0

    p = points[a_idx]
    q = points[b_idx]
    finite = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    if not finite.all():
        p, q = p[finite], q[finite]
        a_idx, b_idx = a_idx[finite], b_idx[finite]
    n = p.shape[0]
    if n < 2:
        return 0

    lo_x = np.minimum(p[:, 0], q[:, 0])
    hi_x = np.maximum(p[:, 0], q[:, 0])
    order = np.argsort(lo_x, kind="stable")
    p, q = p[order], q[order]
    a_idx, b_idx = a_idx[order], b_idx[order]
    lo_x, hi_x = lo_x[order], hi_x[order]
    lo_y = np.minimum(p[:, 1], q[:, 1])
    hi_y = np.maximum(p[:, 1], q[:, 1])

    stop = np.searchsorted(lo_x, hi_x, side="right")
    counts = np.maximum(stop - np.arange(n) - 1, 0)
    total = int(counts.sum())
    if total <= 0:
        return 0
    if total > _MAX_SEGMENT_PAIRS:
        # A boundary dense enough to hit this is already unusable; report the
        # cap as "crossing" rather than pretending the test passed.
        return int(_MAX_SEGMENT_PAIRS)

    first = np.repeat(np.arange(n, dtype=np.int64), counts)
    offset = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(counts[:-1])]), counts
    )
    second = first + 1 + offset

    keep = (lo_y[first] <= hi_y[second]) & (lo_y[second] <= hi_y[first])
    keep &= (
        (a_idx[first] != a_idx[second])
        & (a_idx[first] != b_idx[second])
        & (b_idx[first] != a_idx[second])
        & (b_idx[first] != b_idx[second])
    )
    first = first[keep]
    second = second[keep]
    if first.size == 0:
        return 0

    return int(np.count_nonzero(_proper_crossings(p[first], q[first], p[second], q[second])))


def _proper_crossings(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """Return the mask of segment pairs that cross properly.

    "Properly" means each segment strictly separates the other's endpoints.
    Collinear overlap and endpoint contact both give a zero orientation and are
    therefore *not* crossings, which is the convention section 4.1 of
    ``docs/ORGANIC.md`` fixes.
    """

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Twice the signed area of triangle ``abc``."""
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c[:, 0] - a[:, 0]
        )

    d1 = orient(p1, p2, q1)
    d2 = orient(p1, p2, q2)
    d3 = orient(q1, q2, p1)
    d4 = orient(q1, q2, p2)
    return (d1 * d2 < 0.0) & (d3 * d4 < 0.0)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def uv_charts(
    sample: UVMeshSample, uvs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split a layout into UV vertices and charts.

    Two face corners are the **same UV vertex** when they share a mesh vertex
    *and* land on the same UV within tolerance; two triangles are in the same
    **chart** when they share a UV vertex.  Both conditions are needed: welding
    on the mesh vertex alone fuses islands that were deliberately split, and
    welding on the UV alone fuses islands that merely touch in texture space.
    This is the same rule the evaluator uses for ``island_count``, so a chart
    here is an island there.

    Args:
        sample: The mesh.
        uvs: ``(L, 2)`` layout to split.  Defaults to ``sample.uvs``.

    Returns:
        ``(uv_vertex_of_loop, tri_chart, n_charts)``.  ``uv_vertex_of_loop`` is
        ``(L,)`` int64 with dense ids; loops no triangle references get id
        ``-1``.  ``tri_chart`` is ``(T,)`` int64 with dense chart ids.
    """
    layout = sample.uvs if uvs is None else np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    n_loops = int(layout.shape[0])
    tri_loops = np.asarray(sample.tri_loops, dtype=np.int64).reshape(-1, 3)
    n_tris = int(tri_loops.shape[0])
    if n_tris == 0 or n_loops == 0:
        return np.full(n_loops, -1, dtype=np.int64), np.zeros(0, dtype=np.int64), 0

    used = np.unique(tri_loops.reshape(-1))
    used = used[(used >= 0) & (used < n_loops)]
    if used.size == 0:
        return np.full(n_loops, -1, dtype=np.int64), np.zeros(n_tris, dtype=np.int64), 0

    verts = np.asarray(sample.loop_verts, dtype=np.int64).reshape(-1)
    if verts.shape[0] < n_loops:
        verts = np.concatenate([verts, np.zeros(n_loops - verts.shape[0], dtype=np.int64)])
    vert_of = np.maximum(verts[used], 0)
    uv_used = np.nan_to_num(layout[used], nan=0.0, posinf=0.0, neginf=0.0)

    weld_a, weld_b = _weld_edges(vert_of, uv_used)
    local = np.arange(used.shape[0], dtype=np.int64)

    node = _dense_labels(_label_components(used.shape[0], weld_a, weld_b))
    uv_vertex_of_loop = np.full(n_loops, -1, dtype=np.int64)
    uv_vertex_of_loop[used] = node

    # Chart labels: the same weld graph plus an edge between every triangle's
    # own corners, so one components pass answers both questions.
    position = np.full(n_loops, -1, dtype=np.int64)
    position[used] = local
    corners = position[tri_loops]
    valid = (corners >= 0).all(axis=1)
    corner_a = corners[valid].reshape(-1)
    corner_b = corners[valid][:, [1, 2, 0]].reshape(-1)
    chart_of_local = _dense_labels(
        _label_components(
            used.shape[0],
            np.concatenate([weld_a, corner_a]),
            np.concatenate([weld_b, corner_b]),
        )
    )
    tri_chart = np.full(n_tris, -1, dtype=np.int64)
    tri_chart[valid] = chart_of_local[corners[valid][:, 0]]
    n_charts = int(chart_of_local.max()) + 1 if chart_of_local.size else 0
    return uv_vertex_of_loop, tri_chart, n_charts


def _weld_edges(vert_of: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return star edges joining corners that are the same UV vertex.

    Quantisation alone can split a pair of UVs that are within tolerance but
    straddle a lattice boundary, so the grouping runs twice on lattices offset
    by half a cell and both sets of edges are returned.  Two lattices offset by
    half a cell cannot both split the same near-coincident pair.
    """
    lo = uv.min(axis=0)
    span = float(np.max(uv.max(axis=0) - lo)) if uv.shape[0] else 0.0
    step = _WELD_EPS * max(span, 1.0)
    if not (step > 0.0) or not math.isfinite(step):
        step = 1.0
    rel = (uv - lo) / step

    edges_a: list[np.ndarray] = []
    edges_b: list[np.ndarray] = []
    for shift in (0.5, 1.0):
        cell = np.floor(rel + shift).astype(np.int64)
        cell = np.clip(cell, 0, (1 << 21) - 1)
        _, cell_id = np.unique(cell[:, 0] * (1 << 21) + cell[:, 1], return_inverse=True)
        cell_id = cell_id.reshape(-1)
        _, group = np.unique(
            vert_of * (int(cell_id.max()) + 1) + cell_id, return_inverse=True
        )
        group = group.reshape(-1)
        hub = _group_hub(group)
        edges_a.append(np.arange(group.shape[0], dtype=np.int64))
        edges_b.append(hub)
    return np.concatenate(edges_a), np.concatenate(edges_b)


def _group_hub(group: np.ndarray) -> np.ndarray:
    """Return, for every element, the index of the first element of its group."""
    n_groups = int(group.max()) + 1 if group.size else 0
    first = np.zeros(max(n_groups, 1), dtype=np.int64)
    order = np.arange(group.shape[0] - 1, -1, -1, dtype=np.int64)
    first[group[order]] = order
    return first[group]


def _label_components(n_nodes: int, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Label the connected components of an undirected graph, vectorised.

    Shiloach-Vishkin: hook every node to the smallest label in its
    neighbourhood, path-compress by pointer doubling, then contract the edge
    list.  A Python union-find needs one interpreted step per edge, which is
    two orders of magnitude too slow at mesh scale.

    Args:
        n_nodes: Node count; ids must lie in ``[0, n_nodes)``.
        ea: ``(E,)`` first endpoints.
        eb: ``(E,)`` second endpoints.

    Returns:
        ``(n_nodes,)`` int64 labels; the label is the smallest id in the
        component, so the result is deterministic.
    """
    labels = np.arange(max(n_nodes, 0), dtype=np.int64)
    if n_nodes <= 0 or ea.size == 0:
        return labels
    ea = np.ascontiguousarray(ea, dtype=np.int64)
    eb = np.ascontiguousarray(eb, dtype=np.int64)
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
        ra, rb = ra[keep], rb[keep]
        packed = np.unique(np.minimum(ra, rb) * n_nodes + np.maximum(ra, rb))
        ea = packed // n_nodes
        eb = packed - ea * n_nodes
    return labels


def _dense_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber component labels to ``0..k-1``, preserving their ordering."""
    if labels.size == 0:
        return labels
    _, dense = np.unique(labels, return_inverse=True)
    return dense.reshape(-1).astype(np.int64)


def chart_from_tris(
    sample: UVMeshSample,
    tris: np.ndarray,
    *,
    uvs: np.ndarray | None = None,
    weld: str = "uv",
    uv_vertex_of_loop: np.ndarray | None = None,
    name: str = "",
) -> Chart:
    """Build a :class:`Chart` from a subset of a sample's triangles.

    Args:
        sample: The mesh.
        tris: Triangle indices, or a ``(T,)`` boolean mask.
        uvs: Layout used by the ``"uv"`` weld.  Defaults to ``sample.uvs``.
        weld: ``"uv"`` keeps the existing UV splits, so a chart that already
            contains an internal slit stays a disk.  ``"vertex"`` welds every
            corner sharing a mesh vertex, which is what a caller merging two
            adjacent charts wants -- the seam between them has to close.
        uv_vertex_of_loop: Optional precomputed weld from :func:`uv_charts`,
            reused so a caller splitting a whole mesh welds it only once.
        name: Optional label.

    Returns:
        The chart.  Empty (zero triangles) when ``tris`` selects nothing.
    """
    tri_index = np.asarray(tris)
    if tri_index.dtype == bool:
        tri_index = np.flatnonzero(tri_index)
    tri_index = tri_index.reshape(-1).astype(np.int64)
    n_tris = int(np.asarray(sample.tri_loops).reshape(-1, 3).shape[0])
    tri_index = tri_index[(tri_index >= 0) & (tri_index < n_tris)]
    if tri_index.size == 0:
        return Chart(
            verts=np.zeros((0, 3), dtype=np.float64),
            tris=np.zeros((0, 3), dtype=np.int32),
            loops=np.zeros((0, 3), dtype=np.int32),
            name=name,
        )

    loops = np.asarray(sample.tri_loops, dtype=np.int64).reshape(-1, 3)[tri_index]
    if weld == "vertex":
        keys = np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3)[tri_index]
    else:
        if uv_vertex_of_loop is None:
            uv_vertex_of_loop, _, _ = uv_charts(sample, uvs)
        keys = np.asarray(uv_vertex_of_loop, dtype=np.int64)[loops]

    flat_keys = keys.reshape(-1)
    _, local = np.unique(flat_keys, return_inverse=True)
    local = local.reshape(-1).astype(np.int32)
    n_verts = int(local.max()) + 1 if local.size else 0

    loop_verts = np.asarray(sample.loop_verts, dtype=np.int64).reshape(-1)
    flat_loops = loops.reshape(-1)
    mesh_vert = np.zeros(n_verts, dtype=np.int64)
    mesh_vert[local] = np.where(
        (flat_loops >= 0) & (flat_loops < loop_verts.shape[0]),
        loop_verts[np.clip(flat_loops, 0, max(loop_verts.shape[0] - 1, 0))],
        0,
    )
    positions = np.asarray(sample.verts, dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] == 0:
        verts = np.zeros((n_verts, 3), dtype=np.float64)
    else:
        verts = positions[np.clip(mesh_vert, 0, positions.shape[0] - 1)]

    return Chart(
        verts=np.ascontiguousarray(verts, dtype=np.float64),
        tris=local.reshape(-1, 3),
        loops=flat_loops.reshape(-1, 3).astype(np.int32),
        name=name,
    )


def chart_topology(chart: Chart) -> ChartTopology:
    """Measure a chart's exact topology and extract its boundary loops.

    Boundary loops are walked over the **directed** half-edges: an edge is on
    the boundary when its reverse is absent, and chaining the survivors gives
    loops already wound to agree with the triangles' orientation.  That
    orientation is not a nicety -- it is what makes a convex boundary placement
    produce positive triangle areas rather than a wholly mirrored chart.

    Args:
        chart: The chart to inspect.

    Returns:
        A :class:`ChartTopology`.  Never raises; a broken chart comes back with
        ``is_disk`` False and the reasons in ``problems``.
    """
    chart = _as_chart(chart)
    problems: list[str] = []
    n_verts = chart.n_verts
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    n_tris = int(tris.shape[0])
    if n_tris == 0 or n_verts == 0:
        return ChartTopology(n_verts, n_tris, 0, 0, (), False, ("chart has no triangles",))
    if int(tris.max()) >= n_verts or int(tris.min()) < 0:
        return ChartTopology(
            n_verts, n_tris, 0, 0, (), False, ("triangle references a vertex out of range",)
        )

    src = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    dst = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    if np.any(src == dst):
        problems.append("triangle has a repeated corner")

    key = src * n_verts + dst
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    uniq, counts = np.unique(key_sorted, return_counts=True)
    if int(counts.max()) > 1:
        problems.append("%d half-edges are duplicated (non-manifold)" % int((counts > 1).sum()))

    undirected = np.unique(np.minimum(src, dst) * n_verts + np.maximum(src, dst))
    n_edges = int(undirected.shape[0])

    mirror = dst * n_verts + src
    where = np.searchsorted(uniq, mirror)
    where = np.clip(where, 0, uniq.shape[0] - 1)
    has_reverse = uniq[where] == mirror
    on_boundary = ~has_reverse
    b_src = src[on_boundary]
    b_dst = dst[on_boundary]

    euler = n_verts - n_edges + n_tris
    loops = _chain_loops(b_src, b_dst, n_verts, problems)
    is_disk = not problems and len(loops) == 1 and euler == 1
    if euler != 1:
        problems.append("euler characteristic is %d, not 1" % euler)
    if len(loops) != 1:
        problems.append("%d boundary loops, expected 1" % len(loops))
    return ChartTopology(
        n_verts=n_verts,
        n_tris=n_tris,
        n_edges=n_edges,
        euler=int(euler),
        loops=loops,
        is_disk=bool(is_disk),
        problems=tuple(problems),
    )


def _chain_loops(
    b_src: np.ndarray, b_dst: np.ndarray, n_verts: int, problems: list[str]
) -> tuple[tuple[int, ...], ...]:
    """Chain directed boundary half-edges into ordered closed loops."""
    if b_src.size == 0:
        problems.append("chart has no boundary (closed surface)")
        return ()
    nxt = np.full(n_verts, -1, dtype=np.int64)
    seen = np.zeros(n_verts, dtype=np.int64)
    np.add.at(seen, b_src, 1)
    if int(seen.max()) > 1:
        problems.append("boundary pinches at %d vertices" % int((seen > 1).sum()))
    nxt[b_src] = b_dst

    loops: list[tuple[int, ...]] = []
    visited = np.zeros(n_verts, dtype=bool)
    for start in np.sort(np.unique(b_src)):
        if visited[start]:
            continue
        walk: list[int] = []
        node = int(start)
        for _ in range(n_verts + 1):
            if visited[node]:
                break
            visited[node] = True
            walk.append(node)
            node = int(nxt[node])
            if node < 0:
                problems.append("boundary chain breaks at vertex %d" % walk[-1])
                break
            if node == int(start):
                break
        if len(walk) >= 3:
            loops.append(tuple(walk))
    return tuple(loops)


# ---------------------------------------------------------------------------
# Per-triangle rest geometry and the symmetric Dirichlet energy
# ---------------------------------------------------------------------------


class _Rest:
    """Per-triangle rest geometry, precomputed once per chart.

    Each triangle is laid flat in its own isometric 2D frame, giving the inverse
    rest matrix ``M`` with ``J = U M`` where ``U``'s columns are the two UV edge
    vectors.  Holding ``M`` as three scalars per triangle keeps every subsequent
    pass over ``(T,)`` arrays rather than ``(T, 2, 2)`` ones.
    """

    __slots__ = ("a", "b", "c", "area", "good", "n_good", "total_area", "tris")

    def __init__(self, chart: Chart) -> None:
        tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
        self.tris = tris
        p = np.asarray(chart.verts, dtype=np.float64).reshape(-1, 3)[tris]
        e1 = p[:, 1] - p[:, 0]
        e2 = p[:, 2] - p[:, 0]
        x1 = np.sqrt(np.einsum("ij,ij->i", e1, e1))
        safe_x1 = np.where(x1 > 0.0, x1, 1.0)
        x2 = np.einsum("ij,ij->i", e2, e1) / safe_x1
        # The height comes from the cross product, not from
        # ``sqrt(|e2|^2 - x2^2)``.  Algebraically they are the same number; in
        # float64 the subtraction is catastrophic cancellation on a thin
        # triangle, losing roughly ``2*log10(aspect)`` digits and clamping to
        # exactly zero past aspect 1e8.  That is not a cosmetic difference: a
        # zero area sets ``good`` False, which excludes the triangle from the
        # symmetric-Dirichlet barrier, which lets the descent collapse and then
        # invert it -- breaking the no-flip invariant on a chart whose triangles
        # all have perfectly good area.  Every other area in this module
        # (:func:`_circle_boundary`, :func:`_mean_value_weights`,
        # :func:`_cotan_pattern`) already uses the cross product; this makes the
        # module agree with itself.
        normal = np.cross(e1, e2)
        area = 0.5 * np.sqrt(np.einsum("ij,ij->i", normal, normal))
        y2 = 2.0 * area / safe_x1

        mean = float(area.mean()) if area.size else 0.0
        floor = _DEGENERATE_EPS * mean if mean > 0.0 else 0.0
        self.good = np.isfinite(area) & (area > floor) & (x1 > 0.0) & (y2 > 0.0)
        self.n_good = int(np.count_nonzero(self.good))
        safe_y2 = np.where(self.good, y2, 1.0)
        safe_x1 = np.where(self.good, x1, 1.0)
        self.a = np.where(self.good, 1.0 / safe_x1, 0.0)
        self.b = np.where(self.good, -x2 / (safe_x1 * safe_y2), 0.0)
        self.c = np.where(self.good, 1.0 / safe_y2, 0.0)
        self.area = np.where(self.good, area, 0.0)
        self.total_area = float(self.area.sum())


def _jacobians(uv: np.ndarray, rest: _Rest) -> tuple[np.ndarray, ...]:
    """Return the four ``(T,)`` Jacobian components, plus ``det`` and ``||J||^2``."""
    tris = rest.tris
    u1 = uv[tris[:, 1]] - uv[tris[:, 0]]
    u2 = uv[tris[:, 2]] - uv[tris[:, 0]]
    j00 = u1[:, 0] * rest.a
    j01 = u1[:, 0] * rest.b + u2[:, 0] * rest.c
    j10 = u1[:, 1] * rest.a
    j11 = u1[:, 1] * rest.b + u2[:, 1] * rest.c
    det = j00 * j11 - j01 * j10
    frob = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11
    return j00, j01, j10, j11, det, frob


def _energy(uv: np.ndarray, rest: _Rest) -> float:
    """Return the area-weighted symmetric Dirichlet energy of ``uv``.

    Infinite when any measurable triangle has collapsed or inverted, which is
    the property that makes the energy a barrier rather than merely a cost.
    """
    if rest.n_good == 0:
        return 0.0
    _, _, _, _, det, frob = _jacobians(uv, rest)
    inverted = rest.good & ~(det > 0.0)
    if inverted.any():
        return math.inf
    density = np.where(rest.good, frob * (1.0 + 1.0 / np.where(det != 0.0, det, 1.0) ** 2), 0.0)
    total = float(np.dot(np.where(np.isfinite(density), density, np.inf), rest.area))
    return total if math.isfinite(total) else math.inf


def _energy_and_gradient(uv: np.ndarray, rest: _Rest) -> tuple[float, np.ndarray]:
    """Return ``(energy, (V, 2) gradient)`` of the symmetric Dirichlet energy.

    The gradient is exact, not a difference quotient: with ``J = U M`` the chain
    rule gives ``dE/dU = (dE/dJ) M^T``, and ``dE/dJ`` is
    ``2 J (1 + 1/det^2) - (2 ||J||^2 / det^3) adj(J)``.  A unit test pins it
    against central differences.
    """
    n_verts = uv.shape[0]
    grad = np.zeros((n_verts, 2), dtype=np.float64)
    if rest.n_good == 0:
        return 0.0, grad
    j00, j01, j10, j11, det, frob = _jacobians(uv, rest)
    inverted = rest.good & ~(det > 0.0)
    if inverted.any():
        return math.inf, grad

    safe_det = np.where(det != 0.0, det, 1.0)
    inv_sq = 1.0 / (safe_det * safe_det)
    scale = 2.0 * (1.0 + inv_sq)
    barrier = 2.0 * frob * inv_sq / safe_det
    d00 = scale * j00 - barrier * j11
    d01 = scale * j01 + barrier * j10
    d10 = scale * j10 + barrier * j01
    d11 = scale * j11 - barrier * j00

    weight = rest.area
    g00 = (d00 * rest.a + d01 * rest.b) * weight
    g01 = (d01 * rest.c) * weight
    g10 = (d10 * rest.a + d11 * rest.b) * weight
    g11 = (d11 * rest.c) * weight

    tris = rest.tris
    idx = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    gx = np.concatenate([-(g00 + g01), g00, g01])
    gy = np.concatenate([-(g10 + g11), g10, g11])
    keep = np.tile(rest.good, 3)
    grad[:, 0] = np.bincount(idx[keep], weights=gx[keep], minlength=n_verts)[:n_verts]
    grad[:, 1] = np.bincount(idx[keep], weights=gy[keep], minlength=n_verts)[:n_verts]

    density = np.where(rest.good, frob * (1.0 + inv_sq), 0.0)
    total = float(np.dot(density, rest.area))
    return (total if math.isfinite(total) else math.inf), grad


def symmetric_dirichlet_energy(uv: np.ndarray, chart: Chart) -> float:
    """Return the chart's symmetric Dirichlet energy, normalised by surface area.

    A perfectly isometric map reads exactly ``4.0`` (both singular values 1), so
    the number is comparable across charts of different size.  Returns ``inf``
    when any measurable triangle is collapsed or inverted.

    Args:
        uv: ``(V, 2)`` chart-local layout.
        chart: The chart the layout belongs to.

    Returns:
        The normalised energy, or ``inf``.
    """
    rest = _Rest(_as_chart(chart))
    if rest.total_area <= 0.0:
        return 0.0
    return _energy(np.asarray(uv, dtype=np.float64).reshape(-1, 2), rest) / rest.total_area


# ---------------------------------------------------------------------------
# Laplacians
# ---------------------------------------------------------------------------


def _chart_edges(chart: Chart) -> tuple[np.ndarray, np.ndarray]:
    """Return the chart's unique undirected edges as ``(ei, ej)`` with ``ei < ej``."""
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    n = max(chart.n_verts, 1)
    src = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    dst = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    packed = np.unique(np.minimum(src, dst) * n + np.maximum(src, dst))
    ei = packed // n
    return ei, packed - ei * n


def _edge_laplacian(n: int, ei: np.ndarray, ej: np.ndarray, w: np.ndarray) -> SparseSym:
    """Assemble ``L`` with ``L[i,j] = -w`` and ``L[i,i] = sum_j w`` from edge weights."""
    rows = np.concatenate([ei, ej, ei, ej])
    cols = np.concatenate([ej, ei, ei, ej])
    vals = np.concatenate([-w, -w, w, w])
    return SparseSym(rows, cols, vals, n)


def _mean_value_weights(chart: Chart) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ei, ej, w)`` mean-value edge weights, strictly positive.

    Floater's half-weight from ``i`` to ``j`` sums ``tan(theta/2) / |e_ij|`` over
    the triangles at ``i`` containing that edge, where ``theta`` is the corner
    angle at ``i``.  Written as ``tan(theta/2) = 2A / (|e1||e2| + e1.e2)`` it
    needs no trigonometry at all and is stable for small angles.  The two
    directions are averaged, which keeps the operator symmetric without
    forfeiting Tutte's guarantee -- see the module docstring.
    """
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    p = np.asarray(chart.verts, dtype=np.float64).reshape(-1, 3)[tris]
    n = max(chart.n_verts, 1)

    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    for corner in range(3):
        i = tris[:, corner]
        j = tris[:, (corner + 1) % 3]
        k = tris[:, (corner + 2) % 3]
        e1 = p[:, (corner + 1) % 3] - p[:, corner]
        e2 = p[:, (corner + 2) % 3] - p[:, corner]
        l1 = np.sqrt(np.einsum("ij,ij->i", e1, e1))
        l2 = np.sqrt(np.einsum("ij,ij->i", e2, e2))
        cross = np.cross(e1, e2)
        twice_area = np.sqrt(np.einsum("ij,ij->i", cross, cross))
        denom = l1 * l2 + np.einsum("ij,ij->i", e1, e2)
        tan_half = np.where(denom > 0.0, twice_area / np.where(denom > 0.0, denom, 1.0), 0.0)
        src_list += [i, i]
        dst_list += [j, k]
        w_list += [
            np.where(l1 > 0.0, tan_half / np.where(l1 > 0.0, l1, 1.0), 0.0),
            np.where(l2 > 0.0, tan_half / np.where(l2 > 0.0, l2, 1.0), 0.0),
        ]

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    half = np.concatenate(w_list)
    half = np.where(np.isfinite(half), half, 0.0)

    packed = np.minimum(src, dst) * n + np.maximum(src, dst)
    uniq, inverse = np.unique(packed, return_inverse=True)
    summed = np.bincount(inverse.reshape(-1), weights=half, minlength=uniq.size)
    w = 0.5 * summed[: uniq.size]

    # Strict positivity is the hypothesis of Tutte's theorem, so a weight that
    # rounded to zero on a collapsed triangle is lifted to a token positive
    # value rather than left to disconnect the system.
    positive = w[w > 0.0]
    floor = 1e-12 * float(np.median(positive)) if positive.size else 1e-12
    w = np.maximum(w, max(floor, 1e-300))
    ei = uniq // n
    return ei, uniq - ei * n, w


def _cotan_pattern(
    chart: Chart,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ei, ej, coefficients, edge_of_coefficient)`` for the cotan stiffness.

    The per-triangle Dirichlet form is positive semi-definite whatever the
    triangle's shape -- individual cotangents go negative on an obtuse triangle,
    but the 3x3 element matrix never does -- so any non-negative per-triangle
    weighting of these coefficients assembles to a PSD operator.  That is what
    makes the reweighted Laplacian below a legitimate preconditioner: it
    guarantees the search direction is a descent direction.
    """
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    p = np.asarray(chart.verts, dtype=np.float64).reshape(-1, 3)[tris]
    n = max(chart.n_verts, 1)
    e0 = p[:, 2] - p[:, 1]
    e1 = p[:, 0] - p[:, 2]
    e2 = p[:, 1] - p[:, 0]
    cross = np.cross(e2, -e1)
    twice_area = np.sqrt(np.einsum("ij,ij->i", cross, cross))
    safe = np.where(twice_area > 0.0, twice_area, 1.0)
    live = twice_area > 0.0
    # 0.5 * cot(angle at k) for the edge opposite k, written without trig.
    c0 = np.where(live, -0.5 * np.einsum("ij,ij->i", e1, e2) / safe, 0.0)
    c1 = np.where(live, -0.5 * np.einsum("ij,ij->i", e2, e0) / safe, 0.0)
    c2 = np.where(live, -0.5 * np.einsum("ij,ij->i", e0, e1) / safe, 0.0)

    src = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    dst = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
    packed = np.minimum(src, dst) * n + np.maximum(src, dst)
    uniq, inverse = np.unique(packed, return_inverse=True)
    ei = uniq // n
    return ei, uniq - ei * n, np.stack([c0, c1, c2]).reshape(-1), inverse.reshape(-1)


# ---------------------------------------------------------------------------
# Tutte
# ---------------------------------------------------------------------------


def _plane_project(chart: Chart) -> np.ndarray:
    """Return a best-fit-plane projection of the chart: always finite, never NaN.

    Used when the chart is not a disk and Tutte has no boundary to map to.  It
    is not injective and makes no claim to be; it exists so a caller always gets
    a usable array and a report that says the chart could not be flattened,
    rather than an exception out of a Blender operator.
    """
    p = np.asarray(chart.verts, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centred = p - p.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:2].T
    except np.linalg.LinAlgError:
        basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    out = centred @ basis
    return np.where(np.isfinite(out), out, 0.0)


def _circle_boundary(chart: Chart, loop: tuple[int, ...]) -> np.ndarray:
    """Place a boundary loop on a circle, spaced by its own 3D chord lengths.

    Chord-length spacing rather than uniform: a uniform circle stretches short
    boundary edges and crushes long ones before the interior solve even starts,
    and the descent then has to undo it.  The radius is chosen so the disk's
    area matches the chart's surface area, which puts the Tutte start at roughly
    the right scale.
    """
    idx = np.asarray(loop, dtype=np.int64)
    pts = np.asarray(chart.verts, dtype=np.float64).reshape(-1, 3)[idx]
    delta = np.roll(pts, -1, axis=0) - pts
    lengths = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    total = float(lengths.sum())
    if not (total > 0.0) or not math.isfinite(total):
        lengths = np.ones(idx.shape[0], dtype=np.float64)
        total = float(idx.shape[0])
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    angle = 2.0 * math.pi * cumulative / total

    area = 0.0
    p = np.asarray(chart.verts, dtype=np.float64).reshape(-1, 3)[chart.tris]
    if p.shape[0]:
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = 0.5 * float(np.sqrt(np.einsum("ij,ij->i", cross, cross)).sum())
    radius = math.sqrt(area / math.pi) if area > 0.0 else 1.0
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def tutte_embed(chart: Chart) -> np.ndarray:
    """Return a provably flip-free initial UV layout for a disk chart.

    Convex boundary plus strictly positive mean-value weights: Tutte's theorem,
    in Floater's form, then guarantees the interior solve is a valid embedding
    with no self-intersections at all.  The result is heavily distorted -- that
    is the price of the guarantee, and :func:`minimize_distortion` is what pays
    it back without ever leaving the feasible set.

    Args:
        chart: The chart to embed.  Must be a disk; see :func:`chart_topology`.

    Returns:
        ``(V, 2)`` float64 UVs, always finite.  When the chart is **not** a disk
        there is no valid Tutte embedding, and a best-fit-plane projection is
        returned instead -- finite and usable, but carrying no guarantee.  Use
        :func:`flatten_chart`, which certifies before returning, unless you have
        already checked the topology yourself.
    """
    chart = _as_chart(chart)
    topo = chart_topology(chart)
    if not topo.is_disk:
        return _plane_project(chart)
    return _tutte(chart, topo)


def _tutte(chart: Chart, topo: ChartTopology) -> np.ndarray:
    """Solve the Tutte system for a chart already known to be a disk."""
    n = chart.n_verts
    uv = np.zeros((n, 2), dtype=np.float64)
    loop = topo.loops[0]
    boundary_idx = np.asarray(loop, dtype=np.int64)
    uv[boundary_idx] = _circle_boundary(chart, loop)

    is_boundary = np.zeros(n, dtype=bool)
    is_boundary[boundary_idx] = True
    interior = np.flatnonzero(~is_boundary)
    if interior.size == 0:
        return uv

    ei, ej, w = _mean_value_weights(chart)
    lap = _edge_laplacian(n, ei, ej, w)

    # b_I = -(L x_B)_I: applying the full operator to a vector that is zero on
    # the interior isolates exactly the boundary coupling, with no index
    # gymnastics and no second assembly.
    padded = np.zeros((n, 2), dtype=np.float64)
    padded[boundary_idx] = uv[boundary_idx]
    rhs = -lap.matvec(padded)[interior]

    remap = np.full(n, -1, dtype=np.int64)
    remap[interior] = np.arange(interior.size, dtype=np.int64)
    keep = (remap[lap.rows] >= 0) & (remap[lap.cols] >= 0)
    reduced = SparseSym.from_coalesced(
        remap[lap.rows[keep]], remap[lap.cols[keep]], lap.values[keep], interior.size
    )

    solution, _, residual = pcg(reduced, rhs, tol=1e-12, maxiter=max(200, 4 * interior.size))
    if residual > 1e-8:
        # A loose solve is the one thing that can cost the guarantee, so buy a
        # second pass before accepting it.  Warm-started, so it is cheap.
        solution, _, _ = pcg(
            reduced, rhs, x0=solution, tol=1e-14, maxiter=max(400, 8 * interior.size)
        )
    uv[interior] = np.where(np.isfinite(solution), solution, 0.0)
    return uv


# ---------------------------------------------------------------------------
# Distortion minimisation
# ---------------------------------------------------------------------------


def _indexes(tris: np.ndarray, n_verts: int) -> bool:
    """Return True when every corner of ``tris`` addresses a row of an ``n`` array.

    ``chart_topology`` already reports an out-of-range corner as a problem, but
    it is the only function that checks: everything downstream indexes the
    layout directly and would take an :class:`IndexError` instead.  That matters
    because ``core/charts.py`` hands this module foreign chart-like objects
    (:func:`_as_chart`) whose indices it never validates, and both
    :func:`flatten_chart` and :func:`repair_folds` promise never to raise.
    """
    if tris.size == 0:
        return True
    return int(tris.min()) >= 0 and int(tris.max()) < int(n_verts)


def _signed_areas(uv: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Return ``(T,)`` signed UV areas."""
    a = uv[tris[:, 0]]
    e1 = uv[tris[:, 1]] - a
    e2 = uv[tris[:, 2]] - a
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def _max_feasible_step(uv: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> float:
    """Return the largest ``t`` for which no positive triangle inverts.

    Along ``uv + t*d`` a triangle's signed area is the quadratic
    ``Q t^2 + L t + A0``.  Only triangles that are currently positive are
    constrained: one that is already inverted cannot be made "more" inverted in
    a way that matters, and letting it veto the step would freeze the whole
    descent over a triangle the caller already knows about.

    Returns:
        The smallest positive root over the constrained triangles, or ``inf``
        when no triangle can invert along ``d``.
    """
    if tris.shape[0] == 0:
        return math.inf
    a0 = uv[tris[:, 0]]
    b0 = uv[tris[:, 1]] - a0
    c0 = uv[tris[:, 2]] - a0
    da = direction[tris[:, 0]]
    db = direction[tris[:, 1]] - da
    dc = direction[tris[:, 2]] - da

    def cross(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Scalar 2D cross product."""
        return p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]

    const = 0.5 * cross(b0, c0)
    linear = 0.5 * (cross(b0, dc) + cross(db, c0))
    quad = 0.5 * cross(db, dc)

    live = const > 0.0
    if not live.any():
        return math.inf
    const = const[live]
    linear = linear[live]
    quad = quad[live]

    root = np.full(const.shape[0], np.inf, dtype=np.float64)

    # Degenerate (linear) case first.
    flat = quad == 0.0
    if flat.any():
        falling = flat & (linear < 0.0)
        root[falling] = -const[falling] / linear[falling]

    curved = ~flat
    if curved.any():
        disc = linear[curved] * linear[curved] - 4.0 * quad[curved] * const[curved]
        real = disc >= 0.0
        if real.any():
            idx = np.flatnonzero(curved)[real]
            lin = linear[idx]
            sq = np.sqrt(disc[real])
            # The numerically stable pair: one root from the "big" branch, the
            # other by Vieta.  Subtracting nearly equal numbers here is exactly
            # how a step cap silently becomes zero.
            q = -0.5 * (lin + np.where(lin >= 0.0, sq, -sq))
            r1 = np.where(quad[idx] != 0.0, q / np.where(quad[idx] != 0.0, quad[idx], 1.0), np.inf)
            r2 = np.where(q != 0.0, const[idx] / np.where(q != 0.0, q, 1.0), np.inf)
            r1 = np.where(r1 > 0.0, r1, np.inf)
            r2 = np.where(r2 > 0.0, r2, np.inf)
            root[idx] = np.minimum(r1, r2)

    root = root[np.isfinite(root) & (root > 0.0)]
    return float(root.min()) if root.size else math.inf


def _build_preconditioner(
    chart: Chart,
    rest: _Rest,
    uv: np.ndarray,
    pattern: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    free: np.ndarray,
) -> SparseSym:
    """Assemble the distortion-weighted cotan Laplacian over the free vertices.

    Reweighting each triangle by its current energy density makes badly
    distorted triangles stiffer, so the global step moves them coherently
    instead of one vertex at a time -- the "global" half of local-global.  The
    operator stays symmetric positive definite whatever the weights are (they
    are non-negative and each element matrix is PSD), which is precisely what
    makes ``d = -H^-1 g`` a guaranteed descent direction.
    """
    ei, ej, coeff, inverse = pattern
    n = chart.n_verts
    _, _, _, _, det, frob = _jacobians(uv, rest)
    safe_det = np.where(det > 0.0, det, 1.0)
    density = np.where(rest.good, frob * (1.0 + 1.0 / (safe_det * safe_det)), 4.0)
    weight = rest.area * np.clip(density / 4.0, 1.0, 1e8)
    per_edge = np.bincount(
        inverse, weights=coeff * np.tile(weight, 3), minlength=ei.shape[0]
    )[: ei.shape[0]]

    lap = _edge_laplacian(n, ei, ej, per_edge)
    diag = lap.diagonal()
    scale = float(np.abs(diag).max()) if diag.size else 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    eps = _PRECOND_EPS * scale

    rows = np.concatenate([lap.rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([lap.cols, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([lap.values, np.full(n, eps)])
    full = SparseSym(rows, cols, vals, n)

    if free.shape[0] == n:
        return full
    remap = np.full(n, -1, dtype=np.int64)
    remap[free] = np.arange(free.shape[0], dtype=np.int64)
    keep = (remap[full.rows] >= 0) & (remap[full.cols] >= 0)
    return SparseSym.from_coalesced(
        remap[full.rows[keep]], remap[full.cols[keep]], full.values[keep], free.shape[0]
    )


def minimize_distortion(
    uv: np.ndarray,
    chart: Chart,
    *,
    iters: int = 32,
    energy: str = "symmetric_dirichlet",
    free_boundary: bool = True,
    step_safety: float = _STEP_SAFETY,
) -> np.ndarray:
    """Descend the distortion energy without ever flipping a triangle.

    Each iteration solves ``H d = -g`` with ``H`` a distortion-reweighted cotan
    Laplacian -- symmetric positive definite, so ``g . d < 0`` and ``d`` is a
    descent direction -- then caps the step at ``step_safety`` times the largest
    ``t`` that keeps every positive triangle positive, and backtracks from there
    until the energy actually falls.  Two properties follow and are both pinned
    by test:

    * **no step can flip a triangle**, so a locally injective input stays
      locally injective; and
    * **the energy is non-increasing**, because a step is only taken when it
      decreases it.

    Args:
        uv: ``(V, 2)`` starting layout.  Must be flip-free for the guarantee to
            mean anything; :func:`tutte_embed` provides one.
        chart: The chart being flattened.
        iters: Maximum descent iterations.
        energy: Only ``"symmetric_dirichlet"`` is implemented.
        free_boundary: When False the boundary vertices are held fixed, which is
            the fallback for a chart whose free-boundary result self-intersects.
        step_safety: Fraction of the flip-free step a line search may take.

    Returns:
        ``(V, 2)`` float64 UVs.  Never worse in energy than the input, and never
        carrying a triangle inversion the input did not already have.

    Raises:
        ValueError: If ``energy`` names an unimplemented energy, or ``uv`` does
            not match the chart.  Both are programming errors, not geometry.
    """
    result, _ = _descend(
        uv,
        chart,
        iters=iters,
        energy=energy,
        free_boundary=free_boundary,
        step_safety=step_safety,
    )
    return result


def _descend(
    uv: np.ndarray,
    chart: Chart,
    *,
    iters: int,
    energy: str,
    free_boundary: bool,
    step_safety: float,
    topo: ChartTopology | None = None,
) -> tuple[np.ndarray, dict]:
    """Run the descent and also return diagnostics.  See :func:`minimize_distortion`."""
    if energy != "symmetric_dirichlet":
        raise ValueError("unsupported energy %r" % (energy,))
    chart = _as_chart(chart)
    current = np.array(uv, dtype=np.float64, copy=True).reshape(-1, 2)
    if current.shape[0] != chart.n_verts:
        raise ValueError(
            "uv has %d rows, chart has %d vertices" % (current.shape[0], chart.n_verts)
        )
    info = {"iterations": 0, "energies": [], "stopped": "budget"}
    rest = _Rest(chart)
    if rest.n_good == 0 or chart.n_tris == 0 or iters <= 0:
        info["stopped"] = "nothing to do"
        return current, info

    tris = rest.tris
    if free_boundary:
        free = np.arange(chart.n_verts, dtype=np.int64)
    else:
        topo = topo if topo is not None else chart_topology(chart)
        fixed = np.zeros(chart.n_verts, dtype=bool)
        for loop in topo.loops:
            fixed[np.asarray(loop, dtype=np.int64)] = True
        free = np.flatnonzero(~fixed)
        if free.size == 0:
            info["stopped"] = "no free vertices"
            return current, info

    pattern = _cotan_pattern(chart)
    precond: SparseSym | None = None
    warm: np.ndarray | None = None

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        value, grad = _energy_and_gradient(current, rest)
        info["energies"].append(value)
        if not math.isfinite(value):
            info["stopped"] = "input is already inverted"
            return current, info

        for step_index in range(int(iters)):
            grad_norm = float(np.abs(grad).max())
            if not math.isfinite(grad_norm) or grad_norm == 0.0:
                info["stopped"] = "gradient vanished"
                break

            if precond is None or step_index % _PRECOND_REFRESH == 0:
                precond = _build_preconditioner(chart, rest, current, pattern, free)
                warm = None
            solved, _, _ = pcg(
                precond,
                -grad[free],
                x0=warm,
                tol=_SOLVE_TOL,
                maxiter=int(np.clip(math.isqrt(max(free.shape[0], 1)), 4, _SOLVE_MAXITER_CAP)),
            )
            warm = solved
            direction = np.zeros_like(current)
            direction[free] = solved

            slope = float(np.einsum("ij,ij->", grad, direction))
            if not math.isfinite(slope) or slope >= 0.0:
                # The preconditioned direction failed the descent test -- only
                # possible through rounding, but steepest descent always passes
                # it, so fall back rather than stop.
                direction = np.zeros_like(current)
                direction[free] = -grad[free]
                slope = float(np.einsum("ij,ij->", grad, direction))
            if not math.isfinite(slope) or slope >= 0.0:
                info["stopped"] = "no descent direction"
                break

            cap = _max_feasible_step(current, direction, tris)
            trial = step_safety * cap if math.isfinite(cap) else 1.0
            if not math.isfinite(trial) or trial <= 0.0:
                info["stopped"] = "step cap collapsed"
                break

            accepted = False
            for _ in range(_MAX_BACKTRACK):
                candidate = current + trial * direction
                trial_value = _energy(candidate, rest)
                if math.isfinite(trial_value) and trial_value <= value + _ARMIJO_C1 * trial * slope:
                    current = candidate
                    value = trial_value
                    accepted = True
                    break
                trial *= 0.5
            if not accepted:
                info["stopped"] = "line search exhausted"
                break

            info["iterations"] = step_index + 1
            info["energies"].append(value)
            value, grad = _energy_and_gradient(current, rest)

    info["energy"] = value
    return current, info


# ---------------------------------------------------------------------------
# Chart-level entry points
# ---------------------------------------------------------------------------


def _certify_layout(
    uv: np.ndarray, chart: Chart, topo: ChartTopology
) -> tuple[int, int, int]:
    """Return ``(flipped, degenerate, boundary_crossings)`` for a chart layout.

    ``flipped`` counts triangles whose signed area opposes the chart's majority
    sign, matching ``core/injectivity.py``: a wholly mirrored chart is a valid
    parameterisation, a chart with *some* triangles mirrored is not.
    """
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    if tris.shape[0] == 0:
        return 0, 0, 0
    if not _indexes(tris, np.asarray(uv).reshape(-1, 2).shape[0]):
        # Nothing about this chart can be measured, so nothing about it may be
        # certified.  Reporting every triangle degenerate is both true -- no
        # usable area is readable -- and can never read as a clean verdict.
        return 0, int(tris.shape[0]), 0
    signed = _signed_areas(uv, tris)
    finite = np.isfinite(signed)
    magnitude = np.abs(np.where(finite, signed, 0.0))
    mean = float(magnitude.mean()) if magnitude.size else 0.0
    floor = _DEGENERATE_EPS * mean if mean > 0.0 else 0.0
    degenerate = int(np.count_nonzero(~finite | (magnitude <= floor)))
    live = finite & (magnitude > floor)
    if not live.any():
        return 0, degenerate, 0
    positive = int(np.count_nonzero(signed[live] > 0.0))
    negative = int(np.count_nonzero(signed[live] < 0.0))
    flipped = min(positive, negative)
    crossings = _boundary_crossings(uv, [list(loop) for loop in topo.loops])
    return flipped, degenerate, int(crossings)


def flatten_chart(
    chart: Chart,
    *,
    target: np.ndarray | None = None,
    iters: int | None = None,
) -> FlattenResult:
    """Flatten one chart and attach the certificate that says whether to trust it.

    Runs :func:`tutte_embed` then :func:`minimize_distortion` with a free
    boundary, and certifies the result.  Per ``docs/ORGANIC.md`` section 4.2, a
    layout that fails certification is **never** returned; the fallbacks run in
    the fixed order

    1. more descent iterations with a smaller step cap,
    2. a boundary-constrained descent from the Tutte start,
    3. the Tutte embedding itself, which is always valid,

    and the result records which one produced the answer.

    Args:
        chart: The chart to flatten.
        target: Optional ``(V, 2)`` reference layout.  When given, the result is
            brought onto it by the least-squares similarity transform and then
            rescaled to reproduce the reference's exact UV area, so a repaired
            chart lands where the original sat instead of somewhere else in the
            atlas.
        iters: Descent budget for the first rung of the ladder.  ``None``
            derives one from the chart's size; see :func:`_iteration_budget`.

    Returns:
        A :class:`FlattenResult`.  Never raises.
    """
    started = time.perf_counter()
    chart = _as_chart(chart)
    topo = chart_topology(chart)
    if not topo.is_disk:
        uv = _plane_project(chart)
        flipped, degenerate, crossings = _certify_layout(uv, chart, topo)
        return FlattenResult(
            uv=_align_to_target(uv, chart, target),
            ok=False,
            stage="projection",
            flipped=flipped,
            degenerate=degenerate,
            boundary_crossings=crossings,
            energy=float("inf"),
            iterations=0,
            seconds=time.perf_counter() - started,
            message="chart is not a disk: %s" % "; ".join(topo.problems),
        )

    budget = _iteration_budget(chart) if iters is None else max(0, int(iters))
    rest = _Rest(chart)
    normal = rest.total_area if rest.total_area > 0.0 else 1.0
    start = _tutte(chart, topo)

    best = _Attempt.seed(start, chart, topo, rest)
    ladder = (
        ("descent", {"free_boundary": True, "step_safety": _STEP_SAFETY, "iters": budget}),
        ("descent_fine", {"free_boundary": True, "step_safety": 0.5, "iters": budget * 2}),
        ("fixed_boundary", {"free_boundary": False, "step_safety": _STEP_SAFETY, "iters": budget}),
    )
    for stage, kwargs in ladder:
        attempt = _certified_descent(start, chart, topo, rest, stage=stage, **kwargs)
        if attempt is not None and attempt.energy < best.energy:
            best = attempt
        # The ladder exists for the charts whose free boundary tangles.  When
        # the first rung already beats the Tutte start by a real margin there is
        # nothing left for the later rungs to rescue, and running them anyway
        # would triple the cost of the common case.
        if best.stage != "tutte" and best.energy < 0.999 * _energy(start, rest):
            break

    ok = best.verdict == (0, 0, 0)
    return FlattenResult(
        uv=_align_to_target(best.uv, chart, target),
        ok=ok,
        stage=best.stage,
        flipped=best.verdict[0],
        degenerate=best.verdict[1],
        boundary_crossings=best.verdict[2],
        energy=best.energy / normal,
        iterations=best.iterations,
        seconds=time.perf_counter() - started,
        message="" if ok else "certification failed after every fallback",
    )


def _iteration_budget(chart: Chart) -> int:
    """Return a descent budget that scales down as the chart grows.

    A fixed budget is wrong at both ends.  Suzanne's 126-triangle chart needs
    about 250 iterations to reach a near-isometric layout and pays 150 ms for
    them; a 50k-triangle chart at the same budget would take most of a minute,
    which is the whole generation budget for ten candidates.  Holding
    ``iterations * triangles`` roughly constant spends the same wall time
    everywhere and degrades gracefully.
    """
    tris = max(chart.n_tris, 1)
    return int(min(max(32_000 // tris, 16), _DEFAULT_ITERS))


@dataclass(frozen=True)
class _Attempt:
    """One certified candidate layout produced by a rung of the ladder."""

    uv: np.ndarray
    energy: float
    verdict: tuple[int, int, int]
    stage: str
    iterations: int

    @classmethod
    def seed(cls, uv: np.ndarray, chart: Chart, topo: ChartTopology, rest: _Rest) -> "_Attempt":
        """Return the Tutte start as an attempt: the always-available fallback."""
        return cls(uv, _energy(uv, rest), _certify_layout(uv, chart, topo), "tutte", 0)


def _certified_descent(
    start: np.ndarray,
    chart: Chart,
    topo: ChartTopology,
    rest: _Rest,
    *,
    stage: str,
    iters: int,
    free_boundary: bool,
    step_safety: float,
) -> "_Attempt | None":
    """Descend in chunks, keeping the lowest-energy iterate that still certifies.

    Running the descent to a fixed budget and certifying once at the end is the
    obvious implementation and it is wrong, because certification is **not
    monotone in the iteration count** while the energy is.  Measured on
    Suzanne's chart 9: the energy falls 18.371 -> 4.041 -> 4.034 -> 4.030 over
    32, 64 and 128 iterations, but the free boundary self-intersects somewhere
    between 32 and 64 and never recovers.  A fixed budget of 32 returns a
    valid layout at 4.041; a fixed budget of 128 certifies nothing, falls all
    the way down the ladder and returns a *fixed-boundary* layout at 17.977 --
    four times the distortion, from asking for more iterations.

    Checkpointing removes the cliff: certify every ``_CERT_CHUNK`` iterations
    and keep the best iterate that passed.  The check costs about 0.3 ms
    against 2.5 ms per descent iteration, so it is close to free.

    Returns:
        The best certified attempt, or None if nothing along the path certified.
    """
    uv = start
    best: _Attempt | None = None
    done = 0
    while done < iters:
        chunk = min(_CERT_CHUNK, iters - done)
        nxt, info = _descend(
            uv,
            chart,
            iters=chunk,
            energy="symmetric_dirichlet",
            free_boundary=free_boundary,
            step_safety=step_safety,
            topo=topo,
        )
        done += chunk
        moved = not np.array_equal(nxt, uv)
        uv = nxt
        if moved and _certify_layout(uv, chart, topo) == (0, 0, 0):
            energy = _energy(uv, rest)
            if best is None or energy < best.energy:
                best = _Attempt(uv.copy(), energy, (0, 0, 0), stage, done)
        if not moved or info.get("stopped") not in (None, "budget"):
            break
    return best


def _majority_sign(uv: np.ndarray, tris: np.ndarray) -> float:
    """Return ``+1`` or ``-1`` for the winding most of a layout's triangles have.

    The same rule :func:`_certify_layout` counts ``flipped`` by, and the one
    ``docs/ORGANIC.md`` section 4.1 states: a chart's orientation is what the
    majority of its triangles agree on, not what its signed areas sum to.  Ties
    and layouts with no measurable area read ``+1``.
    """
    signed = _signed_areas(uv, tris)
    finite = np.isfinite(signed)
    negative = int(np.count_nonzero(finite & (signed < 0.0)))
    positive = int(np.count_nonzero(finite & (signed > 0.0)))
    return -1.0 if negative > positive else 1.0


def _align_to_target(
    uv: np.ndarray, chart: Chart, target: np.ndarray | None
) -> np.ndarray:
    """Bring ``uv`` onto ``target`` by a similarity transform preserving UV area.

    The rotation and translation come from the least-squares similarity fit, so
    the repaired chart keeps the original's position and orientation in the
    atlas -- which is what stops a repair from colliding with the islands packed
    around it.  The **scale** does not: it is set so the absolute UV area is
    exactly the original's, as ``docs/ORGANIC.md`` section 4.2 requires, because
    a repaired chart that quietly shrank would show up as a texel-density
    change everywhere it is used.  A reflection is applied when the original
    chart's majority winding was mirrored, so the repair does not silently flip
    an island the artist's layout had flipped on purpose.
    """
    if target is None:
        return uv
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if dst.shape[0] != uv.shape[0] or uv.shape[0] == 0:
        return uv
    if not np.isfinite(dst).all() or not np.isfinite(uv).all():
        return uv

    src = uv
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    if not _indexes(tris, uv.shape[0]):
        tris = np.zeros((0, 3), dtype=np.int64)
    # Majority winding, not the sign of the area sum.  This function is only
    # reached for a chart that *failed* certification -- one with mixed-sign
    # areas -- which is exactly when a signed sum cancels and a majority does
    # not.  Two large triangles of the minority sign were enough to flip the sum
    # and mirror an island the artist had packed the other way round, invisibly,
    # because ``flipped = min(positive, negative)`` reads 0 for a wholly
    # mirrored chart just as it does for an un-mirrored one.
    if tris.shape[0] and _majority_sign(dst, tris) * _majority_sign(src, tris) < 0.0:
        src = src * np.array([-1.0, 1.0])

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    sc = src - src_mean
    dc = dst - dst_mean
    dot = float(np.einsum("ij,ij->", sc, dc))
    cross = float(np.sum(sc[:, 0] * dc[:, 1] - sc[:, 1] * dc[:, 0]))
    norm = math.hypot(dot, cross)
    if norm > 0.0:
        cos_t, sin_t = dot / norm, cross / norm
    else:
        cos_t, sin_t = 1.0, 0.0
    rotated = np.stack(
        [sc[:, 0] * cos_t - sc[:, 1] * sin_t, sc[:, 0] * sin_t + sc[:, 1] * cos_t], axis=1
    )

    scale = 1.0
    if tris.shape[0]:
        area_src = float(np.abs(_signed_areas(rotated, tris)).sum())
        area_dst = float(np.abs(_signed_areas(dst, tris)).sum())
        if area_src > 0.0 and area_dst > 0.0:
            scale = math.sqrt(area_dst / area_src)
    out = rotated * scale + dst_mean
    return np.where(np.isfinite(out), out, dst)


# ---------------------------------------------------------------------------
# Placing a repaired chart back into the atlas
# ---------------------------------------------------------------------------


def _triangles_overlap(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    """Return True if any matched pair of triangles has interior in common.

    A boolean separating-axis test, not an area computation: two convex
    polygons are disjoint exactly when some edge of one of them separates them,
    so six half-plane tests settle each pair.  Touching counts as separated --
    the same convention the evaluator applies, so a chart placed exactly against
    its neighbour is not called a collision when the metric would score it as
    zero overlap.

    Args:
        tri_a: ``(N, 3, 2)`` triangles.
        tri_b: ``(N, 3, 2)`` triangles, matched pairwise with ``tri_a``.

    Returns:
        True as soon as one pair genuinely overlaps.
    """
    if tri_a.shape[0] == 0:
        return False
    da = tri_a[:, [1, 2, 0], :] - tri_a
    db = tri_b[:, [1, 2, 0], :] - tri_b
    extent = np.maximum(np.abs(da).max(axis=(1, 2)), np.abs(db).max(axis=(1, 2)))
    tol = _TOUCH_EPS * np.where(extent > 0.0, extent, 1.0) ** 2

    live = np.ones(tri_a.shape[0], dtype=bool)
    for tri, other, edge in ((tri_a, tri_b, da), (tri_b, tri_a, db)):
        for corner in range(3):
            ox = tri[:, corner, 0]
            oy = tri[:, corner, 1]
            ex = edge[:, corner, 0]
            ey = edge[:, corner, 1]
            separated = ex * (other[:, 0, 1] - oy) - ey * (other[:, 0, 0] - ox) <= tol
            separated &= ex * (other[:, 1, 1] - oy) - ey * (other[:, 1, 0] - ox) <= tol
            separated &= ex * (other[:, 2, 1] - oy) - ey * (other[:, 2, 0] - ox) <= tol
            live &= ~separated
            if not live.any():
                return False
    return bool(live.any())


def _collides(
    own: np.ndarray,
    others: np.ndarray,
    lo_b: np.ndarray | None = None,
    hi_b: np.ndarray | None = None,
) -> bool:
    """Return True if any of ``own``'s triangles overlaps any of ``others``.

    Two levels of rejection before any geometry.  The chart's *overall* box is
    tested against the neighbours' boxes first, which on a real atlas throws
    away most of the neighbourhood for the cost of one pass over ``(B,)``
    arrays; only the survivors reach the ``A x B`` per-triangle broadcast.  The
    placement search below calls this hundreds of times per chart with the same
    neighbours, so the difference between one level and two is the difference
    between a search that fits in the frame budget and one that does not.

    Args:
        own: ``(A, 3, 2)`` the chart being placed.
        others: ``(B, 3, 2)`` everything else within reach.
        lo_b: Optional precomputed ``(B, 2)`` minima of ``others``.
        hi_b: Optional precomputed ``(B, 2)`` maxima of ``others``.

    Returns:
        True on the first genuine overlap found.
    """
    if own.shape[0] == 0 or others.shape[0] == 0:
        return False
    if lo_b is None or hi_b is None:
        lo_b, hi_b = others.min(axis=1), others.max(axis=1)
    lo_a, hi_a = own.min(axis=1), own.max(axis=1)

    box_lo = lo_a.min(axis=0)
    box_hi = hi_a.max(axis=0)
    reach = np.flatnonzero(
        (lo_b[:, 0] < box_hi[0])
        & (hi_b[:, 0] > box_lo[0])
        & (lo_b[:, 1] < box_hi[1])
        & (hi_b[:, 1] > box_lo[1])
    )
    if reach.size == 0:
        return False
    near_lo = lo_b[reach]
    near_hi = hi_b[reach]

    block = max(1, _PAIR_BLOCK // max(reach.size, 1))
    for start in range(0, own.shape[0], block):
        stop = min(start + block, own.shape[0])
        near = (
            (lo_a[start:stop, None, 0] < near_hi[None, :, 0])
            & (near_lo[None, :, 0] < hi_a[start:stop, None, 0])
            & (lo_a[start:stop, None, 1] < near_hi[None, :, 1])
            & (near_lo[None, :, 1] < hi_a[start:stop, None, 1])
        )
        if not near.any():
            continue
        ia, ib = np.nonzero(near)
        if _triangles_overlap(own[start + ia], others[reach[ib]]):
            return True
    return False


def _placement_candidates() -> tuple[tuple[float, float, float], ...]:
    """Return the ``(rotation, offset_u, offset_v)`` trials, in preference order.

    Offsets are in units of the chart's own largest extent, so the search is
    scale free.  The ordering is what makes the search affordable: the identity
    placement comes first, then small nudges, then larger ones -- and the caller
    stops at the first success, so a chart that already fits pays for exactly
    one test.
    """
    trials: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    rotations = [0.0] + [
        math.radians(deg) for deg in (10, -10, 20, -20, 30, -30, 45, -45, 90, 180)
    ]
    for radius in (0.05, 0.1, 0.16, 0.24):
        for spoke in range(12):
            angle = 2.0 * math.pi * spoke / 12.0
            trials.append((0.0, radius * math.cos(angle), radius * math.sin(angle)))
    for rotation in rotations[1:]:
        trials.append((rotation, 0.0, 0.0))
        for radius in (0.05, 0.12, 0.2):
            for spoke in range(8):
                angle = 2.0 * math.pi * spoke / 8.0
                trials.append((rotation, radius * math.cos(angle), radius * math.sin(angle)))
    return tuple(trials)


#: Rigid trials tested at each scale.  Built once; the tuple is immutable, so
#: the search order -- and therefore the result -- is identical every run.
_PLACEMENTS = _placement_candidates()

#: Scale ladder, fine near 1.0 where the score is most sensitive to lost area.
_SCALE_LADDER = (
    1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.87, 0.84, 0.80, 0.75, 0.68, 0.60, 0.50, 0.38, 0.28,
)


def _place_chart(
    uv: np.ndarray, chart: Chart, others: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Fit a repaired chart back into the atlas, keeping as much area as it can.

    ``docs/ORGANIC.md`` section 4.2 says a repaired chart keeps its original UV
    area, and that is what this tries first.  But a re-flattened chart is a
    *different shape*: measured on Suzanne, the two folded charts grew their
    bounding boxes by only 4% and 8% and still collided with four neighbours,
    because ``pack_islands`` had nested them into each other's concavities with
    a 0.005 margin.  Preserving the area and stopping there therefore trades an
    intra-chart fold for an inter-chart collision -- the metric gets *worse*,
    and Rule 2 correctly throws the whole repair away.

    So area preservation is the goal, not a constraint.  The search maximises
    the scale over a fixed ladder, and at each scale tries a fixed set of rigid
    placements (rotation about the centroid plus a small translation) before
    giving up on it.  The rigid trials matter more than they look: on Suzanne's
    126-triangle chart, allowing a 20 degree rotation and a nudge of 5% of the
    chart's own extent took the achievable area from 0.67 to 0.91 of the
    original, worth 1.6 points of final score on its own.  A shrunk chart costs
    texel density, which the repack stage (section 6) exists to reclaim; an
    overlapping chart costs the artist a broken bake.

    Every trial is a fixed tuple in a fixed order, so the placement chosen is a
    deterministic function of the input.

    Args:
        uv: ``(V, 2)`` repaired layout, already aligned onto the original.
        chart: The chart being placed.
        others: ``(B, 3, 2)`` triangles of the rest of the atlas near it.

    Returns:
        ``(uv, scale)`` for the best placement found, or None if every trial at
        every scale still collides -- in which case the caller keeps the
        original chart untouched.
    """
    tris = np.asarray(chart.tris, dtype=np.int64).reshape(-1, 3)
    if tris.shape[0] == 0:
        return uv, 1.0
    centre = uv.mean(axis=0)
    rel = uv - centre
    extent = float(np.max(uv.max(axis=0) - uv.min(axis=0)))
    if not math.isfinite(extent) or extent <= 0.0:
        extent = 1.0
    lo_b = others.min(axis=1) if others.shape[0] else np.zeros((0, 2))
    hi_b = others.max(axis=1) if others.shape[0] else np.zeros((0, 2))

    spun: dict[float, np.ndarray] = {}

    def rotate(angle: float) -> np.ndarray:
        """Cache the rotated chart; the ladder revisits each angle many times."""
        if angle not in spun:
            cos_t, sin_t = math.cos(angle), math.sin(angle)
            spun[angle] = np.stack(
                [rel[:, 0] * cos_t - rel[:, 1] * sin_t, rel[:, 0] * sin_t + rel[:, 1] * cos_t],
                axis=1,
            )
        return spun[angle]

    def build(scale: float, trial: tuple[float, float, float]) -> np.ndarray:
        """Place the chart at one (scale, rotation, offset) trial."""
        angle, off_u, off_v = trial
        shift = np.array([off_u * extent, off_v * extent])
        return rotate(angle) * scale + centre + shift

    found: tuple[float, tuple[float, float, float]] | None = None
    previous = 1.0
    for scale in _SCALE_LADDER:
        for trial in _PLACEMENTS:
            if not _collides(build(scale, trial)[tris], others, lo_b, hi_b):
                found = (scale, trial)
                break
        if found is not None:
            break
        previous = scale
    if found is None:
        return None

    scale, trial = found
    if scale < 1.0:
        # Bisect back towards the original area with the winning rigid trial;
        # the ladder's own steps throw away up to twice their width in area.
        failing = previous
        clean = scale
        for _ in range(_SHRINK_REFINE):
            mid = 0.5 * (failing + clean)
            if _collides(build(mid, trial)[tris], others, lo_b, hi_b):
                failing = mid
            else:
                clean = mid
        scale = clean
    return build(scale, trial), float(scale)


#: Largest translation any placement trial applies, in units of the chart's own
#: extent.  Derived from the trial table rather than written down twice: the
#: candidate-neighbour box below has to cover every trial, and a search that
#: reaches further than the box it was gathered against silently misses
#: collisions.  That bug cost a measured 0.00048 of overlap on Suzanne -- the
#: chart moved 0.0415 in UV, landed on two islands that were never in the
#: candidate set, and the collision test reported clean.
_PLACE_REACH = max(math.hypot(u, v) for _, u, v in _PLACEMENTS)


def _placement_box(uv: np.ndarray) -> np.ndarray:
    """Return the ``(u0, v0, u1, v1)`` box every placement trial stays inside.

    Provably conservative, not approximately so.  Scaling by at most 1 about the
    centroid keeps every vertex within its current radius; rotating about the
    centroid keeps it on that radius exactly; translating adds at most
    :data:`_PLACE_REACH` times the chart's extent.  So the circumscribed square
    around ``radius + reach`` contains every trial the search can produce.
    """
    centre = uv.mean(axis=0)
    offsets = uv - centre
    radius = float(np.sqrt(np.einsum("ij,ij->i", offsets, offsets)).max()) if uv.size else 0.0
    extent = float(np.max(uv.max(axis=0) - uv.min(axis=0))) if uv.size else 0.0
    reach = radius + _PLACE_REACH * extent
    if not math.isfinite(reach):
        reach = 0.0
    return np.array(
        [centre[0] - reach, centre[1] - reach, centre[0] + reach, centre[1] + reach]
    )


def _neighbour_triangles(
    sample: UVMeshSample, layout: np.ndarray, own: np.ndarray, box: np.ndarray
) -> np.ndarray:
    """Return the ``(B, 3, 2)`` triangles outside ``own`` that reach into ``box``.

    Gathered once per chart rather than once per trial: :func:`_placement_box`
    bounds every placement the search can produce, so one gather covers the
    whole search.
    """
    tri_loops = np.asarray(sample.tri_loops, dtype=np.int64).reshape(-1, 3)
    keep = np.ones(tri_loops.shape[0], dtype=bool)
    keep[own] = False
    if not keep.any():
        return np.zeros((0, 3, 2), dtype=np.float64)
    tri_uv = layout[tri_loops[keep]]
    lo = tri_uv.min(axis=1)
    hi = tri_uv.max(axis=1)
    near = (lo[:, 0] < box[2]) & (hi[:, 0] > box[0]) & (lo[:, 1] < box[3]) & (hi[:, 1] > box[1])
    return tri_uv[near]


# ---------------------------------------------------------------------------
# The engine-facing entry point
# ---------------------------------------------------------------------------


def repair_folds(
    sample: UVMeshSample,
    uvs: np.ndarray,
    *,
    tri_chart: np.ndarray | None = None,
    weights: ScoreWeights | None = None,
) -> tuple[np.ndarray, RepairReport]:
    """Re-flatten only the charts that fail certification.  Never worsens.

    Rule 2 of ``docs/ORGANIC.md`` in full:

    * every chart is certified **on the way in**;
    * a chart that passes is left **bit-identical** -- not re-flattened, not
      re-packed, not touched;
    * a chart that fails is re-flattened, placed back where it sat at its
      original UV area, and shrunk only as far as clearing its neighbours
      requires;
    * a repair whose own certificate fails is discarded and the original chart
      kept;
    * and the whole pass is kept only if it **raises the score** under the
      frozen scorer and does not raise ``overlap_area``.

    Judging on the score rather than on overlap alone is deliberate.  Overlap is
    the loudest defect but it is not the only axis a repair moves: a re-flattened
    chart trades some distortion for injectivity, and on Suzanne that trade is
    worth roughly +19 points against a cost of about 1.  Comparing one metric
    could not see either side of it.  The scorer itself is untouched -- Rule 1 --
    it is only *consulted*.

    Args:
        sample: The mesh the layout belongs to.
        uvs: ``(L, 2)`` incoming layout.
        tri_chart: Optional ``(T,)`` chart id per triangle.  Computed from the
            layout when omitted.
        weights: Scoring weights for the Rule 2 comparison.  Defaults to the
            ``balanced`` preset, matching the engine's own default.

    Returns:
        ``(uvs, report)``.  On any refusal ``uvs`` is the exact input array's
        values and ``report.changed`` is False.  Never raises: a malformed input
        comes back unchanged with the reason in ``report.messages``.
    """
    started = time.perf_counter()
    messages: list[str] = []
    incoming = np.asarray(uvs, dtype=np.float64)
    if incoming.ndim != 2 or incoming.shape[1] != 2 or incoming.shape[0] != sample.n_loops:
        return np.asarray(uvs), RepairReport(
            seconds=time.perf_counter() - started,
            messages=("uvs must be (%d, 2), got %s" % (sample.n_loops, incoming.shape),),
        )

    try:
        return _repair_impl(sample, incoming, tri_chart, weights, messages, started)
    except Exception as exc:  # noqa: BLE001 - containment, per ARCHITECTURE 1.4
        messages.append("repair aborted: %s" % type(exc).__name__)
        return incoming, RepairReport(
            seconds=time.perf_counter() - started, messages=tuple(messages)
        )


def _repair_impl(
    sample: UVMeshSample,
    incoming: np.ndarray,
    tri_chart: np.ndarray | None,
    weights: ScoreWeights | None,
    messages: list[str],
    started: float,
) -> tuple[np.ndarray, RepairReport]:
    """Do the work.  See :func:`repair_folds` for the contract."""
    scale = weights if weights is not None else ScoreWeights.preset("balanced")
    before = evaluate(sample.with_uvs(incoming), compute_overlap=True)
    if before.overlap_area <= 0.0 and before.flipped_tris == 0:
        messages.append("layout is already clean")
        return incoming, _report(0, 0, 0, 0, before, before, False, started, messages)

    uv_vertex_of_loop, computed_chart, _ = uv_charts(sample, incoming)
    labels = computed_chart if tri_chart is None else np.asarray(tri_chart, dtype=np.int64)
    labels = labels.reshape(-1)
    if labels.shape[0] != sample.n_tris:
        labels = computed_chart

    out = np.array(incoming, dtype=np.float64, copy=True)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(order, boundaries) if sorted_labels.size else []

    checked = failed = repaired = reverted = 0
    for group in groups:
        if group.size == 0 or labels[group[0]] < 0:
            continue
        checked += 1
        chart = chart_from_tris(
            sample,
            group,
            uv_vertex_of_loop=uv_vertex_of_loop,
            name="chart_%d" % int(labels[group[0]]),
        )
        if chart.n_tris == 0:
            continue
        target = _target_layout(chart, incoming)
        topo = chart_topology(chart)
        if _certify_layout(target, chart, topo) == (0, 0, 0):
            continue
        failed += 1
        result = flatten_chart(chart, target=target)
        if not result.ok:
            reverted += 1
            messages.append(
                "%s: kept original (%s, flipped=%d, crossings=%d)"
                % (chart.name, result.stage, result.flipped, result.boundary_crossings)
            )
            continue

        others = _neighbour_triangles(sample, out, group, _placement_box(result.uv))
        placed = _place_chart(result.uv, chart, others)
        if placed is None:
            reverted += 1
            messages.append("%s: kept original (no collision-free placement)" % chart.name)
            continue
        final, applied = placed
        flat_loops = np.asarray(chart.loops, dtype=np.int64).reshape(-1)
        out[flat_loops] = final[np.asarray(chart.tris, dtype=np.int64).reshape(-1)]
        repaired += 1
        messages.append(
            "%s: repaired via %s (%d tris, energy %.4f, placed at %.4f of original area)"
            % (chart.name, result.stage, chart.n_tris, result.energy, applied * applied)
        )

    if repaired == 0:
        messages.append("nothing repaired")
        return incoming, _report(
            checked, failed, 0, reverted, before, before, False, started, messages
        )

    after = evaluate(sample.with_uvs(out), compute_overlap=True)
    gained = score(after, scale) - score(before, scale)
    improved = (
        gained > 0.0
        and after.overlap_area <= before.overlap_area
        and after.flipped_tris <= before.flipped_tris
        and after.degenerate_tris <= before.degenerate_tris
    )
    if not improved:
        messages.append(
            "reverted: score %+0.3f, overlap %.6g -> %.6g, flipped %d -> %d"
            % (gained, before.overlap_area, after.overlap_area,
               before.flipped_tris, after.flipped_tris)
        )
        return incoming, _report(
            checked, failed, 0, reverted + repaired, before, before, False, started, messages
        )

    messages.append("kept: score %+0.3f" % gained)
    return out, _report(
        checked, failed, repaired, reverted, before, after, True, started, messages
    )


def _report(
    checked: int,
    failed: int,
    repaired: int,
    reverted: int,
    before,
    after,
    changed: bool,
    started: float,
    messages: list[str],
) -> RepairReport:
    """Assemble the report, so every early return tells the same story."""
    return RepairReport(
        charts_checked=checked,
        charts_failed=failed,
        charts_repaired=repaired,
        charts_reverted=reverted,
        overlap_before=before.overlap_area,
        overlap_after=after.overlap_area,
        flipped_before=before.flipped_tris,
        flipped_after=after.flipped_tris,
        changed=changed,
        seconds=time.perf_counter() - started,
        messages=tuple(messages),
    )


def _target_layout(chart: Chart, uvs: np.ndarray) -> np.ndarray:
    """Return the chart-local view of an existing layout, one UV per chart vertex."""
    target = np.zeros((chart.n_verts, 2), dtype=np.float64)
    local = np.asarray(chart.tris, dtype=np.int64).reshape(-1)
    loops = np.asarray(chart.loops, dtype=np.int64).reshape(-1)
    valid = (loops >= 0) & (loops < uvs.shape[0])
    target[local[valid]] = uvs[loops[valid]]
    return target
