"""Certify that a UV layout is injective -- and *prove* it when the chart is a disk.

This module answers one question: does the parametrisation fold over itself?
It is the runtime certificate the fold-repair stage (``core/flatten.py``) gates
on, and it exists because the obvious test -- counting triangles with a negative
signed UV area -- is blind to the defect that actually costs this project points.

The measured case that shapes the whole module
----------------------------------------------

On Suzanne's winning layout (``engine.generate(count=10, seed=1)``, Blender
5.1.2) the evaluator reports ``overlap_area = 0.0013420951`` and
``flipped_tris = 0``.  All 968 signed UV areas are **positive**: the map is
locally injective at every single triangle.  Of the 16 overlapping triangle
pairs, **none share an edge** and 12 share no vertex at all.  So the defect is
not a hinge folding at a crease, which a flip test would catch; it is one region
of a chart lapping over a *distant* region of the same chart.  The only thing
that sees it is the chart **boundary**, which self-intersects.
See ``docs/ORGANIC.md`` section 0.3.

The theorem this rests on
-------------------------

A locally injective piecewise-linear map of a **disk** whose boundary image is a
**simple closed curve** is globally injective.  Therefore, for a disk chart::

    (no flipped triangles) and (no degenerate triangles)
        and (boundary has no self-intersection)
    ==> the chart is globally injective

That is a *proof*, not a heuristic -- which is why :func:`boundary_crossings`
uses exact sign predicates rather than a tolerance, and why
:func:`certify` reports the chart's Euler characteristic-derived disk status
separately.  Precisely where the proof stops:

* **Non-disk charts.**  An annulus (a chart containing one of Suzanne's eye
  sockets, say) can have a locally injective map and two perfectly simple
  boundary loops and still lap over itself, because the theorem needs a disk.
  So can a **bowtie**, which is the trap: two lobes joined at one vertex have
  Euler characteristic 1 *and* a single boundary cycle, so the obvious disk test
  passes, and one lobe can sit inside the other touching only at the pinch with
  nothing crossing anywhere.  ``non_disk_charts`` counts every chart failing any
  of the four conditions in ``_chart_topology``, and ``globally_injective`` is
  never True while it is non-zero.
* **Nesting between charts.**  If chart A's image lies wholly *inside* chart
  B's, no boundary crossing occurs anywhere and the layout is still not
  injective.  Boundary crossings cannot see containment.  This is why
  ``globally_injective`` additionally requires ``overlap_area == 0.0``, which is
  the evaluator's exact convex-clip measurement and does see it.  With
  ``compute_overlap=False`` that evidence is missing and the flag degrades to a
  *necessary* condition -- still exact for a single disk chart, which is the
  case ``repair_folds`` uses it for, but no longer a proof for a whole atlas.
* **Tangential contact.**  Two boundary curves that touch without crossing are
  not counted (see the convention below).  A chart folded so its boundary lies
  exactly along another part of its boundary therefore has zero crossings and a
  positive ``overlap_area``.  Carrying both numbers is what makes the pair
  complete.

Conventions, all of them deliberate
-----------------------------------

**Exactly touching is NOT crossing.**  Two segments count as a crossing only
when they cross *transversally* -- each strictly separates the other's
endpoints.  Shared endpoints, T-junctions and collinear overlap all count as
zero.  This matches the evaluator's overlap convention exactly (see
``_overlap_area`` and ``_pair_intersection_area`` in ``core/evaluator.py``:
triangles sharing an edge are separated by that edge and contribute no area), so
the two measurements can never contradict each other on a clean layout.  It is
also what makes the consecutive segments of a boundary polyline -- which always
share an endpoint -- cost nothing and need no special case.

**Exact predicates.**  A crossing is decided by the sign of ``orient2d``, never
by a tolerance.  The float64 evaluation is accepted only when Shewchuk's static
forward error bound proves its sign; otherwise the determinant is recomputed in
exact integer arithmetic (every float64 is an integer times a power of two, so
scaling all six coordinates to a common power of two is lossless).  A tolerance
here would make the certificate a matter of taste, and the whole point of the
certificate is that it is not.

**Welding.**  Loops are welded into UV vertices when they share a mesh vertex
*and* an exactly equal UV.  Blender computes a solution per UV vertex and writes
the identical float to each of its loops, so exact equality is the right test on
real unwrapper output; a genuine difference is a genuine crack in the chart and
is reported as boundary, which is the honest reading.  Two different mesh
vertices never weld, so islands that merely touch in UV stay separate -- the
same rule the evaluator uses for ``island_count``.

**Charts.**  With no ``tri_chart`` supplied, a chart is a UV island: a connected
component of the welded UV graph.  A caller mid-merge (``core/charts.py``) can
supply its own labelling, in which case a "chart" may span two disjoint UV
regions -- and will then correctly fail to be a disk.

Complexity
----------

:func:`boundary_crossings` must not be quadratic: candidate segment pairs come
from a uniform grid sized to the median segment, so the cost is linear in the
segment count times the local density.  Budget and measurement: 100k boundary
segments in well under 500 ms.

Nothing here imports :mod:`bpy`, and :func:`certify` never raises -- on
catastrophic input it returns a report that refuses to certify, which is the
safe direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ``_connected_components`` is deliberately shared with the evaluator rather
# than re-implemented.  It is a pure, side-effect-free vectorised union-find,
# and a second copy of a subtle algorithm is a second place for it to be subtly
# wrong; the independent check on chart labelling lives in ``tests/reference.py``
# where it belongs, as a naive union-find that shares no code at all.
from .evaluator import _connected_components, evaluate
from .meshdata import UVMeshSample

__all__ = [
    "InjectivityReport",
    "certify",
    "chart_labels",
    "boundary_loops",
    "boundary_loops_by_chart",
    "boundary_segments",
    "boundary_crossings",
    "crossing_pairs",
    "crossings_by_chart",
]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Relative UV-area floor below which a triangle is UV-degenerate, as a fraction
#: of the layout's own mean absolute UV triangle area.  Same rule and same
#: constant as :func:`unwrap_variations.core.meshdata.degenerate_mask` and the
#: evaluator, so the three never disagree about which triangles collapsed.  It
#: has to be relative: an absolute 1e-12 calls every triangle of a 0.1 mm CAD
#: import degenerate.
_DEGENERATE_EPS = 1e-12

#: Grid guards for the broad phase.  A pathological layout (every boundary
#: segment stacked on one spot) would otherwise generate a quadratic number of
#: candidate pairs.  Hitting either cap can only ever *under*-report crossings,
#: and a layout dense enough to hit them is already flagged by ``overlap_area``,
#: which :func:`certify` also requires to be zero -- so the certificate stays
#: sound in the direction that matters.
_MAX_INSERTS = 8_000_000
_MAX_PAIRS = 4_000_000
_PAIR_CHUNK = 1_000_000

#: Finest grid the broad phase will build, per axis.  Nothing is *allocated*
#: per cell -- the grid exists only as sorted (cell, segment) insertions -- so
#: this is purely an overflow guard on the packed cell id, and it should be
#: generous.  It was 4096 (copied from the evaluator's triangle hash) and that
#: turned out to be the wrong number for a boundary: at 400k segments on a
#: smooth curve the cell was clamped to 35 segments wide, which put 6.8M
#: candidate pairs through the predicate and cost 905 ms against 53 ms at 100k
#: -- superlinear for no reason but the cap.  At 65536 the same case sizes the
#: cell to the segments themselves and lands at 205 ms.
_MAX_GRID_DIM = 65_536

#: Shewchuk's static forward error bound for a float64 ``orient2d`` evaluation
#: (*Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
#: Predicates*, 1997), ``ccwerrboundA``.  When ``|det|`` clears this multiple of
#: the summed magnitudes of the two products, the float64 sign is certain.
_ORIENT_EPS = 2.0**-53
_ORIENT_ERRBOUND = (3.0 + 16.0 * _ORIENT_EPS) * _ORIENT_EPS


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectivityReport:
    """The certificate: what was checked, what failed, and whether it is proved.

    Attributes:
        flipped: Triangles whose signed UV area has the sign *opposite* to their
            own chart's majority.  A chart that is uniformly mirrored is still
            injective, so it is not a defect and is not counted -- which is a
            deliberate difference from
            :attr:`~unwrap_variations.core.metrics.UVMetrics.flipped_tris`,
            which counts every negative signed area against the 3D winding
            because a mirrored island breaks tangent-space normal maps.  The two
            answer different questions and are expected to differ on a mirrored
            island and nowhere else.
        degenerate: Triangles with no usable UV area (below the relative floor,
            or with a non-finite UV corner).  Only **UV** degeneracy counts
            here: a sliver in 3D still maps injectively into the plane, so
            unlike ``UVMetrics.degenerate_tris`` -- which also excludes
            3D-degenerate triangles from the distortion means -- it is not a
            defect of the map's injectivity.
        boundary_crossings: Transversal self-intersections of the chart boundary
            polylines, counted within each chart.  Shared endpoints, touching
            and collinear overlap are **not** crossings.
        overlap_area: The evaluator's ``overlap_area`` for the same sample, so
            the certificate and the score can never disagree.  ``0.0`` when
            clean, and ``0.0`` (not measured) when ``compute_overlap=False``.
        locally_injective: ``flipped == 0 and degenerate == 0``.  The map has a
            well-defined, orientation-consistent inverse in a neighbourhood of
            every point.
        globally_injective: A **proof** that no two points of the surface share
            a UV.  True only when the map is locally injective, every chart
            boundary is simple, every chart is a disk, and the exact overlap
            measurement is zero.  See the module docstring for why the last two
            conjuncts are load-bearing and where the guarantee stops.
        charts_checked: Number of distinct charts examined.  Equals
            ``UVMetrics.island_count`` when the charts were derived here.
        non_disk_charts: Charts whose triangulated UV complex is not a disk
            (Euler characteristic ``V - E + F != 1``, or a boundary that is not
            a single closed loop).  The simple-boundary theorem does not apply
            to these, so they can never be certified.  This field is an addition
            to the signature in ``docs/ORGANIC.md`` section 4.1; without it the
            report would have to either claim a proof it does not have or hide
            the reason it cannot.
    """

    flipped: int = 0
    degenerate: int = 0
    boundary_crossings: int = 0
    overlap_area: float = 0.0
    locally_injective: bool = True
    globally_injective: bool = True
    charts_checked: int = 0
    non_disk_charts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of every field."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "InjectivityReport":
        """Rebuild a report from :meth:`to_dict` output, tolerating missing keys."""
        if not data:
            return cls()
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def summary(self) -> str:
        """Return a compact one-line description, for logs and reports."""
        verdict = "injective" if self.globally_injective else "NOT injective"
        return (
            "%s: charts=%d flipped=%d degenerate=%d crossings=%d "
            "non_disk=%d overlap=%.6g"
            % (
                verdict,
                self.charts_checked,
                self.flipped,
                self.degenerate,
                self.boundary_crossings,
                self.non_disk_charts,
                self.overlap_area,
            )
        )


#: The report returned when certification could not be completed at all.  It
#: refuses to certify rather than guessing, because a false certificate is the
#: one failure mode of this module that would silently ship a broken bake.
_REFUSED = InjectivityReport(locally_injective=False, globally_injective=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def certify(
    sample: UVMeshSample,
    *,
    tri_chart: np.ndarray | None = None,
    compute_overlap: bool = True,
) -> InjectivityReport:
    """Certify the injectivity of ``sample``'s UV layout.

    Args:
        sample: The mesh plus the UV layer to certify.
        tri_chart: Optional ``(T,)`` integer chart label per triangle.  When
            omitted, charts are derived as UV islands by :func:`chart_labels`.
            Supply your own when certifying a *proposed* grouping -- a merge
            candidate, say -- that the current UVs do not yet reflect.
        compute_overlap: When False the evaluator's exact overlap measurement is
            skipped, ``overlap_area`` is reported as ``0.0`` and
            ``globally_injective`` loses its defence against one chart nesting
            inside another.  It remains exact for a single disk chart.  This
            argument is an addition to the signature in ``docs/ORGANIC.md``
            section 4.1; it defaults to the contract's behaviour.

    Returns:
        An :class:`InjectivityReport`.  This function never raises: on
        catastrophic input it returns a report with both injectivity flags
        False, because refusing to certify is the safe direction.
    """
    try:
        return _certify_impl(sample, tri_chart, bool(compute_overlap))
    except Exception:  # noqa: BLE001 - containment is the whole point
        return _REFUSED


def _certify_impl(
    sample: UVMeshSample,
    tri_chart: np.ndarray | None,
    compute_overlap: bool,
) -> InjectivityReport:
    """Do the real work.  See :func:`certify` for the contract."""
    data = _Clean(sample)
    uvs, tri_loops, dropped = data.uvs, data.tri_loops, data.n_dropped
    n_tris = tri_loops.shape[0]
    if n_tris == 0:
        # Nothing to fold.  An empty layout is vacuously injective, but a sample
        # whose triangles were all unusable is not certifiable.
        if dropped:
            return InjectivityReport(
                degenerate=dropped, locally_injective=False, globally_injective=False
            )
        return InjectivityReport()

    labels = _resolve_charts(sample, tri_chart, data)
    n_charts = int(labels.max()) + 1 if labels.size else 0

    signed, finite = _signed_uv_areas(uvs, tri_loops)
    degenerate_mask = ~finite | (np.abs(signed) <= _uv_area_floor(signed, finite))
    degenerate = int(np.count_nonzero(degenerate_mask)) + dropped
    flipped = _count_flipped(signed, ~degenerate_mask, labels, n_charts)

    topo = _chart_topology(sample, tri_loops, labels, n_charts)
    crossings_per_chart = _crossings_by_chart(uvs, topo.loops, topo.loop_chart, n_charts)
    crossings = int(crossings_per_chart.sum())

    overlap = 0.0
    if compute_overlap:
        overlap = float(evaluate(sample, compute_overlap=True).overlap_area)

    locally = flipped == 0 and degenerate == 0
    globally = (
        locally and crossings == 0 and topo.non_disk_charts == 0 and not (overlap > 0.0)
    )
    return InjectivityReport(
        flipped=flipped,
        degenerate=degenerate,
        boundary_crossings=crossings,
        overlap_area=overlap,
        locally_injective=locally,
        globally_injective=globally,
        charts_checked=n_charts,
        non_disk_charts=topo.non_disk_charts,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_labels(sample: UVMeshSample) -> np.ndarray:
    """Return a ``(T,)`` int64 chart label per triangle, one per UV island.

    Triangles join the same island when they share a welded UV vertex, i.e. a
    mesh vertex carrying an identical UV.

    Charts are numbered **in the order their first triangle appears**, so chart
    0 always contains triangle 0.  That rule is deliberate: any numbering
    derived from internal node ids is an artefact of how the welding happened to
    sort its keys, and would make "chart 6 is folded" mean different things to
    two implementations of the same partition.  Ordering by first triangle is
    canonical, cheap and means the same thing to the naive oracle in
    ``tests/reference.py``, which shares none of the machinery.

    Args:
        sample: The mesh plus the UV layer to partition.

    Returns:
        ``(T,)`` int64 labels in ``0 .. n_charts - 1``.  Empty when the sample
        has no usable triangles.
    """
    tri_loops = _Clean(sample).tri_loops
    if tri_loops.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    node, n_nodes = _weld_uv_vertices(sample)
    corners = node[tri_loops]
    labels = _connected_components(n_nodes, corners.reshape(-1), corners[:, [1, 2, 0]].reshape(-1))
    return _compact_by_first_appearance(labels[corners[:, 0]])


def _compact_by_first_appearance(raw: np.ndarray) -> np.ndarray:
    """Renumber arbitrary labels to ``0 .. k-1`` in order of first appearance."""
    flat = np.asarray(raw).reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return np.ascontiguousarray(rank[inverse.reshape(-1)], dtype=np.int64)


def _resolve_charts(
    sample: UVMeshSample, tri_chart: np.ndarray | None, data: "_Clean"
) -> np.ndarray:
    """Return compacted ``(T,)`` chart labels, deriving them when not supplied.

    A caller-supplied labelling is accepted as-is apart from being renumbered by
    first appearance, exactly as a derived one is.  A wrong *length* is a caller
    bug that would silently mis-attribute every boundary, so it falls back to
    derived islands rather than guessing an alignment.
    """
    if tri_chart is None:
        return chart_labels(sample)
    given = np.asarray(tri_chart).reshape(-1)
    if given.shape[0] == data.n_total and data.kept is not None:
        given = given[data.kept]
    if given.shape[0] != data.tri_loops.shape[0]:
        return chart_labels(sample)
    return _compact_by_first_appearance(given)


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------


class _ChartTopology:
    """Boundary loops plus the per-chart topology they were derived from.

    Attributes:
        loops: One closed polyline of loop indices per boundary cycle.
        loop_chart: ``(len(loops),)`` chart label of each cycle.
        non_disk_charts: How many charts fail the four-part disk test.
        euler: ``(n_charts,)`` Euler characteristic ``V - E + F``.  Only one of
            the four disk conditions, kept whole because "chart 9 has chi = 0"
            is the first thing worth knowing when a chart will not certify and
            recomputing it from outside would need the whole half-edge table
            again.
    """

    __slots__ = ("loops", "loop_chart", "non_disk_charts", "euler")

    def __init__(
        self,
        loops: list[list[int]],
        loop_chart: np.ndarray,
        non_disk_charts: int,
        euler: np.ndarray,
    ) -> None:
        """Store the extracted boundary and the disk verdict behind it."""
        self.loops = loops
        self.loop_chart = loop_chart
        self.non_disk_charts = non_disk_charts
        self.euler = euler


def boundary_loops(
    sample: UVMeshSample, tri_chart: np.ndarray | None = None
) -> list[list[int]]:
    """Return every chart's boundary as a closed polyline of loop indices.

    A boundary half-edge is a directed triangle edge, over welded UV vertices,
    whose opposite direction does not occur anywhere else in the same chart.
    Those half-edges are chained head-to-tail into closed cycles.  On a
    well-formed chart the chaining is forced; where a UV vertex has several
    outgoing boundary half-edges (a pinch point, or a non-manifold chart) an
    unused one is taken in index order, which is deterministic and consumes
    every half-edge exactly once.

    Args:
        sample: The mesh plus the UV layer.
        tri_chart: Optional ``(T,)`` chart label per triangle; UV islands are
            derived when omitted.

    Returns:
        A list of closed polylines.  Each entry is a list of **loop indices**
        into ``sample.uvs``, one per boundary vertex, ordered along the
        boundary; the closing segment from the last back to the first is
        implied.  Polylines shorter than three vertices are dropped -- they
        enclose no area and can produce no crossing.  Self-loop edges (both
        endpoints welded to the same UV vertex, which only a collapsed triangle
        produces) are excluded.

        The output is **canonical**: each cycle is rotated to start at its
        smallest loop index and the loops are ordered by ``(chart, first
        index)``.  A cycle has no intrinsic starting point, so without that the
        segment numbering would depend on which half-edge the walk happened to
        pick up first -- and the naive oracle, which walks the same cycles in a
        different order, could not be compared against segment for segment.
    """
    data = _Clean(sample)
    tri_loops = data.tri_loops
    if tri_loops.shape[0] == 0:
        return []
    labels = _resolve_charts(sample, tri_chart, data)
    n_charts = int(labels.max()) + 1 if labels.size else 0
    return _chart_topology(sample, tri_loops, labels, n_charts).loops


def boundary_loops_by_chart(
    sample: UVMeshSample, tri_chart: np.ndarray | None = None
) -> dict[int, list[list[int]]]:
    """Return :func:`boundary_loops` grouped by the chart each loop belongs to.

    Args:
        sample: The mesh plus the UV layer.
        tri_chart: Optional ``(T,)`` chart label per triangle.

    Returns:
        ``{chart_label: [loop, ...]}``.  Charts with no boundary at all are
        absent; a chart with two entries is an annulus and cannot be certified.
    """
    data = _Clean(sample)
    tri_loops = data.tri_loops
    if tri_loops.shape[0] == 0:
        return {}
    labels = _resolve_charts(sample, tri_chart, data)
    n_charts = int(labels.max()) + 1 if labels.size else 0
    topo = _chart_topology(sample, tri_loops, labels, n_charts)
    grouped: dict[int, list[list[int]]] = {}
    for loop, chart in zip(topo.loops, topo.loop_chart.tolist(), strict=False):
        grouped.setdefault(int(chart), []).append(loop)
    return grouped


def crossings_by_chart(
    sample: UVMeshSample, tri_chart: np.ndarray | None = None
) -> dict[int, int]:
    """Return the boundary self-intersection count of each chart.

    This is what localises a failure: :func:`certify` reports one total, and
    this says which chart owns it.

    Args:
        sample: The mesh plus the UV layer.
        tri_chart: Optional ``(T,)`` chart label per triangle.

    Returns:
        ``{chart_label: crossings}`` for every chart, including the clean ones,
        so the caller can see the shape of the whole layout at once.
    """
    data = _Clean(sample)
    tri_loops = data.tri_loops
    if tri_loops.shape[0] == 0:
        return {}
    labels = _resolve_charts(sample, tri_chart, data)
    n_charts = int(labels.max()) + 1 if labels.size else 0
    topo = _chart_topology(sample, tri_loops, labels, n_charts)
    counts = _crossings_by_chart(
        np.asarray(sample.uvs, dtype=np.float64).reshape(-1, 2),
        topo.loops,
        topo.loop_chart,
        n_charts,
    )
    return {i: int(counts[i]) for i in range(n_charts)}


def _chart_topology(
    sample: UVMeshSample,
    tri_loops: np.ndarray,
    labels: np.ndarray,
    n_charts: int,
) -> _ChartTopology:
    """Extract boundary loops and each chart's disk status in one pass.

    Both answers come from the same half-edge table, which is the point: a
    boundary loop count that disagreed with the Euler characteristic would mean
    the two were built from different notions of "edge".

    A chart counts as a disk only when **all four** of these hold, because the
    simple-boundary theorem needs a genuine disk and each condition rules out a
    shape that would otherwise sneak past:

    * ``chi = V - E + F == 1`` -- the arithmetic invariant;
    * exactly one boundary loop -- an annulus (a chart holding one of Suzanne's
      eye sockets) has ``chi = 0``, but the loop count says *which* way it
      failed and is what a repair stage needs;
    * no edge with more than two incident triangles -- a non-manifold fin has no
      well-defined boundary walk at all;
    * no boundary loop visiting the same UV vertex twice.  This is the one that
      is easy to miss: a **bowtie** -- two lobes joined at a single vertex --
      has ``chi = 1`` *and* a single boundary cycle, and is still not a disk.
      Its boundary passes through the shared vertex twice, so it is not a simple
      closed curve and the theorem simply does not apply: one lobe can sit
      wholly inside the other, touching only at the pinch, with no transversal
      crossing anywhere for :func:`boundary_crossings` to find.  The test is
      deliberately conservative -- it also refuses a bowtie whose lobes lie side
      by side, which *is* injective -- because refusing to certify something
      valid costs a needless re-flatten, while certifying something invalid
      ships a broken bake.
    """
    node, n_nodes = _weld_uv_vertices(sample)
    corners = node[tri_loops]  # (T, 3) welded UV vertex per triangle corner
    n_tris = corners.shape[0]

    tail = corners.reshape(-1)
    head = corners[:, [1, 2, 0]].reshape(-1)
    tail_loop = tri_loops.reshape(-1)
    edge_chart = np.repeat(labels, 3)

    alive = tail != head
    lo = np.minimum(tail, head)
    hi = np.maximum(tail, head)
    forward = (tail <= head).astype(np.int8)

    # Group the half-edges by (chart, undirected edge).  lexsort's last key is
    # the primary one.
    order = np.lexsort((lo * n_nodes + hi, edge_chart))
    s_lo = lo[order]
    s_hi = hi[order]
    s_chart = edge_chart[order]
    s_forward = forward[order]
    s_alive = alive[order]

    run_id, n_runs = _run_ids(s_chart, s_lo, s_hi)
    live = s_alive
    n_fwd = np.bincount(run_id[live & (s_forward == 1)], minlength=n_runs)
    n_bwd = np.bincount(run_id[live & (s_forward == 0)], minlength=n_runs)
    opposite = np.where(s_forward == 1, n_bwd[run_id], n_fwd[run_id])
    is_boundary = live & (opposite == 0)

    # Euler characteristic per chart, over the welded complex.
    faces = np.bincount(labels, minlength=n_charts).astype(np.int64)
    run_chart = s_chart[_run_starts(run_id, n_runs)]
    run_live = np.bincount(run_id[live], minlength=n_runs) > 0
    edges = np.bincount(run_chart[run_live], minlength=n_charts).astype(np.int64)
    vert_key = np.unique(np.repeat(labels, 3).astype(np.int64) * n_nodes + corners.reshape(-1))
    verts = np.bincount((vert_key // n_nodes).astype(np.int64), minlength=n_charts).astype(np.int64)
    euler = verts - edges + faces

    sel = np.flatnonzero(is_boundary)
    picked = order[sel]
    loops, loop_chart, pinched = _chain_boundary(
        edge_chart[picked],
        tail[picked],
        head[picked],
        tail_loop[picked],
    )

    loops_per_chart = np.bincount(loop_chart, minlength=n_charts)
    fins = run_live & ((n_fwd + n_bwd) > 2)
    non_manifold = np.bincount(run_chart[fins], minlength=n_charts) > 0
    pinch = np.bincount(loop_chart[pinched], minlength=n_charts) > 0
    bad = (euler != 1) | (loops_per_chart != 1) | non_manifold | pinch
    non_disk = int(np.count_nonzero(bad)) if n_tris else 0
    return _ChartTopology(loops, loop_chart, non_disk, euler)


def _run_ids(*keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Return ``(run_id, n_runs)`` for a lexicographically sorted key tuple."""
    total = keys[0].shape[0]
    if total == 0:
        return np.zeros(0, dtype=np.int64), 0
    is_new = np.zeros(total, dtype=bool)
    is_new[0] = True
    for key in keys:
        is_new[1:] |= key[1:] != key[:-1]
    run_id = np.cumsum(is_new) - 1
    return run_id.astype(np.int64), int(run_id[-1]) + 1


def _run_starts(run_id: np.ndarray, n_runs: int) -> np.ndarray:
    """Return the index of the first element of each run."""
    if n_runs == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.zeros(n_runs, dtype=np.int64)
    starts[run_id[::-1]] = np.arange(run_id.shape[0] - 1, -1, -1, dtype=np.int64)
    return starts


def _chain_boundary(
    chart: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
    tail_loop: np.ndarray,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Chain boundary half-edges head-to-tail into closed loops, per chart.

    The walk consumes each half-edge exactly once, so the total cost is linear
    in the boundary size regardless of how tangled the chart is.  A chain that
    dead-ends -- only reachable from malformed input, where a UV vertex has more
    incoming boundary half-edges than outgoing -- is emitted as-is and closed
    implicitly, which keeps every returned polyline closed as the contract
    promises.

    Returns:
        ``(loops, chart_of_loop, pinched)``.  ``pinched`` marks the loops that
        visit the same welded UV vertex twice, which is what a bowtie looks like
        from the boundary and is why the caller cannot call such a chart a disk.
    """
    n = chart.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    chart_list = chart.tolist()
    tail_list = tail.tolist()
    head_list = head.tolist()
    loop_list = tail_loop.tolist()

    # (chart, tail node) -> half-edge indices starting there.  The chart is part
    # of the key so a caller-supplied labelling that splits an island cannot
    # chain one chart's boundary into another's.
    outgoing: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        outgoing.setdefault((chart_list[i], tail_list[i]), []).append(i)
    cursor = dict.fromkeys(outgoing, 0)

    used = bytearray(n)
    loops: list[list[int]] = []
    owners: list[int] = []
    pinched: list[bool] = []
    for start in range(n):
        if used[start]:
            continue
        chain: list[int] = []
        nodes: list[int] = []
        here = start
        while not used[here]:
            used[here] = 1
            chain.append(loop_list[here])
            nodes.append(tail_list[here])
            key = (chart_list[here], head_list[here])
            bucket = outgoing.get(key)
            nxt = -1
            if bucket is not None:
                pos = cursor[key]
                while pos < len(bucket) and used[bucket[pos]]:
                    pos += 1
                cursor[key] = pos
                if pos < len(bucket):
                    nxt = bucket[pos]
            if nxt < 0:
                break
            here = nxt
        if len(chain) >= 3:
            pivot = chain.index(min(chain))
            loops.append(chain[pivot:] + chain[:pivot])
            owners.append(chart_list[start])
            pinched.append(len(set(nodes)) != len(nodes))
    order = sorted(range(len(loops)), key=lambda i: (owners[i], loops[i][0]))
    return (
        [loops[i] for i in order],
        np.asarray([owners[i] for i in order], dtype=np.int64),
        np.asarray([pinched[i] for i in order], dtype=bool),
    )


# ---------------------------------------------------------------------------
# Boundary self-intersection
# ---------------------------------------------------------------------------


def boundary_segments(loops: Iterable[Sequence[int]]) -> np.ndarray:
    """Return the ``(S, 2)`` endpoint index array of every boundary segment.

    Segments are numbered loop by loop, in order, with each loop's closing
    segment (last vertex back to first) last.  That numbering is the contract
    the segment ids in :func:`crossing_pairs` refer to, and the naive oracle in
    ``tests/reference.py`` reproduces it exactly so the two can be compared pair
    by pair rather than only by count.

    Args:
        loops: Closed polylines of indices, as returned by :func:`boundary_loops`.

    Returns:
        ``(S, 2)`` int64 array of ``(from, to)`` indices.
    """
    chunks: list[np.ndarray] = []
    for loop in loops:
        idx = np.asarray(loop, dtype=np.int64).reshape(-1)
        if idx.shape[0] < 2:
            continue
        chunks.append(np.stack([idx, np.roll(idx, -1)], axis=1))
    if not chunks:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(chunks, axis=0)


def boundary_crossings(uv: np.ndarray, loops: Iterable[Sequence[int]]) -> int:
    """Return the number of transversal self-intersections of the boundary.

    This is the function that sees Suzanne's defect.  A flip test cannot: her
    968 signed UV areas are all positive while the chart boundary still laps
    over a distant part of itself (``docs/ORGANIC.md`` section 0.3).

    Two segments count once when they cross **transversally** -- each strictly
    separates the other's endpoints, decided by exact ``orient2d`` signs.
    Segments that share an endpoint (every consecutive pair of a polyline),
    touch at a T-junction, or overlap collinearly are **not** crossings, which
    matches the evaluator's overlap convention.

    Args:
        uv: ``(N, 2)`` coordinates the loop indices refer to; normally
            ``sample.uvs``.
        loops: Closed polylines of indices into ``uv``.  Crossings are counted
            between all supplied segments, including between different loops, so
            pass one chart's loops to ask about that chart alone.

    Returns:
        The crossing count.  ``0`` means the supplied boundary is a set of
        simple, pairwise non-crossing closed curves.
    """
    return int(crossing_pairs(uv, loops).shape[0])


def crossing_pairs(uv: np.ndarray, loops: Iterable[Sequence[int]]) -> np.ndarray:
    """Return the ``(K, 2)`` segment-id pairs that cross, for diagnostics.

    Args:
        uv: ``(N, 2)`` coordinates the loop indices refer to.
        loops: Closed polylines of indices into ``uv``.

    Returns:
        ``(K, 2)`` int64 array of segment ids ``(a, b)`` with ``a < b``, sorted
        lexicographically, so the result is deterministic and directly
        comparable with the naive oracle.  Segment ids index the array
        :func:`boundary_segments` returns for the same ``loops``.
    """
    seg = boundary_segments(loops)
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return _crossing_pairs_impl(points, seg, None, 0)[0]


def _crossings_by_chart(
    uv: np.ndarray,
    loops: Sequence[Sequence[int]],
    loop_chart: np.ndarray,
    n_charts: int,
) -> np.ndarray:
    """Return a ``(n_charts,)`` crossing count, restricting pairs to one chart.

    Every chart's segments go through the broad phase together -- one grid over
    the whole layout rather than one per chart -- and cross-chart candidate
    pairs are dropped before any predicate runs.  Two islands overlapping each
    other is a packing failure, measured by ``overlap_area``; this number is
    strictly about a chart folding onto itself.
    """
    counts = np.zeros(max(n_charts, 0), dtype=np.int64)
    if not loops or n_charts <= 0:
        return counts
    seg = boundary_segments(loops)
    if seg.shape[0] == 0:
        return counts
    # ``boundary_segments`` drops polylines shorter than two vertices, so the
    # per-segment chart labels have to be built from exactly the same filter or
    # every later loop's crossings land against the wrong chart.
    owners = np.asarray(loop_chart, dtype=np.int64).reshape(-1)
    kept = [(owners[i], len(loop)) for i, loop in enumerate(loops) if len(loop) >= 2]
    group = np.repeat(
        np.asarray([o for o, _ in kept], dtype=np.int64),
        np.asarray([n for _, n in kept], dtype=np.int64),
    )
    pairs, pair_group = _crossing_pairs_impl(uv, seg, group, n_charts)
    if pairs.shape[0]:
        counts += np.bincount(pair_group, minlength=n_charts)
    return counts


def _crossing_pairs_impl(
    uv: np.ndarray,
    seg: np.ndarray,
    group: np.ndarray | None,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pairs, pair_group)``; ``group`` restricts pairs to one group."""
    empty = (np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64))
    n_seg = seg.shape[0]
    if n_seg < 2 or uv.shape[0] == 0:
        return empty
    if seg.max(initial=-1) >= uv.shape[0] or seg.min(initial=0) < 0:
        return empty

    p = uv[seg[:, 0]]
    q = uv[seg[:, 1]]
    usable = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    lo = np.minimum(p, q)
    hi = np.maximum(p, q)
    if not usable.all():
        # A non-finite endpoint has no position, so it can neither cross nor be
        # crossed.  Collapse it to the origin so the grid maths stays finite and
        # exclude it from every pair below.
        lo = np.where(usable[:, None], lo, 0.0)
        hi = np.where(usable[:, None], hi, 0.0)

    pair_a, pair_b = _aabb_candidate_pairs(lo, hi)
    if pair_a.size == 0:
        return empty

    keep = usable[pair_a] & usable[pair_b]
    # Segments sharing an endpoint index touch; touching is not crossing.  The
    # exact predicate would reach the same verdict -- a shared endpoint makes
    # orient2d exactly zero -- but rejecting them here is free and keeps the
    # consecutive pairs of every polyline out of the predicate entirely.
    keep &= (
        (seg[pair_a, 0] != seg[pair_b, 0])
        & (seg[pair_a, 0] != seg[pair_b, 1])
        & (seg[pair_a, 1] != seg[pair_b, 0])
        & (seg[pair_a, 1] != seg[pair_b, 1])
    )
    if group is not None:
        keep &= group[pair_a] == group[pair_b]
    if not keep.any():
        return empty
    pair_a = pair_a[keep]
    pair_b = pair_b[keep]

    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    for start in range(0, pair_a.shape[0], _PAIR_CHUNK):
        sub_a = pair_a[start : start + _PAIR_CHUNK]
        sub_b = pair_b[start : start + _PAIR_CHUNK]
        cross = _properly_cross(p[sub_a], q[sub_a], p[sub_b], q[sub_b])
        if cross.any():
            out_a.append(sub_a[cross])
            out_b.append(sub_b[cross])
    if not out_a:
        return empty

    hit_a = np.concatenate(out_a)
    hit_b = np.concatenate(out_b)
    order = np.lexsort((hit_b, hit_a))
    pairs = np.stack([hit_a[order], hit_b[order]], axis=1)
    if group is None or n_groups <= 0:
        return pairs, np.zeros(pairs.shape[0], dtype=np.int64)
    return pairs, group[pairs[:, 0]]


def _properly_cross(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """Return a bool mask: does segment ``p1-q1`` transversally cross ``p2-q2``?

    The textbook test, with the degenerate cases resolved the way this module
    defines them.  Segment 1 crosses segment 2 exactly when 2's endpoints lie
    strictly on opposite sides of 1's supporting line *and* vice versa.  All
    four orientations must be non-zero, which is what rules out shared
    endpoints, T-junctions and collinear overlap.

    The two sides are evaluated in sequence and the second only on the pairs
    that survived the first, so a clean layout -- where nearly every candidate
    is rejected immediately -- pays for barely more than one predicate per pair.
    """
    n = p1.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    s3 = _orient2d_sign(p1, q1, p2)
    s4 = _orient2d_sign(p1, q1, q2)
    first = (s3 != 0) & (s4 != 0) & (s3 != s4)
    out = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(first)
    if idx.size == 0:
        return out
    s1 = _orient2d_sign(p2[idx], q2[idx], p1[idx])
    s2 = _orient2d_sign(p2[idx], q2[idx], q1[idx])
    out[idx] = (s1 != 0) & (s2 != 0) & (s1 != s2)
    return out


def _orient2d_sign(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the **exact** sign of ``orient2d(a, b, c)`` for stacked points.

    Positive means ``c`` lies to the left of the directed line ``a -> b``, zero
    means the three points are exactly collinear.

    The float64 evaluation is accepted only where Shewchuk's static error bound
    proves the sign; the rest -- which is only ever near-collinear input -- is
    recomputed exactly.  ``float.as_integer_ratio`` gives every float64 as an
    integer over a power of two, so scaling all six coordinates by the largest
    of those powers turns the determinant into an exact Python-integer
    expression.  No tolerance is involved anywhere, which is what makes a
    "no crossings" answer a proof rather than an opinion.

    Args:
        a: ``(N, 2)`` first point of the directed line.
        b: ``(N, 2)`` second point of the directed line.
        c: ``(N, 2)`` query point.

    Returns:
        ``(N,)`` int8 array of ``-1``, ``0`` or ``+1``.
    """
    acx = a[:, 0] - c[:, 0]
    acy = a[:, 1] - c[:, 1]
    bcx = b[:, 0] - c[:, 0]
    bcy = b[:, 1] - c[:, 1]
    det_left = acx * bcy
    det_right = acy * bcx
    det = det_left - det_right

    # Using |left| + |right| as the bound unconditionally is more conservative
    # than Shewchuk's branch on matching signs: it can only send extra pairs to
    # the exact path, never accept a wrong sign.  When both products are zero
    # the bound is zero and the comparison passes, which is correct -- the
    # determinant is then exactly zero with no cancellation at all.
    bound = _ORIENT_ERRBOUND * (np.abs(det_left) + np.abs(det_right))
    with np.errstate(invalid="ignore"):
        # A non-finite determinant -- only reachable by overflow from finite but
        # astronomically scaled coordinates -- never clears the bound, so it is
        # routed to the exact path below; zeroing it here only keeps the cast
        # quiet in the meantime.
        sign = np.sign(np.where(np.isfinite(det), det, 0.0)).astype(np.int8)
        uncertain = np.flatnonzero(~(np.abs(det) >= bound))
    if uncertain.size:
        ax = a[:, 0]
        ay = a[:, 1]
        bx = b[:, 0]
        by = b[:, 1]
        cx = c[:, 0]
        cy = c[:, 1]
        for i in uncertain.tolist():
            sign[i] = _orient2d_sign_exact(
                float(ax[i]),
                float(ay[i]),
                float(bx[i]),
                float(by[i]),
                float(cx[i]),
                float(cy[i]),
            )
    return sign


def _orient2d_sign_exact(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> int:
    """Return the sign of ``orient2d`` computed in exact integer arithmetic.

    Every finite float64 is exactly ``n / 2**k``.  Scaling all six coordinates
    to the common power ``2**max(k)`` makes them integers whose determinant is
    an exact Python integer, so the sign is exact by construction.  Non-finite
    input never reaches here: those segments are excluded before the broad
    phase.
    """
    scaled: list[int] = []
    numerators: list[tuple[int, int]] = []
    widest = 0
    for value in (ax, ay, bx, by, cx, cy):
        num, den = value.as_integer_ratio()
        shift = den.bit_length() - 1
        numerators.append((num, shift))
        if shift > widest:
            widest = shift
    for num, shift in numerators:
        scaled.append(num << (widest - shift))
    iax, iay, ibx, iby, icx, icy = scaled
    det = (iax - icx) * (iby - icy) - (iay - icy) * (ibx - icx)
    if det > 0:
        return 1
    return -1 if det < 0 else 0


# ---------------------------------------------------------------------------
# Broad phase
# ---------------------------------------------------------------------------


def _aabb_candidate_pairs(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return index pairs whose axis-aligned boxes strictly overlap.

    A uniform grid sized to the median segment, exactly as the evaluator's
    overlap pass does for triangles: cost is linear in the segment count times
    the local density, never the ``O(n^2)`` the contract rules out.  Segments
    are inserted into every cell their box touches, so the grid can only ever
    produce *extra* candidates, never miss a pair.

    The strict ``<`` in the box test is safe for this module's definition of a
    crossing.  A transversal crossing puts each segment's endpoints strictly on
    opposite sides of the other's supporting line, so both boxes must strictly
    overlap in both axes even when one of them is degenerate -- an axis-aligned
    segment, say.  Boxes that merely touch bound segments that at most touch,
    and touching is not crossing.
    """
    n = lo.shape[0]
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if n < 2:
        return empty

    origin = lo.min(axis=0)
    span = hi.max(axis=0) - origin
    extent = float(max(span[0], span[1]))
    if not (extent > 0.0):
        # Every segment collapses to the same point: nothing can cross.
        return empty

    sizes = (hi - lo).max(axis=1)
    cell = float(np.median(sizes))
    if not (cell > 0.0):
        cell = extent / 512.0
    cell = max(cell, extent / _MAX_GRID_DIM)

    for _ in range(32):
        i0 = np.floor((lo[:, 0] - origin[0]) / cell).astype(np.int64)
        i1 = np.floor((hi[:, 0] - origin[0]) / cell).astype(np.int64)
        j0 = np.floor((lo[:, 1] - origin[1]) / cell).astype(np.int64)
        j1 = np.floor((hi[:, 1] - origin[1]) / cell).astype(np.int64)
        counts = (i1 - i0 + 1) * (j1 - j0 + 1)
        total = int(counts.sum())
        if total <= _MAX_INSERTS:
            break
        cell *= 2.0
    else:  # pragma: no cover - one cell holds every segment long before this
        return empty
    if total <= 1:
        return empty

    grid_w = int(i1.max()) + 2
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    owner = np.repeat(np.arange(n, dtype=np.int64), counts)
    offset = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    width = np.repeat(i1 - i0 + 1, counts)
    cell_id = (np.repeat(j0, counts) + offset // width) * grid_w + (
        np.repeat(i0, counts) + offset % width
    )

    pair_a, pair_b = _cell_pairs(cell_id, owner)
    if pair_a.size == 0:
        return empty
    packed = np.unique(pair_a * n + pair_b)
    pair_a = packed // n
    pair_b = packed - pair_a * n

    keep = (
        (lo[pair_a, 0] < hi[pair_b, 0])
        & (lo[pair_b, 0] < hi[pair_a, 0])
        & (lo[pair_a, 1] < hi[pair_b, 1])
        & (lo[pair_b, 1] < hi[pair_a, 1])
    )
    return pair_a[keep], pair_b[keep]


def _cell_pairs(cell_id: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(a, b)`` owner pairs that share at least one grid cell.

    Insertions are sorted by cell and each run is walked at a widening stride,
    which enumerates every within-cell pair exactly and needs no per-cell Python
    loop.  Generation stops at ``_MAX_PAIRS``: a boundary dense enough to reach
    that is already catastrophically self-overlapping and will be caught by the
    ``overlap_area`` conjunct instead.
    """
    order = np.argsort(cell_id, kind="stable")
    sorted_cells = cell_id[order]
    sorted_owner = owner[order]

    total = sorted_cells.shape[0]
    is_new = np.empty(total, dtype=bool)
    is_new[0] = True
    np.not_equal(sorted_cells[1:], sorted_cells[:-1], out=is_new[1:])
    run_starts = np.flatnonzero(is_new)
    run_of = np.cumsum(is_new) - 1
    within = np.arange(total, dtype=np.int64) - run_starts[run_of]
    run_sizes = np.diff(np.append(run_starts, total))
    size_of = run_sizes[run_of]
    longest = int(run_sizes.max())
    if longest < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)

    chunks_a: list[np.ndarray] = []
    chunks_b: list[np.ndarray] = []
    produced = 0
    for stride in range(1, longest):
        picked = np.flatnonzero((within + stride) < size_of)
        if picked.size == 0:
            break
        first = sorted_owner[picked]
        second = sorted_owner[picked + stride]
        keep = first != second
        if keep.any():
            chunks_a.append(np.minimum(first[keep], second[keep]))
            chunks_b.append(np.maximum(first[keep], second[keep]))
            produced += int(keep.sum())
        if produced >= _MAX_PAIRS:
            break
    if not chunks_a:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(chunks_a), np.concatenate(chunks_b)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class _Clean:
    """Validated views of a sample's UV topology, with the corrupt rows removed.

    Mirrors the evaluator's conditioning: triangles whose loop indices fall
    outside the UV array are corrupt input, so they are dropped and counted
    rather than allowed to raise an ``IndexError`` into a Blender operator.
    ``kept`` records which original triangles survived, so a caller-supplied
    ``tri_chart`` -- which is indexed by *original* triangle -- can still be
    lined up with them instead of being silently discarded.
    """

    __slots__ = ("uvs", "tri_loops", "kept", "n_total", "n_dropped")

    def __init__(self, sample: UVMeshSample) -> None:
        """Condition ``sample``'s UV arrays; never raises on malformed input."""
        uvs = np.asarray(sample.uvs, dtype=np.float64)
        self.uvs = uvs.reshape(-1, 2) if uvs.size else np.zeros((0, 2), dtype=np.float64)

        tri_loops = np.asarray(sample.tri_loops, dtype=np.int64)
        tri_loops = tri_loops.reshape(-1, 3) if tri_loops.size else np.zeros((0, 3), np.int64)
        self.n_total = tri_loops.shape[0]
        self.kept: np.ndarray | None = None
        if self.n_total == 0 or self.uvs.shape[0] == 0:
            self.tri_loops = np.zeros((0, 3), np.int64)
            self.n_dropped = self.n_total
            return

        if int(tri_loops.min()) < 0 or int(tri_loops.max()) >= self.uvs.shape[0]:
            ok = (tri_loops >= 0).all(axis=1) & (tri_loops < self.uvs.shape[0]).all(axis=1)
            self.kept = np.flatnonzero(ok)
            tri_loops = tri_loops[ok]
        self.tri_loops = np.ascontiguousarray(tri_loops)
        self.n_dropped = self.n_total - self.tri_loops.shape[0]


def _weld_uv_vertices(sample: UVMeshSample) -> tuple[np.ndarray, int]:
    """Return ``(node_of_loop, n_nodes)``: loops welded into UV vertices.

    Two loops weld when they share a mesh vertex and an exactly equal UV -- see
    the module docstring for why exact equality is the right rule on unwrapper
    output.  The UV pair is compared by viewing the contiguous ``(L, 2)``
    float64 block as ``(L,)`` complex128, which is the same 16 bytes and turns a
    two-column ``np.unique`` into a single-column one.
    """
    uvs = np.ascontiguousarray(np.asarray(sample.uvs, dtype=np.float64).reshape(-1, 2))
    n_loops = uvs.shape[0]
    if n_loops == 0:
        return np.zeros(0, dtype=np.int64), 0

    loop_verts = np.asarray(sample.loop_verts, dtype=np.int64).reshape(-1)
    if loop_verts.shape[0] < n_loops:
        loop_verts = np.concatenate(
            [loop_verts, np.zeros(n_loops - loop_verts.shape[0], dtype=np.int64)]
        )
    # A negative vertex index is corrupt input; folding it to 0 keeps the weld
    # key well formed and, at worst, welds two loops that should not have been.
    vert = np.maximum(loop_verts[:n_loops], 0)

    _, uv_id = np.unique(uvs.view(np.complex128).reshape(-1), return_inverse=True)
    uv_id = uv_id.reshape(-1).astype(np.int64)
    key = vert * (int(uv_id.max()) + 1) + uv_id
    _, node = np.unique(key, return_inverse=True)
    node = np.ascontiguousarray(node.reshape(-1), dtype=np.int64)
    return node, int(node.max()) + 1


def _signed_uv_areas(uvs: np.ndarray, tri_loops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(signed_area, finite)`` per triangle, from UVs alone."""
    u0 = uvs[tri_loops[:, 0]]
    u1 = uvs[tri_loops[:, 1]]
    u2 = uvs[tri_loops[:, 2]]
    d1 = u1 - u0
    d2 = u2 - u0
    signed = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    finite = np.isfinite(u0).all(axis=1) & np.isfinite(u1).all(axis=1)
    finite &= np.isfinite(u2).all(axis=1) & np.isfinite(signed)
    return signed, finite


def _uv_area_floor(signed: np.ndarray, finite: np.ndarray) -> float:
    """Return the relative UV-area floor below which a triangle has collapsed."""
    count = int(np.count_nonzero(finite))
    if count == 0:
        return _DEGENERATE_EPS
    mean = float(np.abs(signed[finite]).sum()) / count
    if not np.isfinite(mean) or mean <= 0.0:
        return _DEGENERATE_EPS
    return _DEGENERATE_EPS * mean


def _count_flipped(
    signed: np.ndarray, usable: np.ndarray, labels: np.ndarray, n_charts: int
) -> int:
    """Count triangles whose UV winding opposes their own chart's majority.

    A chart that is *uniformly* mirrored is still injective -- artists mirror
    islands on purpose -- so the majority, not the sign, is the reference.  Ties
    resolve to positive, which only matters for a chart split exactly in half,
    and such a chart is broken under either reading.
    """
    if n_charts <= 0 or signed.size == 0:
        return 0
    positive = np.bincount(labels[usable & (signed > 0.0)], minlength=n_charts)
    negative = np.bincount(labels[usable & (signed < 0.0)], minlength=n_charts)
    majority_positive = positive >= negative
    opposes = np.where(majority_positive[labels], signed < 0.0, signed > 0.0)
    return int(np.count_nonzero(usable & opposes))
