"""Candidate generation and branching: the coordinator that ties it together.

The engine is the only place that knows the full pipeline:

    extract the mesh once
      -> for each parameter recipe
           -> run a backend to produce UVs
           -> derive the seams that layout actually implies
           -> measure distortion
           -> score
      -> rank, index, hand back

Two invariants drive the design:

1. **Extract once, unwrap many.** Pulling a mesh out of Blender is the single
   most expensive non-solver step, and the geometry does not change between
   candidates -- only the UVs do.  Extracting per candidate would make a
   10-slot run roughly ten times slower for no benefit.

2. **The mesh is put back exactly as it was found.** Backends drive
   ``bpy.ops.uv.*``, which mutates seams, selection and mode.  The engine takes
   a snapshot up front and restores it in a ``finally``, so an exception, a
   cancellation or a backend that misbehaves all leave the artist's mesh
   untouched.  Generation is a measurement, not an edit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..backends import seams as seam_mod
from ..backends.base import UnwrapRequest, get_backend
from .candidate import Candidate, CandidateStore
from .evaluator import evaluate
from .meshdata import UVMeshSample
from .metrics import ScoreWeights, pareto_front, score
from .params import UnwrapParams, mutate, sample_grid

__all__ = [
    "GenerationReport",
    "generate",
    "branch",
    "derive_uv_seams",
    "validate_object",
    "chart_seams",
    "chart_cut_request",
    "CHART_CUT_KEY",
    "CHART_SEED_KEY",
]

#: Progress callback signature: ``(fraction_0_to_1, human_message)``.
ProgressFn = Callable[[float, str], None]
#: Cancellation probe: return ``True`` to stop after the current candidate.
CancelFn = Callable[[], bool]


@dataclass
class GenerationReport:
    """Everything one generation run produced, successes and failures alike.

    Attributes:
        candidates: Successful candidates, already ranked and indexed.
        seconds: Wall-clock duration of the whole run.
        failures: ``(params, reason)`` for every recipe that did not produce a
            usable layout.  Surfaced in the UI rather than swallowed -- a run
            where 7 of 10 recipes failed is something the artist must see.
        warnings: Non-fatal notes (stale store discarded, packing unavailable,
            a backend skipped because it is not installed, and so on).
        cancelled: ``True`` when the user interrupted the run.
    """

    candidates: list[Candidate] = field(default_factory=list)
    seconds: float = 0.0
    failures: list[tuple[UnwrapParams, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        """Whether the run produced at least one usable candidate."""
        return bool(self.candidates)

    def summary(self) -> str:
        """Return a one-line ASCII summary suitable for the status bar."""
        parts = ["%d candidate(s) in %.2fs" % (len(self.candidates), self.seconds)]
        if self.failures:
            parts.append("%d failed" % len(self.failures))
        if self.cancelled:
            parts.append("cancelled")
        return ", ".join(parts)


def validate_object(obj: Any) -> tuple[bool, str]:
    """Check that ``obj`` is something we can actually unwrap.

    Returns ``(False, reason)`` rather than raising so callers can put the
    reason straight into ``self.report({'ERROR'}, ...)``.
    """
    if obj is None:
        return False, "No active object."
    if getattr(obj, "type", None) != "MESH":
        return False, "Active object %r is not a mesh." % getattr(obj, "name", "?")
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return False, "Object has no mesh data."
    try:
        n_polys = len(mesh.polygons)
        n_loops = len(mesh.loops)
    except (AttributeError, TypeError):
        return False, "Mesh data is not readable."
    if n_polys == 0 or n_loops == 0:
        return False, "Mesh has no faces to unwrap."
    return True, ""


def derive_uv_seams(
    sample: UVMeshSample,
    topo: "seam_mod.EdgeTopology",
    uvs: np.ndarray,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    """Return the ``(E,)`` seam mask that a given UV layout actually implies.

    Rather than reporting the seams we *asked* for, this reads back the seams
    the result genuinely has: an interior edge is a seam exactly when its two
    incident faces do not agree on the UV coordinates of its two endpoints.

    This matters because it is the only way to get a meaningful seam length for
    methods that ignore seams entirely.  ``smart_project`` and ``cube_project``
    invent their own charts, so asking "what seams did we mark?" would report
    zero for layouts that are in fact heavily cut.  Measuring the layout makes
    every method comparable on the same footing, which is the whole point of
    ranking them against each other.

    Args:
        sample: The mesh being measured.
        topo: Edge topology for ``sample`` (built once, reused per candidate).
        uvs: ``(L, 2)`` candidate UVs.
        eps: Coordinate tolerance below which two UVs count as identical.

    Returns:
        ``(E,)`` bool array. Boundary and non-manifold edges are never seams --
        there is no second face to disagree with.
    """
    n_edges = sample.n_edges
    out = np.zeros(n_edges, dtype=bool)
    if n_edges == 0 or sample.n_tris == 0:
        return out

    # Only manifold interior edges (exactly two incident faces) can be seams.
    interior = np.asarray(topo.face_count, dtype=np.int64) == 2
    if not interior.any():
        return out

    idx = np.nonzero(interior)[0]
    fa = np.asarray(topo.face_a, dtype=np.int64)[idx]
    fb = np.asarray(topo.face_b, dtype=np.int64)[idx]

    n_tris = sample.n_tris
    valid = (fa >= 0) & (fa < n_tris) & (fb >= 0) & (fb < n_tris)
    if not valid.all():
        idx, fa, fb = idx[valid], fa[valid], fb[valid]
    if idx.size == 0:
        return out

    ev = np.asarray(sample.edge_verts, dtype=np.int64)[idx]
    tv_a, tl_a = sample.tri_verts[fa], sample.tri_loops[fa]
    tv_b, tl_b = sample.tri_verts[fb], sample.tri_loops[fb]
    rows = np.arange(idx.size)

    def corner_loops(tv: np.ndarray, tl: np.ndarray, vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Loop index of ``vert`` within each face, plus a found mask."""
        eq = tv == vert[:, None]
        found = eq.any(axis=1)
        col = eq.argmax(axis=1)
        return tl[rows, col], found

    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] < sample.n_loops:
        return out

    seam = np.zeros(idx.size, dtype=bool)
    for corner in (0, 1):
        vert = ev[:, corner]
        la, ok_a = corner_loops(tv_a, tl_a, vert)
        lb, ok_b = corner_loops(tv_b, tl_b, vert)
        # A vertex missing from one of the two faces means the triangulation
        # diagonal was matched instead of a real edge; treat it as continuous
        # rather than inventing a seam.
        both = ok_a & ok_b
        delta = np.abs(uv[la] - uv[lb]).max(axis=1)
        seam |= both & (delta > eps)

    out[idx] = seam
    return out


def dead_layout_reason(uvs: np.ndarray, previous: np.ndarray | None = None) -> str:
    """Return why ``uvs`` is not a usable layout, or ``""`` when it is fine.

    This exists because of a verified and genuinely nasty Blender behaviour:
    **``bpy.ops.uv.unwrap`` returns ``{'FINISHED'}`` even when it fails.**  On a
    closed mesh with no seams it prints "Unwrap failed to solve 1 of 1
    island(s)" to the console, leaves the previous UVs completely untouched, and
    still reports success.  Trusting the operator's return status therefore
    scores a candidate that did nothing at all, and -- worse -- it inherits
    whatever UVs the previous candidate left behind, so it can silently
    duplicate a neighbour's result.

    Six of ten slots on an icosphere were dead this way before this check
    existed, every one of them reported as a valid candidate.

    The check is deliberately at the engine level rather than inside a single
    backend: it protects every backend, including third-party ones, and costs
    a few microseconds against a solver that costs milliseconds.

    Args:
        uvs: ``(L, 2)`` candidate UVs.
        previous: Optional UVs the mesh carried before this attempt.  When the
            result is bit-identical to them, the solver almost certainly did
            nothing.

    Returns:
        A human-readable reason, or the empty string when the layout is usable.
    """
    arr = np.asarray(uvs, dtype=np.float64)
    if arr.size == 0:
        return "backend produced no UVs"
    if not np.isfinite(arr).all():
        return "layout contains NaN or infinite coordinates"

    span = arr.max(axis=0) - arr.min(axis=0)
    if float(span.max()) <= 1e-9:
        # Every UV collapsed onto a single point. This is what an unsolved
        # unwrap leaves behind on a freshly created (zero-filled) UV layer.
        return "unwrap produced a degenerate layout (all UVs coincident)"

    if previous is not None:
        prev = np.asarray(previous, dtype=np.float64)
        if prev.shape == arr.shape and np.array_equal(prev, arr):
            return "unwrap left the UVs unchanged (solver reported success but did nothing)"
    return ""


# ---------------------------------------------------------------------------
# Chart cuts: segmentation seams for surfaces a heuristic cannot open
# ---------------------------------------------------------------------------
#
# ``DUAL_GRAPH`` cuts the complement of a maximum spanning forest of the face
# dual graph.  On a *faceted* mesh that is exactly right -- a cube comes out at
# its seven-edge cross net.  On a *smooth* closed mesh it is a disaster, because
# the complement of a dual spanning tree is always a spanning tree of every
# interior vertex: 161 of 480 edges on a subdivision-3 icosphere (seam length
# 45.1 against a model whose sqrt(area) is 3.5) and 1105 of 2256 on a UV sphere.
# The layout is perfectly isometric -- a polyhedral net is developable by
# construction -- but the cut is unusable, it packs at 0.22 efficiency, and
# ``MINIMUM_STRETCH`` folds the resulting spiky chart over itself (measured
# overlap 7.9e-03 on the icosphere).
#
# The cut below is the other classic answer: segment the surface into charts
# (variational-shape-approximation style clustering of polygon normals) and cut
# on the chart boundaries.  Those are closed loops, so the cut is short, and
# each chart is compact, so it packs and it does not fold.
#
# The one thing clustering cannot guarantee is that a chart is a *disk* -- the
# top half of a torus is an annulus, and a solver handed an annulus either fails
# or folds.  So every chart is checked and bisected until it is one.  Disk-ness
# is decided exactly, by Euler characteristic, from three bincounts.

#: Solvers that need the surface cut open into disks before they can run.
_SEAM_DRIVEN = ("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")

#: ``params.extra`` key requesting an engine-computed chart cut; the value is
#: the number of seed clusters.
CHART_CUT_KEY = "chart_cut"
#: ``params.extra`` key rotating the seed directions, so several chart recipes
#: on one mesh produce genuinely different -- and equally valid -- layouts.
CHART_SEED_KEY = "chart_seed"

#: Cluster counts the escalation path chooses between.
_CHART_CLUSTERS = (3, 4, 6, 8, 12)
#: Guard on the bisection loop.  Each round at most doubles the chart count, so
#: twelve is far more than any real mesh needs; it exists so a pathological
#: input cannot spin.
_CHART_MAX_SPLITS = 12


def _label_components(n_nodes: int, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Label the connected components of an undirected graph, vectorised.

    Hook every node to the smallest label in its neighbourhood, then flatten the
    pointer forest by doubling.  Returns ``(n_nodes,)`` int64 labels where the
    label is the smallest node id in the component.
    """
    labels = np.arange(max(n_nodes, 0), dtype=np.int64)
    if n_nodes <= 0 or ea.size == 0:
        return labels
    ea = np.ascontiguousarray(ea, dtype=np.int64)
    eb = np.ascontiguousarray(eb, dtype=np.int64)
    for _ in range(64):
        la = labels[ea]
        lb = labels[eb]
        if np.array_equal(la, lb):
            break
        smallest = np.minimum(la, lb)
        np.minimum.at(labels, ea, smallest)
        np.minimum.at(labels, eb, smallest)
        for _ in range(32):
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
    return labels


def _seed_directions(clusters: int, seed: int) -> np.ndarray:
    """Return ``(k, 3)`` evenly spread unit directions, rotated by ``seed``.

    A Fibonacci lattice spreads k directions over the sphere far more evenly
    than random sampling, and a seeded random rotation of the whole set is what
    makes two chart recipes with the same ``k`` produce different charts rather
    than the same one twice.
    """
    k = max(int(clusters), 1)
    i = np.arange(k, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / k
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * radius, np.sin(theta) * radius, z], axis=1)
    if not seed:
        return dirs
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    quat = rng.normal(size=4)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return dirs
    w, x, y, z4 = quat / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z4 * z4), 2 * (x * y - z4 * w), 2 * (x * z4 + y * w)],
            [2 * (x * y + z4 * w), 1 - 2 * (x * x + z4 * z4), 2 * (y * z4 - x * w)],
            [2 * (x * z4 - y * w), 2 * (y * z4 + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return dirs @ rot.T


#: Chart area, as a fraction of the mean chart area, below which a chart is
#: absorbed into its largest neighbour instead of becoming its own island.
#:
#: Clustering by normal is a global operation, so one cluster routinely lands in
#: several disconnected places on an organic model -- Suzanne's four normal
#: clusters come apart into 75 connected charts, most of them slivers around the
#: ears and eyes.  Every sliver is an island the artist has to place by hand, and
#: cutting one buys almost no distortion relief.  Absorbing them measured
#: (Suzanne, 3 clusters, Blender 4.5.4): 52 islands at score 15.6 -> 8 islands at
#: score 26.9.  It is a no-op on a mesh whose charts are already even, which is
#: every primitive in the benchmark.
_CHART_MIN_AREA_SHARE = 0.3
#: Absorption rounds; each one merges every current sliver, so this is a
#: safety net rather than a schedule.
_CHART_ABSORB_ROUNDS = 8


def chart_seams(
    sample: UVMeshSample,
    topo: "seam_mod.EdgeTopology",
    *,
    clusters: int = 6,
    seed: int = 0,
    max_splits: int = _CHART_MAX_SPLITS,
    min_area_share: float = _CHART_MIN_AREA_SHARE,
) -> np.ndarray:
    """Return an ``(E,)`` seam mask that cuts ``sample`` into disk-shaped charts.

    Algorithm
    ---------

    1. **Contract tessellation diagonals.**  ``tri_verts`` comes from
       ``loop_triangles``, so a quad is two triangles sharing a diagonal that is
       not a mesh edge and therefore cannot be cut.  Contracting those
       adjacencies turns the triangle soup back into polygons, which is the
       level every step below works at.
    2. **Cluster polygon normals.**  Area-weighted k-means seeded from a
       Fibonacci lattice of ``clusters`` directions.  This is Cohen-Steiner's
       variational shape approximation with the proxy fixed to a plane normal:
       cheap, deterministic, and it lands cuts on the creases of a hard-surface
       model and on the "cube map" boundaries of a smooth one.
    3. **Cut between clusters, and absorb the slivers.**  The boundary of a face
       partition is a set of closed loops (or paths ending on a real boundary),
       which is why this cut is short where a spanning-tree cut is long.  A
       cluster can land in several disconnected places, though, so any chart
       below ``min_area_share`` of the mean is merged back into its largest
       neighbour -- see :data:`_CHART_MIN_AREA_SHARE`.
    4. **Bisect until every chart is a disk.**  For a chart with ``F`` polygons,
       ``U`` uncut interior mesh edges and ``Vint`` vertices touching no cut,
       boundary or non-manifold edge, the Euler characteristic is exactly
       ``chi = F - U + Vint``; a connected orientable chart is a disk precisely
       when ``chi == 1`` (``chi = 2 - 2*genus - boundary_loops``).  Any chart
       that is not is split in half along its principal axis and the test is
       repeated.  An annulus -- the top half of a torus -- becomes two disks in
       one round.

    Args:
        sample: The mesh to cut.  Nothing is mutated.
        topo: Prebuilt edge topology for ``sample``.
        clusters: Number of seed clusters.  More clusters means more, smaller,
            flatter charts.
        seed: Rotates the seed directions; different seeds give different charts.
        max_splits: Guard on the bisection loop.
        min_area_share: Charts smaller than this fraction of the mean chart area
            are absorbed into a neighbour.  ``0`` disables absorption.

    Returns:
        ``(E,)`` bool.  Never raises; an empty or degenerate mesh yields an
        all-``False`` mask.
    """
    n_edges = int(topo.n_edges)
    n_tris = int(topo.n_tris)
    seam = np.zeros(max(n_edges, 0), dtype=bool)
    if n_tris == 0 or n_edges == 0:
        return seam

    verts = np.asarray(sample.verts, dtype=np.float64).reshape(-1, 3)
    if verts.shape[0] == 0:
        return seam
    tri_verts = np.clip(
        np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3), 0, verts.shape[0] - 1
    )

    # (1) polygons
    group = _label_components(n_tris, topo.diag_a, topo.diag_b)
    is_rep = group == np.arange(n_tris, dtype=np.int64)

    areas = np.asarray(topo.areas, dtype=np.float64)
    weighted_normals = np.asarray(topo.normals, dtype=np.float64) * areas[:, None]
    poly_normal = np.zeros((n_tris, 3), dtype=np.float64)
    np.add.at(poly_normal, group, weighted_normals)
    poly_centre = np.zeros((n_tris, 3), dtype=np.float64)
    np.add.at(poly_centre, group, verts[tri_verts].mean(axis=1) * areas[:, None])
    poly_area = np.zeros(n_tris, dtype=np.float64)
    np.add.at(poly_area, group, areas)
    poly_centre /= np.where(poly_area > 0.0, poly_area, 1.0)[:, None]
    length = np.linalg.norm(poly_normal, axis=1)
    live = length > 0.0
    unit = np.where(live[:, None], poly_normal / np.where(live, length, 1.0)[:, None], 0.0)

    # (2) area-weighted k-means over polygon normals
    dirs = _seed_directions(clusters, seed)
    label = np.argmax(unit @ dirs.T, axis=1).astype(np.int64)
    for _ in range(12):
        accum = np.zeros_like(dirs)
        np.add.at(accum, label, poly_normal)
        norms = np.linalg.norm(accum, axis=1)
        keep = norms > 1e-12
        dirs = np.where(keep[:, None], accum / np.where(keep, norms, 1.0)[:, None], dirs)
        nxt = np.argmax(unit @ dirs.T, axis=1).astype(np.int64)
        if np.array_equal(nxt, label):
            break
        label = nxt
    tri_label = label[group]

    manifold = np.flatnonzero(topo.face_count == 2)
    forced = topo.face_count >= 3
    opening = (topo.face_count == 1) | forced
    face_a = np.asarray(topo.face_a, dtype=np.int64)
    face_b = np.asarray(topo.face_b, dtype=np.int64)
    mf_a = face_a[manifold]
    mf_b = face_b[manifold]

    # (3b) absorb slivers
    if min_area_share > 0.0:
        tri_label = _absorb_slivers(
            tri_label, topo, manifold, mf_a, mf_b, float(min_area_share)
        )
    next_label = int(tri_label.max()) + 1 if tri_label.size else 1

    for _ in range(max(int(max_splits), 0) + 1):
        # (3) cut between clusters
        cross = tri_label[mf_a] != tri_label[mf_b]
        seam = np.zeros(n_edges, dtype=bool)
        seam[manifold[cross]] = True
        seam |= forced

        uncut = manifold[~cross]
        # Diagonals never separate a polygon, so they always join two charts.
        chart = _label_components(
            n_tris,
            np.concatenate([face_a[uncut], topo.diag_a]),
            np.concatenate([face_b[uncut], topo.diag_b]),
        )

        # (4) chi = F - U + Vint, per chart
        f_count = np.bincount(chart[is_rep], minlength=n_tris)
        u_count = np.bincount(chart[face_a[uncut]], minlength=n_tris)
        blocked = seam | opening
        touched = np.zeros(topo.n_verts, dtype=bool)
        if blocked.any():
            touched[topo.edge_v0[blocked]] = True
            touched[topo.edge_v1[blocked]] = True
        vert_face = np.full(topo.n_verts, -1, dtype=np.int64)
        vert_face[topo.edge_v0[uncut]] = face_a[uncut]
        vert_face[topo.edge_v1[uncut]] = face_a[uncut]
        interior = (~touched) & (vert_face >= 0)
        v_count = np.bincount(chart[vert_face[interior]], minlength=n_tris)

        broken = (f_count > 0) & ((f_count - u_count + v_count) != 1)
        if not broken.any():
            break

        promoted = np.zeros(n_tris, dtype=np.int64)
        changed = False
        for chart_id in np.flatnonzero(broken).tolist():
            members = np.flatnonzero((chart == chart_id) & is_rep)
            if members.size < 2:
                continue
            points = poly_centre[members]
            centred = points - points.mean(axis=0)
            try:
                axis = np.linalg.eigh(centred.T @ centred)[1][:, -1]
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                axis = np.array([1.0, 0.0, 0.0])
            order = np.argsort(centred @ axis, kind="stable")
            promoted[members[order[members.size // 2:]]] = next_label
            next_label += 1
            changed = True
        if not changed:
            break
        moved = promoted[group]
        tri_label = np.where(moved > 0, moved, tri_label)
    return seam


def _absorb_slivers(
    tri_label: np.ndarray,
    topo: "seam_mod.EdgeTopology",
    manifold: np.ndarray,
    mf_a: np.ndarray,
    mf_b: np.ndarray,
    min_area_share: float,
) -> np.ndarray:
    """Merge every under-sized chart into its largest neighbour.

    Charts are the connected components of the same-label face graph, so a
    cluster that lands in three places is three charts and only the big one is
    worth its own island.  Merging is done by relabelling the small chart's
    faces to the neighbour's label, which merges exactly those two components --
    a distant component that happens to share the label stays separate, because
    charts are defined by adjacency, not by label.

    Args:
        tri_label: ``(T,)`` current label per triangle.
        topo: Prebuilt edge topology.
        manifold: Indices of the manifold mesh edges.
        mf_a: First incident face of each manifold edge.
        mf_b: Second incident face of each manifold edge.
        min_area_share: Threshold as a fraction of the mean chart area.

    Returns:
        A new ``(T,)`` label array.
    """
    labels = np.array(tri_label, dtype=np.int64, copy=True)
    areas = np.asarray(topo.areas, dtype=np.float64)
    n_tris = topo.n_tris
    for _ in range(_CHART_ABSORB_ROUNDS):
        cross = labels[mf_a] != labels[mf_b]
        uncut = manifold[~cross]
        chart = _label_components(
            n_tris,
            np.concatenate([topo.face_a[uncut], topo.diag_a]),
            np.concatenate([topo.face_b[uncut], topo.diag_b]),
        )
        chart_area = np.bincount(chart, weights=areas, minlength=n_tris)
        live = chart_area > 0.0
        if int(live.sum()) < 2:
            return labels
        small = live & (chart_area < chart_area[live].mean() * min_area_share)
        if not small.any():
            return labels

        cut = manifold[cross]
        side_a, side_b = chart[topo.face_a[cut]], chart[topo.face_b[cut]]
        # For every small chart, the adjacent chart with the largest area wins.
        donor = np.where(small[side_a], side_a, side_b)
        host = np.where(small[side_a], side_b, side_a)
        usable = (small[side_a] | small[side_b]) & (side_a != side_b)
        donor, host = donor[usable], host[usable]
        if donor.size == 0:
            return labels
        order = np.lexsort((chart_area[host], donor))
        donor, host = donor[order], host[order]
        keep = np.empty(donor.size, dtype=bool)
        keep[-1] = True
        keep[:-1] = donor[1:] != donor[:-1]  # last of each run = largest host
        donor, host = donor[keep], host[keep]

        # A chain of merges (small -> small -> big) is resolved next round.
        target = np.arange(n_tris, dtype=np.int64)
        target[donor] = host
        move = target[chart] != chart
        if not move.any():
            return labels
        # Every face of a donor chart takes its host's label.  All faces of one
        # chart carry the same label by construction, so this scatter is
        # well-defined whichever representative lands last.
        chart_label = np.zeros(n_tris, dtype=np.int64)
        chart_label[chart] = labels
        labels = np.where(move, chart_label[target[chart]], labels)
    return labels


def chart_cut_request(params: UnwrapParams) -> tuple[int, int] | None:
    """Return ``(clusters, seed)`` when ``params`` asks for a chart cut.

    The request travels in ``params.extra``, which already participates in
    :meth:`UnwrapParams.signature`, so two chart recipes differing only in their
    cluster count de-duplicate correctly.  Projection methods ignore seams
    entirely, so a chart request on one of them is meaningless and is refused
    here rather than silently producing a duplicate of the plain recipe.
    """
    if params.method not in _SEAM_DRIVEN:
        return None
    extra = params.extra if isinstance(params.extra, dict) else {}
    raw = extra.get(CHART_CUT_KEY)
    if raw is None:
        return None
    try:
        clusters = int(raw)
        seed = int(extra.get(CHART_SEED_KEY, 0) or 0)
    except (TypeError, ValueError):
        return None
    if clusters < 1:
        return None
    return clusters, seed


class _CutCache:
    """Memoises chart cuts within one generation run.

    A ten-slot grid on a closed mesh asks for several chart cuts and the refill
    rounds ask again; recomputing a cut per request would repeat the clustering
    and the bisection loop for nothing.
    """

    __slots__ = ("_sample", "_topo", "_cache")

    def __init__(self, sample: UVMeshSample, topo: "seam_mod.EdgeTopology") -> None:
        self._sample = sample
        self._topo = topo
        self._cache: dict[tuple[int, int], np.ndarray | None] = {}

    def get(self, clusters: int, seed: int) -> np.ndarray | None:
        """Return the cut for ``(clusters, seed)``, or ``None`` if it raised.

        An *empty* cut is a legitimate answer, not a failure: a mesh that is
        already one disk -- a plane, a grid, an open shell -- needs no seams at
        all.  A closed mesh cannot come back empty, because the bisection loop
        only stops once every chart has ``chi == 1`` and a closed component
        starts at 2.
        """
        key = (int(clusters), int(seed))
        if key not in self._cache:
            try:
                mask = chart_seams(
                    self._sample, self._topo, clusters=key[0], seed=key[1]
                )
            except Exception:  # noqa: BLE001 - a cut failure must not kill the run
                mask = None
            self._cache[key] = mask
        return self._cache[key]


def repair_recipe(
    params: UnwrapParams,
    sample: UVMeshSample,
    topo: "seam_mod.EdgeTopology",
) -> tuple[UnwrapParams, str]:
    """Rewrite a recipe that provably cannot work on *this* mesh.

    Two cases are provably dead before a solver is ever started:

    * **A closed surface with an empty seam set.**  There is no boundary, so a
      seam-driven solver has no disk to flatten and gives up -- and, because
      Blender reports success anyway (see :func:`dead_layout_reason`), the slot
      is wasted silently.  On a UV sphere that was six of ten slots.
    * **``EXISTING`` on a mesh with no marked seams.**  The recipe's entire
      premise -- "respect what the artist marked" -- is void, and what comes back
      is either the same dead layout or, on an open mesh, a single sheet
      stretched over the whole model (measured L2 stretch of 50 on Suzanne).

    Both escalate to a :func:`chart_seams` cut, which is guaranteed to open every
    component into disks.  The cluster count and rotation are derived from the
    recipe's own signature, so several recipes escalating in one run produce
    genuinely *different* charts instead of collapsing onto one layout and being
    de-duplicated away.

    Anything that might work is left untouched -- in particular ``NONE`` on an
    open surface, which is the legitimate "flatten the whole thing as one sheet"
    control case.

    Args:
        params: The recipe to check.
        sample: The mesh it will run against.
        topo: Prebuilt edge topology for ``sample``.

    Returns:
        ``(params, note)``. ``note`` is empty when nothing was changed.
    """
    if params.method not in _SEAM_DRIVEN:
        return params, ""
    if sample.n_edges == 0:
        return params, ""
    if chart_cut_request(params) is not None:
        return params, ""
    try:
        # ``boundary_mask`` is a property, not a method -- calling it used to
        # raise TypeError straight into the except below, which silently
        # disabled this whole repair path.
        closed = not bool(np.asarray(topo.boundary_mask).any())
        mask = seam_mod.compute_seams(
            sample,
            params.seam_strategy,
            angle_deg=params.seam_angle_deg,
            curvature_threshold=params.curvature_threshold,
            existing=sample.edge_seam,
            seed=params.seed,
        )
        empty = not bool(np.asarray(mask).any())
    except Exception:  # noqa: BLE001 - a probe failure must not kill the run
        return params, ""

    if not empty:
        return params, ""
    if not closed and params.seam_strategy != "EXISTING":
        # An open surface can be flattened without extra cuts; only EXISTING is
        # escalated there, because "keep nothing" is not a recipe.
        return params, ""

    spread = int(params.signature()[:8], 16)
    clusters = _CHART_CLUSTERS[spread % len(_CHART_CLUSTERS)]
    extra = dict(params.extra) if isinstance(params.extra, dict) else {}
    extra[CHART_CUT_KEY] = int(clusters)
    extra[CHART_SEED_KEY] = int((spread >> 8) & 0xFFFF) + 1
    return (
        params.replace(seam_strategy="EXISTING", extra=extra),
        "%s yields no seams%s; escalated to a %d-chart cut"
        % (params.seam_strategy, " on a closed mesh" if closed else "", clusters),
    )


def _resolve_backend_id(params: UnwrapParams, allowed: Sequence[str] | None) -> str:
    """Pick the backend for ``params``, honouring an optional allow-list."""
    wanted = params.backend or "blender_native"
    if allowed and wanted not in allowed:
        return allowed[0]
    return wanted


def _run_recipes(
    obj: Any,
    recipes: Sequence[UnwrapParams],
    *,
    weights: ScoreWeights,
    backends: Sequence[str] | None,
    progress: ProgressFn | None,
    should_cancel: CancelFn | None,
    compute_overlap: bool,
    parent_signature: str | None,
    generation: int,
) -> GenerationReport:
    """Execute a list of recipes against ``obj`` and return a ranked report.

    This is the shared body of :func:`generate` and :func:`branch`; the only
    thing that differs between them is how the recipe list was produced.
    """
    from . import bridge  # imported lazily: only this path needs bpy

    started = time.perf_counter()
    report = GenerationReport()

    ok, why = validate_object(obj)
    if not ok:
        report.warnings.append(why)
        report.seconds = time.perf_counter() - started
        return report

    snapshot = None
    try:
        snapshot = bridge.snapshot_state(obj)
    except Exception as exc:  # noqa: BLE001 - never take Blender down
        report.warnings.append("Could not snapshot mesh state: %s" % exc)

    try:
        sample = bridge.extract_sample(obj)
        problems = sample.validate()
        # "no triangles" is fatal; everything else is worth saying out loud but
        # is survivable -- degenerate faces are excluded from the metrics.
        for problem in problems:
            report.warnings.append("Geometry: %s" % problem)
        if sample.n_tris == 0:
            report.seconds = time.perf_counter() - started
            return report

        topo = seam_mod.build_edge_topology(sample)
        cuts = _CutCache(sample, topo)
        total = max(len(recipes), 1)
        # Tracks the last layout a backend produced, so a solver that
        # silently does nothing and inherits it can be detected.
        previous_uvs: np.ndarray | None = None
        seen_signatures: set[str] = set()

        for i, params in enumerate(recipes):
            if should_cancel is not None:
                try:
                    if should_cancel():
                        report.cancelled = True
                        break
                except Exception:  # noqa: BLE001 - a bad probe must not abort
                    pass

            if progress is not None:
                try:
                    progress(i / total, "Unwrapping %d/%d: %s"
                             % (i + 1, total, params.describe()))
                except Exception:  # noqa: BLE001 - UI callback is advisory
                    pass

            # Rewrite recipes that cannot possibly work on this mesh, and
            # de-duplicate: an escalation can collide with an existing recipe.
            params, note = repair_recipe(params, sample, topo)
            if note:
                report.warnings.append("%s: %s" % (params.describe(), note))
            if params.signature() in seen_signatures:
                report.failures.append(
                    (params, "duplicate of another candidate after repair")
                )
                continue
            seen_signatures.add(params.signature())

            backend_id = _resolve_backend_id(params, backends)
            try:
                backend = get_backend(backend_id)
            except Exception as exc:  # noqa: BLE001
                report.failures.append((params, "backend %r unavailable: %s"
                                        % (backend_id, exc)))
                continue

            available, why_not = backend.is_available()
            if not available:
                report.failures.append((params, "%s: %s" % (backend_id, why_not)))
                continue

            # A chart-cut recipe carries its seams in the sample rather than in
            # a strategy name: the backend stamps ``sample.edge_seam`` when the
            # strategy is EXISTING, so handing it a re-seamed copy is all it
            # takes.  The mesh itself is untouched -- the sample is a value.
            request_sample = sample
            wanted_cut = chart_cut_request(params)
            if wanted_cut is not None:
                mask = cuts.get(*wanted_cut)
                if mask is None:
                    report.failures.append(
                        (params, "chart cut (%d clusters) could not be computed"
                         % wanted_cut[0])
                    )
                    continue
                request_sample = sample.with_seams(mask)

            result = backend.safe_unwrap(
                UnwrapRequest(obj_name=obj.name, params=params, sample=request_sample)
            )
            if not result.ok or result.uvs is None:
                report.failures.append((params, result.message or "backend failed"))
                continue

            uvs = np.asarray(result.uvs, dtype=np.float32).reshape(-1, 2)
            if uvs.shape[0] != sample.n_loops:
                report.failures.append(
                    (params, "backend returned %d UVs, expected %d"
                     % (uvs.shape[0], sample.n_loops))
                )
                continue

            # Blender can report success while doing nothing at all; see
            # dead_layout_reason. Compare against the previous candidate's UVs
            # too, because an unsolved unwrap silently inherits them.
            dead = dead_layout_reason(uvs, previous_uvs)
            if dead:
                report.failures.append((params, dead))
                previous_uvs = uvs
                continue
            previous_uvs = uvs

            try:
                seam_mask = derive_uv_seams(sample, topo, uvs)
                measured = sample.with_uvs(uvs).with_seams(seam_mask)
                metrics = evaluate(measured, compute_overlap=compute_overlap)
                value = score(metrics, weights)
            except Exception as exc:  # noqa: BLE001 - a metric bug must not
                # discard an otherwise valid unwrap; keep it, unranked.
                report.warnings.append("Scoring failed for %s: %s"
                                       % (params.describe(), exc))
                metrics, value = None, 0.0

            report.candidates.append(
                Candidate(
                    index=len(report.candidates),
                    params=params,
                    uvs=uvs,
                    metrics=metrics,
                    score=value,
                    ok=True,
                    message=result.message,
                    parent=parent_signature,
                    generation=generation,
                    seconds=result.seconds,
                )
            )
    except Exception as exc:  # noqa: BLE001 - containment is the whole point
        report.warnings.append("Generation aborted: %s: %s" % (type(exc).__name__, exc))
    finally:
        # Always, always put the mesh back.
        if snapshot is not None:
            try:
                bridge.restore_state(obj, snapshot)
            except Exception as exc:  # noqa: BLE001
                report.warnings.append("Could not restore mesh state: %s" % exc)
        if progress is not None:
            try:
                progress(1.0, "Done")
            except Exception:  # noqa: BLE001
                pass

    _rank_in_place(report.candidates, weights)
    report.seconds = time.perf_counter() - started
    return report


def _rank_in_place(candidates: list[Candidate], weights: ScoreWeights) -> None:
    """Sort candidates best-first, renumber them, and tag the Pareto front.

    Slot numbers follow the ranking so slot 0 is always the best pick.  A
    stable secondary key on the parameter signature keeps the order
    reproducible when two candidates score identically.
    """
    if not candidates:
        return
    candidates.sort(key=lambda c: (-float(c.score or 0.0), c.signature))

    measured = [c for c in candidates if c.metrics is not None]
    front: set[int] = set()
    if measured:
        try:
            picks = pareto_front([c.metrics for c in measured])
            front = {id(measured[i]) for i in picks if 0 <= i < len(measured)}
        except Exception:  # noqa: BLE001 - ranking is advisory, never fatal
            front = set()

    for i, cand in enumerate(candidates):
        cand.index = i
        # Recorded on the candidate so the UI can badge it without recomputing.
        cand.pareto = id(cand) in front  # type: ignore[attr-defined]


#: How many refill rounds `generate` will run before giving up on reaching the
#: requested candidate count. Each round costs real solver time, so this is a
#: guard against pathological meshes where almost every recipe fails.
_MAX_REFILL_ROUNDS = 4


def _merge(into: GenerationReport, part: GenerationReport, cap: int) -> None:
    """Fold one batch's results into the accumulating report, up to `cap`."""
    seen = {c.signature for c in into.candidates}
    for cand in part.candidates:
        if len(into.candidates) >= cap:
            break
        if cand.signature in seen:
            continue
        seen.add(cand.signature)
        into.candidates.append(cand)
    into.failures.extend(part.failures)
    for note in part.warnings:
        if note not in into.warnings:
            into.warnings.append(note)
    into.seconds += part.seconds


def generate(
    obj: Any,
    *,
    count: int = 10,
    seed: int = 0,
    weights: ScoreWeights | None = None,
    base: UnwrapParams | None = None,
    backends: Sequence[str] | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    compute_overlap: bool = True,
    store: bool = True,
) -> GenerationReport:
    """Produce ``count`` distinct unwrap candidates for ``obj``.

    Args:
        obj: The Blender object to unwrap.  Must be a mesh with faces.
        count: How many candidates to generate.
        seed: Determinism seed -- the same seed reproduces the same grid.
        weights: Scoring weights; defaults to the ``balanced`` preset.
        base: Optional starting recipe whose unsearched fields are inherited.
        backends: Optional allow-list of backend ids.
        progress: Optional ``(fraction, message)`` progress callback.
        should_cancel: Optional probe; return ``True`` to stop early.
        compute_overlap: Whether to run the (more expensive) overlap test.
        store: Whether to write the results into the object's candidate store.

    Returns:
        A :class:`GenerationReport`.  Never raises -- inspect ``warnings`` and
        ``failures`` to find out what went wrong.
    """
    weights = weights or ScoreWeights.preset("balanced")
    target = max(int(count), 0)
    if target == 0:
        return GenerationReport()

    # Deliver `target` USABLE candidates, not `target` attempts. Recipes that a
    # given mesh cannot solve are rejected rather than shown as dead slots, so
    # without a refill pass a closed mesh would hand back a half-empty grid.
    report = GenerationReport()
    tried: set[str] = set()
    round_seed = int(seed)
    # Each refill round repeats the per-run setup (snapshot, extract, topology),
    # so rounds are not free. Without a total budget a SMALL request could cost
    # more than a large one -- measured at 26.2s for count=4 against 8.7s for
    # count=10 on a cube, because the small request needed more rounds. Bound
    # the total recipes attempted and stop as soon as a round adds nothing.
    attempt_budget = max(target * 2, target + 6)
    attempted = 0
    for attempt in range(_MAX_REFILL_ROUNDS):
        want = target - len(report.candidates)
        if want <= 0 or attempted >= attempt_budget:
            break
        # Over-request modestly on refills, since some will fail again, but
        # never beyond the remaining budget.
        ask = want if attempt == 0 else min(want * 2, attempt_budget - attempted)
        if ask <= 0:
            break
        batch = sample_grid(ask, seed=round_seed, base=base)
        batch = [p for p in batch if p.signature() not in tried]
        tried.update(p.signature() for p in batch)
        if not batch:
            round_seed += 977
            continue
        attempted += len(batch)
        before = len(report.candidates)

        part = _run_recipes(
            obj,
            batch,
            weights=weights,
            backends=backends,
            progress=progress if attempt == 0 else None,
            should_cancel=should_cancel,
            compute_overlap=compute_overlap,
            parent_signature=None,
            generation=0,
        )
        _merge(report, part, target)
        if part.cancelled:
            report.cancelled = True
            break
        if len(report.candidates) == before:
            # This round contributed nothing usable; more rounds of the same
            # will not either, and each one costs a full setup.
            break
        round_seed += 977

    _rank_in_place(report.candidates, weights)
    if store and report.candidates:
        _commit(obj, report)
    return report


def branch(
    obj: Any,
    parent: Candidate,
    *,
    count: int = 10,
    strength: float = 0.35,
    seed: int = 0,
    weights: ScoreWeights | None = None,
    backends: Sequence[str] | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    compute_overlap: bool = True,
    store: bool = True,
    keep_parent: bool = True,
) -> GenerationReport:
    """Produce ``count`` mutations clustered around ``parent``.

    This is the "10 more like that one" half of the workflow.  The parent is
    carried into the new set by default so the artist can always tell whether a
    branch actually improved on what they started from -- losing the reference
    point is the fastest way to make an iterative tool feel random.

    Args:
        obj: The Blender object to unwrap.
        parent: The candidate to vary around.
        count: How many children to generate.
        strength: Mutation magnitude in ``[0, 1]``.
        seed: Determinism seed.
        weights: Scoring weights; defaults to the ``balanced`` preset.
        backends: Optional allow-list of backend ids.
        progress: Optional progress callback.
        should_cancel: Optional cancellation probe.
        compute_overlap: Whether to run the overlap test.
        store: Whether to write results into the candidate store.
        keep_parent: Whether to re-include the parent in the new set.

    Returns:
        A :class:`GenerationReport`.  Never raises.
    """
    weights = weights or ScoreWeights.preset("balanced")
    if parent is None:
        report = GenerationReport()
        report.warnings.append("No parent candidate selected to branch from.")
        return report

    n_children = max(int(count), 0)
    if keep_parent and n_children > 0:
        n_children -= 1
    recipes = mutate(
        parent.params, count=n_children, strength=float(strength), seed=int(seed)
    )
    if keep_parent:
        recipes = [parent.params, *recipes]

    report = _run_recipes(
        obj,
        recipes,
        weights=weights,
        backends=backends,
        progress=progress,
        should_cancel=should_cancel,
        compute_overlap=compute_overlap,
        parent_signature=parent.signature,
        generation=int(getattr(parent, "generation", 0)) + 1,
    )
    if store and report.candidates:
        _commit(obj, report)
    return report


def _commit(obj: Any, report: GenerationReport) -> None:
    """Write a report's candidates into the object's store, tolerantly."""
    try:
        store = CandidateStore.for_object(obj)
        store.set(report.candidates)
    except Exception as exc:  # noqa: BLE001 - results are still returned
        report.warnings.append("Could not store candidates: %s" % exc)
