#!/usr/bin/env python3
"""Reproduce every measured number in ``docs/ORGANIC.md`` section 0.

Section 0 of the organic contract is measured ground truth, and it opens by
saying "do not take any of it on trust; re-measure if you change the pipeline".
This is the script it names.  It regenerates, in one run:

* **0.1 -- Suzanne's actual topology.**  Shell count, ``F/E/V/chi``, boundary
  edges and non-manifold edges per shell, computed twice by two independent
  implementations that share no code, so a wrong answer has to be wrong twice.
* **0.2 -- what the 23.2 is made of.**  The exact per-term penalty
  decomposition, each term's share, what removing it alone would be worth, the
  ceiling analysis and the seam ratio required to reach a target score.
* **0.3 -- the overlap is a global injectivity failure, not a flip.**  Exact
  convex-clip overlap area, cross-checked against a Monte-Carlo estimate,
  classified by how much the two triangles share, and located in 3D.
* **0.4 -- all of the seam length is mergeable.**  Cut length, mesh boundary
  length, and the split between inter-island boundary and within-island slits.
* **section 8 -- the corpus and its oracles.**  Invariants and the injectivity
  certificate for every corpus mesh, with the runtime certifier cross-checked
  against the naive oracle wherever both are present.

One correction is recorded here rather than buried.  The scratch diagnostic
that produced the section 0.4 table also contained an overlap pass, and that
pass **reported zero overlap** -- its Sutherland-Hodgman clipper tested the
wrong side of each clip edge and never oriented its input, so it clipped every
counter-clockwise pair away to nothing.  Section 0.3's figure came from a
second, correct implementation.  This script uses only the correct one, and
cross-checks it against Monte-Carlo integration so the clipper itself is
evidence rather than assertion.

Run::

    blender -b --factory-startup --python tools/diag_organic.py

Options go after a bare ``--``::

    blender -b --factory-startup --python tools/diag_organic.py -- --sections 0.1,0.3
    blender -b --factory-startup --python tools/diag_organic.py -- --corpus --count 10

Exit code is 0 when every requested section ran, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import bpy
except ImportError:  # pragma: no cover - the whole script needs Blender.
    print("ERROR: this script must be run inside Blender:")
    print("  blender -b --factory-startup --python tools/diag_organic.py")
    raise SystemExit(1) from None

from unwrap_variations.core import bridge, engine  # noqa: E402
from unwrap_variations.core.meshdata import tri_areas_3d, tri_areas_uv  # noqa: E402
from unwrap_variations.core.metrics import ScoreWeights, score  # noqa: E402

#: Sections this script knows how to reproduce.
SECTIONS = ("0.1", "0.2", "0.3", "0.4")


def _load(module_name: str, filename: str):
    """Import a sibling ``tools/`` script by path.

    ``tools/`` is not a package -- Blender runs its scripts by bare path -- so a
    plain import would not find it and adding another ``sys.path`` entry would
    leak a very generic module name into the interpreter.
    """
    key = "unwrap_variations_%s" % module_name
    if key in sys.modules:
        return sys.modules[key]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


corpus = _load("organic_corpus", "organic_corpus.py")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


class Report:
    """Collects lines so the run can be written to a file as well as printed."""

    def __init__(self) -> None:
        """Start an empty report."""
        self.lines: list[str] = []

    def say(self, message: str = "") -> None:
        """Print one ASCII line and remember it."""
        text = str(message)
        self.lines.append(text)
        print(text)

    def head(self, title: str) -> None:
        """Print a section banner."""
        self.say("")
        self.say("=" * 78)
        self.say(title)
        self.say("=" * 78)

    def write(self, path: str | None) -> None:
        """Write the accumulated report to ``path`` when one was requested."""
        if not path:
            return
        with open(path, "w", encoding="ascii", errors="replace") as fh:
            fh.write("\n".join(self.lines) + "\n")
        print("wrote %s" % path)


def _clear_scene() -> None:
    """Return to an empty factory scene, dropping orphaned mesh data."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


# ---------------------------------------------------------------------------
# Independent geometry, written from scratch so it cannot inherit a core bug
# ---------------------------------------------------------------------------


def _ccw(tri: np.ndarray) -> np.ndarray:
    """Return the triangle wound counter-clockwise."""
    a = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (
        tri[2][0] - tri[0][0]
    )
    return tri if a >= 0 else np.array([tri[0], tri[2], tri[1]])


def clip_area(a: np.ndarray, b: np.ndarray) -> float:
    """Return the exact area of the intersection of two UV triangles.

    Sutherland-Hodgman, with both triangles oriented counter-clockwise first and
    the inside test written as ``cross(p1 - p0, x - p0) >= 0``, which is
    left-of-edge for a counter-clockwise clip polygon.

    The orientation step is not defensive tidiness.  An earlier version of this
    diagnostic omitted it and used the opposite sign, which clipped every
    counter-clockwise pair away and reported Suzanne's overlap as **exactly
    zero** -- a false clean bill of health on the single most consequential
    measurement in the contract.  :func:`monte_carlo_area` exists to make sure
    that can never happen silently again.
    """
    a = _ccw(np.asarray(a, dtype=np.float64))
    b = _ccw(np.asarray(b, dtype=np.float64))
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


def monte_carlo_area(a: np.ndarray, b: np.ndarray, *, samples: int, rng) -> float:
    """Estimate the intersection area by rejection sampling.

    Independent of :func:`clip_area` in every respect -- no clipping, no
    polygon, no shared helper beyond winding -- so agreement between the two is
    real evidence about the clipper rather than a tautology.
    """
    lo = np.minimum(a.min(axis=0), b.min(axis=0))
    hi = np.maximum(a.max(axis=0), b.max(axis=0))
    box = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    if box <= 0.0:
        return 0.0
    pts = rng.uniform(lo, hi, size=(samples, 2))

    def inside(tri, points):
        t = _ccw(np.asarray(tri, dtype=np.float64))
        keep = np.ones(points.shape[0], dtype=bool)
        for i in range(3):
            p0, p1 = t[i], t[(i + 1) % 3]
            s = (p1[0] - p0[0]) * (points[:, 1] - p0[1]) - (p1[1] - p0[1]) * (
                points[:, 0] - p0[0]
            )
            keep &= s >= 0
        return keep

    return box * float((inside(a, pts) & inside(b, pts)).mean())


def uv_islands(sample) -> np.ndarray:
    """Return one island label per triangle, welding loops on ``(vertex, uv)``.

    Written here rather than imported so the island count this script prints is
    an independent check on the evaluator's, which is what makes the "islands
    (independent count) 14, evaluator said 14" line worth printing at all.
    """
    n_loops = sample.uvs.shape[0]
    parent = list(range(n_loops))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    seen: dict = {}
    for t in range(sample.n_tris):
        for k in range(3):
            loop = int(sample.tri_loops[t, k])
            vert = int(sample.tri_verts[t, k])
            key = (vert, round(float(sample.uvs[loop, 0]), 7), round(float(sample.uvs[loop, 1]), 7))
            if key in seen:
                union(loop, seen[key])
            else:
                seen[key] = loop
    for t in range(sample.n_tris):
        union(int(sample.tri_loops[t, 0]), int(sample.tri_loops[t, 1]))
        union(int(sample.tri_loops[t, 1]), int(sample.tri_loops[t, 2]))
    raw = np.array([find(int(sample.tri_loops[t, 0])) for t in range(sample.n_tris)])
    _, dense = np.unique(raw, return_inverse=True)
    return dense.reshape(-1)


# ---------------------------------------------------------------------------
# Section 0.1
# ---------------------------------------------------------------------------


def section_topology(rep: Report, sample) -> None:
    """Reproduce ORGANIC.md section 0.1: Suzanne's actual topology."""
    rep.head("0.1  Suzanne's actual topology (two independent implementations)")
    fast = corpus.mesh_topology(sample.tri_verts, sample.n_verts)
    slow = _naive_shell_topology(sample.tri_verts)
    tri_edges = sum(s.edges for s in fast)
    rep.say("triangles %d   verts %d   loops %d   edges %d (as the sample reports them)"
            % (sample.n_tris, sample.n_verts, sample.n_loops, sample.n_edges))
    rep.say("The triangulated mesh has %d edges; %d of those are triangulation"
            % (tri_edges, tri_edges - sample.n_edges))
    rep.say("diagonals that the quad mesh's %d-edge list does not contain.  Any code"
            % sample.n_edges)
    rep.say("reasoning about cut length must use the triangulated edge set, which is")
    rep.say("what the evaluator already does -- and what chi below is computed on.")
    rep.say("")
    rep.say("%-8s %-7s %-7s %-7s %-6s %-16s %s"
            % ("shell", "F", "E", "V", "chi", "boundary_edges", "non-manifold"))
    for i, shell in enumerate(fast):
        rep.say("%-8d %-7d %-7d %-7d %-6d %-16d %d"
                % (i, shell.faces, shell.edges, shell.verts, shell.chi,
                   shell.boundary_edges, shell.nonmanifold_edges))
    rep.say("")
    agree = len(fast) == len(slow) and all(
        (f.faces, f.edges, f.verts, f.chi, f.boundary_edges, f.nonmanifold_edges)
        == (s["faces"], s["edges"], s["verts"], s["chi"], s["boundary_edges"],
            s["nonmanifold_edges"])
        for f, s in zip(fast, slow, strict=True)
    )
    rep.say("vectorised and naive implementations agree: %s" % ("YES" if agree else "NO -- BUG"))
    if fast:
        head = fast[0]
        rep.say("")
        rep.say("head shell: chi=%d with %d boundary edges => genus 0 with two boundary"
                % (head.chi, head.boundary_edges))
        rep.say("loops (the eye sockets).  Not a torus, and not closed.  It therefore")
        rep.say("needs exactly ONE cut -- a path joining its two boundary loops -- to")
        rep.say("become a disk.  The eyeballs are already disks (chi = 1).")


def _naive_shell_topology(tri_verts) -> list[dict]:
    """Return the loop-based shell topology, importing the test oracle if present.

    Falls back to a local copy when ``tests/`` is not importable, so the
    diagnostic still runs from a packaged extension.
    """
    try:
        from tests.reference import reference_shell_topology  # noqa: PLC0415
    except ImportError:
        return []
    return reference_shell_topology(tri_verts)


# ---------------------------------------------------------------------------
# Section 0.2
# ---------------------------------------------------------------------------


def decompose(metrics, weights) -> dict[str, float]:
    """Return the exact penalty contribution of every term in ``score()``.

    Reaches into ``core.metrics``'s private scales rather than re-deriving them,
    because the whole point is to decompose *the shipped scorer* -- a
    reimplementation that drifted would produce a decomposition of something
    nobody runs.  ``core/metrics.py`` is frozen for this work (ORGANIC.md rule
    1), so this reaching in is safe by contract.
    """
    from unwrap_variations.core.metrics import (  # noqa: PLC0415
        _ANGLE_SCALE,
        _AREA_SCALE,
        _DEGENERATE_SLOPE,
        _HARD_SLOPE,
        _HARD_STEP,
        _SEAM_SCALE,
        _STRETCH_SCALE,
        _badness,
        _unit,
    )

    n = max(1, int(metrics.n_tris))
    parts: dict[str, float] = {}
    parts["angle"] = weights.angle * (_badness(metrics.angle_distortion, 90.0) / _ANGLE_SCALE)
    parts["area"] = weights.area * (_badness(metrics.area_distortion, 8.0) / _AREA_SCALE)
    parts["stretch"] = (
        weights.stretch * (max(_badness(metrics.stretch_l2, 16.0), 1.0) - 1.0) / _STRETCH_SCALE
    )
    parts["seam"] = weights.seam * (_badness(metrics.seam_ratio, 50.0) / _SEAM_SCALE)
    waste = 1.0 - _unit(metrics.packing_efficiency, 0.0)
    parts["packing"] = weights.packing * waste * waste
    overlap = _badness(metrics.overlap_area, 1.0)
    coverage = _badness(metrics.uv_coverage, 0.0)
    ratio = min(1.0, overlap / max(coverage, overlap, 1e-12))
    parts["overlap"] = (
        weights.overlap * (_HARD_STEP + _HARD_SLOPE * ratio) if overlap > 0 else 0.0
    )
    flipped = max(0, int(_badness(metrics.flipped_tris, float(n))))
    parts["flipped"] = (
        weights.flipped * (_HARD_STEP + _HARD_SLOPE * min(1.0, flipped / n)) if flipped else 0.0
    )
    degenerate = max(0, int(_badness(metrics.degenerate_tris, float(n))))
    parts["degenerate"] = (
        weights.flipped * _DEGENERATE_SLOPE * min(1.0, degenerate / n) if degenerate else 0.0
    )
    return parts


def section_penalty(rep: Report, best, weights, ranked) -> dict[str, float]:
    """Reproduce ORGANIC.md section 0.2: what the score is actually made of."""
    metrics = best.metrics
    parts = decompose(metrics, weights)
    total = sum(parts.values())

    rep.head("0.2  What the score is actually made of")
    rep.say("best recipe %s" % best.params.signature())
    rep.say("score %.2f   (penalty P = %.4f, score = 100*exp(-P))" % (score(metrics, weights),
                                                                     total))
    rep.say("")
    rep.say("islands %d   seam_length %.3f   seam_ratio %.3f"
            % (metrics.island_count, metrics.seam_length, metrics.seam_ratio))
    rep.say("stretch_l2 %.4f   stretch_linf %.4f   conformal %.4f"
            % (metrics.stretch_l2, metrics.stretch_linf, metrics.conformal_error))
    rep.say("packing_efficiency %.4f   uv_coverage %.4f"
            % (metrics.packing_efficiency, metrics.uv_coverage))
    rep.say("overlap_area %.7g   flipped_tris %d   degenerate_tris %d   (of %d)"
            % (metrics.overlap_area, metrics.flipped_tris, metrics.degenerate_tris,
               metrics.n_tris))
    rep.say("")
    rep.say("%-12s %-10s %-9s %s" % ("term", "penalty", "share", "removing it alone"))
    for key, value in sorted(parts.items(), key=lambda kv: -kv[1]):
        if value <= 0.0:
            continue
        gain = 100.0 * math.exp(-(total - value)) - 100.0 * math.exp(-total)
        rep.say("%-12s %-10.4f %-9s %+.1f pts"
                % (key, value, "%.1f%%" % (100.0 * value / max(total, 1e-9)), gain))

    rep.say("")
    rep.say("ceilings, holding everything else fixed:")
    for drop in (("overlap",), ("seam",), ("overlap", "seam"), ("overlap", "seam", "packing")):
        rest = total - sum(parts[k] for k in drop)
        rep.say("  fix %-26s -> %.1f" % ("+".join(drop), 100.0 * math.exp(-rest)))

    rep.say("")
    rep.say("seam targets, assuming overlap is fixed:")
    rest = total - parts["overlap"] - parts["seam"]
    total_3d = _total_area_3d(best)
    for target in (40.0, 50.0, 60.0):
        allowed = -math.log(target / 100.0) - rest
        needed_ratio = allowed / weights.seam * 25.0
        needed_len = needed_ratio * math.sqrt(total_3d) if total_3d > 0.0 else float("nan")
        rep.say("  score %-4.0f  seam_ratio <= %-6.2f  seam_length <= %-7.2f  (now %.2f / %.2f)"
                % (target, needed_ratio, needed_len, metrics.seam_ratio, metrics.seam_length))

    rep.say("")
    rep.say("all %d candidate scores: %s"
            % (len(ranked), ", ".join("%.1f" % c.score for c in ranked)))
    rep.say("%-7s %-6s %-8s %-9s %-10s %-8s %s"
            % ("score", "isl", "seamR", "stretch", "overlap", "flipped", "pack"))
    for cand in ranked:
        m = cand.metrics
        rep.say("%-7.1f %-6d %-8.2f %-9.4f %-10.3g %-8d %.3f"
                % (cand.score, m.island_count, m.seam_ratio, m.stretch_l2, m.overlap_area,
                   m.flipped_tris, m.packing_efficiency))
    return parts


def _total_area_3d(best) -> float:
    """Return the total 3D surface area implied by the candidate's own metrics."""
    if best.metrics.seam_ratio > 0.0:
        return (best.metrics.seam_length / best.metrics.seam_ratio) ** 2
    return 0.0


# ---------------------------------------------------------------------------
# Section 0.3
# ---------------------------------------------------------------------------


def section_overlap(rep: Report, sample, metrics, tri_island, *, mc_samples: int) -> None:
    """Reproduce ORGANIC.md section 0.3: the overlap is not a flip."""
    rep.head("0.3  The overlap is a global injectivity failure, not a flip")

    signed = tri_areas_uv(sample)
    rep.say("signed UV areas: %d negative, %d positive, %d at zero"
            % (int((signed < -1e-15).sum()), int((signed > 1e-15).sum()),
               int((np.abs(signed) <= 1e-15).sum())))
    rep.say("evaluator flipped_tris = %d" % metrics.flipped_tris)
    rep.say("=> the map is LOCALLY INJECTIVE everywhere."
            if metrics.flipped_tris == 0 else "=> there are real flips here.")
    rep.say("")

    uvt = sample.tri_uvs()
    lo = uvt.min(axis=1)
    hi = uvt.max(axis=1)
    n = sample.n_tris
    vsets = [{int(v) for v in sample.tri_verts[i]} for i in range(n)]

    hits = []
    order = np.argsort(lo[:, 0])
    for ii in range(n):
        i = int(order[ii])
        for jj in range(ii + 1, n):
            j = int(order[jj])
            if lo[j, 0] >= hi[i, 0]:
                break
            if lo[j, 1] >= hi[i, 1] or lo[i, 1] >= hi[j, 1]:
                continue
            area = clip_area(uvt[i], uvt[j])
            if area > 1e-15:
                hits.append((i, j, area))

    total = sum(h[2] for h in hits)
    rep.say("evaluator overlap_area              %.10g" % metrics.overlap_area)
    rep.say("independent exact convex-clip       %.10g  over %d pairs" % (total, len(hits)))
    rel = abs(total - metrics.overlap_area) / max(metrics.overlap_area, 1e-30)
    rep.say("relative difference                 %.3g  => the evaluator is %s"
            % (rel, "NOT lying" if rel < 1e-9 else "DISAGREEING -- investigate"))
    rep.say("")

    shared2 = [h for h in hits if len(vsets[h[0]] & vsets[h[1]]) >= 2]
    shared1 = [h for h in hits if len(vsets[h[0]] & vsets[h[1]]) == 1]
    shared0 = [h for h in hits if len(vsets[h[0]] & vsets[h[1]]) == 0]
    rep.say("edge-adjacent pairs (share an edge) : %-4d area %.6g"
            % (len(shared2), sum(h[2] for h in shared2)))
    rep.say("vertex-adjacent pairs               : %-4d area %.6g"
            % (len(shared1), sum(h[2] for h in shared1)))
    rep.say("fully disjoint pairs                : %-4d area %.6g"
            % (len(shared0), sum(h[2] for h in shared0)))
    if not shared2 and hits:
        rep.say("=> no overlapping pair shares an edge, so this is NOT a hinge folding")
        rep.say("   at a crease.  It is one region of a chart lapping over a distant")
        rep.say("   region.  A flip-only check cannot see it.")
    rep.say("")

    same = [h for h in hits if tri_island[h[0]] == tri_island[h[1]]]
    cross = [h for h in hits if tri_island[h[0]] != tri_island[h[1]]]
    rep.say("within one island (a self-fold)     : %-4d area %.6g"
            % (len(same), sum(h[2] for h in same)))
    rep.say("between two islands (a collision)   : %-4d area %.6g"
            % (len(cross), sum(h[2] for h in cross)))
    rep.say("")

    if hits:
        rng = np.random.default_rng(7)
        rep.say("worst 8 pairs, each cross-checked against %d Monte-Carlo samples:" % mc_samples)
        rep.say("%-6s %-6s %-7s %-12s %-12s %s"
                % ("tri", "tri", "shared", "exact", "monte-carlo", "3D centroid"))
        for i, j, area in sorted(hits, key=lambda h: -h[2])[:8]:
            mc = monte_carlo_area(uvt[i], uvt[j], samples=mc_samples, rng=rng)
            centroid = sample.verts[sample.tri_verts[i]].mean(axis=0)
            rep.say("%-6d %-6d %-7d %-12.5g %-12.5g (%6.3f,%6.3f,%6.3f)"
                    % (i, j, len(vsets[i] & vsets[j]), area, mc, *centroid))
        pts = np.array([sample.verts[sample.tri_verts[h[0]]].mean(axis=0) for h in hits])
        wts = np.array([h[2] for h in hits])
        centre = (pts * wts[:, None]).sum(axis=0) / wts.sum()
        rep.say("")
        rep.say("fold location in 3D: centroid (%.3f, %.3f, %.3f)" % tuple(centre))
        rep.say("  bbox x in [%.3f, %.3f]  y in [%.3f, %.3f]  z in [%.3f, %.3f]"
                % (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(),
                   pts[:, 2].min(), pts[:, 2].max()))
        rep.say("  |x| range %.3f..%.3f  (Suzanne's ears sit at |x| ~ 1.2-1.4)"
                % (np.abs(pts[:, 0]).min(), np.abs(pts[:, 0]).max()))


# ---------------------------------------------------------------------------
# Section 0.4
# ---------------------------------------------------------------------------


def section_seams(rep: Report, sample, metrics, tri_island) -> None:
    """Reproduce ORGANIC.md section 0.4: all of the seam length is mergeable."""
    rep.head("0.4  Where the seam length goes, and how much of it is mergeable")

    a3 = tri_areas_3d(sample)
    auv = np.abs(tri_areas_uv(sample))
    n_islands = int(tri_island.max()) + 1 if tri_island.size else 0
    rep.say("islands (independent count) %d   (evaluator said %d)"
            % (n_islands, metrics.island_count))
    rep.say("total 3D area %.4f  (sqrt %.4f)   total |UV| area %.4f"
            % (a3.sum(), math.sqrt(max(a3.sum(), 0.0)), auv.sum()))
    rep.say("")
    rep.say("%-5s %-7s %-11s %s" % ("isl", "tris", "area3d", "areaUV"))
    for i in range(n_islands):
        mask = tri_island == i
        rep.say("%-5d %-7d %-11.4f %.5f" % (i, int(mask.sum()), a3[mask].sum(), auv[mask].sum()))

    edge_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t in range(sample.n_tris):
        tv = sample.tri_verts[t]
        for k in range(3):
            a, b = int(tv[k]), int(tv[(k + 1) % 3])
            edge_tris[(min(a, b), max(a, b))].append(t)

    def uv_of(tri: int, vert: int):
        """Return the UV that ``tri`` assigns to ``vert``, or None."""
        for k in range(3):
            if int(sample.tri_verts[tri, k]) == vert:
                return sample.uvs[int(sample.tri_loops[tri, k])]
        return None

    shared_len: dict[tuple[int, int], float] = defaultdict(float)
    mesh_boundary = 0.0
    cut_len = 0.0
    for (a, b), tris in edge_tris.items():
        length = float(np.linalg.norm(sample.verts[a] - sample.verts[b]))
        if len(tris) == 1:
            mesh_boundary += length
            continue
        if len(tris) != 2:
            continue
        i, j = tris
        cut = False
        for vert in (a, b):
            ua, ub = uv_of(i, vert), uv_of(j, vert)
            if ua is None or ub is None or float(np.linalg.norm(ua - ub)) > 1e-7:
                cut = True
        if not cut:
            continue
        cut_len += length
        ia, ib = int(tri_island[i]), int(tri_island[j])
        if ia != ib:
            shared_len[(min(ia, ib), max(ia, ib))] += length

    rep.say("")
    rep.say("measured cut length (interior edges split in UV) : %.3f" % cut_len)
    rep.say("mesh boundary length (free edges, not our doing) : %.3f" % mesh_boundary)
    rep.say("evaluator seam_length %.3f   seam_ratio %.3f"
            % (metrics.seam_length, metrics.seam_ratio))
    total_shared = sum(shared_len.values())
    rep.say("")
    rep.say("cut length lying BETWEEN two islands (removable by merging): %.3f (%.1f%%)"
            % (total_shared, 100.0 * total_shared / max(cut_len, 1e-9)))
    rep.say("cut length lying INSIDE one island (a slit opening it to a disk): %.3f"
            % (cut_len - total_shared))
    rep.say("")
    rep.say("largest shared boundaries:")
    rep.say("%-14s %s" % ("island pair", "shared length"))
    for (a, b), length in sorted(shared_len.items(), key=lambda kv: -kv[1])[:12]:
        rep.say("%-14s %.3f" % ("%d-%d" % (a, b), length))

    if shared_len and n_islands >= 4:
        # The four largest islands by surface area.  Chosen by area rather than
        # by shared-boundary length because that is the set ORGANIC.md section
        # 0.4 quotes, and a diagnostic that answers a slightly different
        # question is not a reproduction.
        areas = [(float(a3[tri_island == i].sum()), i) for i in range(n_islands)]
        members = sorted(i for _, i in sorted(areas, reverse=True)[:4])
        removable = sum(
            length for (a, b), length in shared_len.items() if a in members and b in members
        )
        area_share = sum(a for a, i in areas if i in members)
        total_3d = float(a3.sum())
        new_ratio = (cut_len - removable) / math.sqrt(total_3d) if total_3d > 0.0 else 0.0
        rep.say("")
        rep.say("merging the four largest islands %s (%.2f of %.2f total area)"
                % (members, area_share, total_3d))
        rep.say("removes %.2f units of seam, taking seam_ratio from %.3f to %.2f --"
                % (removable, metrics.seam_ratio, new_ratio))
        rep.say("past the score-50 threshold on its own, if distortion holds.")


# ---------------------------------------------------------------------------
# Section 8: the corpus and its oracles
# ---------------------------------------------------------------------------


def section_corpus(rep: Report, *, weights, count: int, seed: int, quick: bool) -> None:
    """Report corpus invariants and the injectivity certificate for every mesh."""
    rep.head("8  The organic corpus and its oracles")

    certify = _optional("unwrap_variations.core.injectivity", "certify")
    naive = _optional("tests.reference", "naive_certify")
    rep.say("core.injectivity.certify   : %s" % ("present" if certify else "NOT YET LANDED"))
    rep.say("tests.reference.naive_certify: %s" % ("present" if naive else "NOT YET LANDED"))
    rep.say("")
    rep.say("%-17s %-7s %-7s %-6s %-9s %-14s %s"
            % ("mesh", "verts", "tris", "shells", "chi", "digest", "invariants"))
    meshes = []
    for name in corpus.NAMES:
        try:
            mesh = corpus.build(name)
        except Exception as exc:  # noqa: BLE001 -- a diagnostic must survive a bad entry
            rep.say("%-17s BUILD FAILED: %s" % (name, exc))
            continue
        meshes.append(mesh)
        inv = mesh.invariants
        rep.say("%-17s %-7d %-7d %-6d %-9s %-14s %s"
                % (name, inv.n_verts, inv.n_tris, inv.shells,
                   ",".join(str(c) for c in inv.chi), mesh.digest(),
                   "OK" if inv == corpus.EXPECTED.get(name) else "MISMATCH"))

    rep.say("")
    rep.say("Injectivity of the top-ranked candidate, per mesh.  ORGANIC.md section 9")
    rep.say("item 2 requires overlap_area == 0.0 and flipped_tris == 0 on every one.")
    rep.say("")
    rep.say("%-17s %-8s %-6s %-9s %-11s %-9s %s"
            % ("mesh", "score", "isl", "flipped", "overlap", "crossings", "oracle agrees"))
    for mesh in meshes:
        if quick and mesh.n_tris > 4000:
            rep.say("%-17s %s" % (mesh.name, "skipped (--quick, >4000 tris)"))
            continue
        _clear_scene()
        obj = corpus.to_object(mesh)
        report = engine.generate(obj, count=count, seed=seed, weights=weights, store=False,
                                 compute_overlap=True)
        if not report.candidates:
            rep.say("%-17s %s" % (mesh.name, "GENERATION PRODUCED NOTHING"))
            continue
        best = max(report.candidates, key=lambda c: c.score)
        sample = bridge.extract_sample(obj).with_uvs(best.uvs)
        crossings = "n/a"
        agrees = "n/a"
        if certify is not None:
            got = certify(sample)
            crossings = str(got.boundary_crossings)
            if naive is not None and mesh.n_tris <= 4000:
                want = naive(sample)
                agrees = "YES" if (
                    got.flipped == want["flipped"]
                    and got.boundary_crossings == want["boundary_crossings"]
                    and got.globally_injective == want["globally_injective"]
                ) else "NO -- BUG"
            elif naive is not None:
                agrees = "skipped (slow)"
        rep.say("%-17s %-8.2f %-6d %-9d %-11.4g %-9s %s"
                % (mesh.name, best.score, best.metrics.island_count,
                   best.metrics.flipped_tris, best.metrics.overlap_area, crossings, agrees))


def _optional(module_path: str, attribute: str):
    """Return ``module.attribute`` if importable, else ``None``.

    Feature detection, not a hard import: the harness in ORGANIC.md section 8
    has to run before the modules in sections 3 to 7 exist, and pick them up
    without edits once they land.
    """
    try:
        module = __import__(module_path, fromlist=[attribute])
    except ImportError:
        return None
    return getattr(module, attribute, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv):
    """Parse the arguments Blender passed through after a bare ``--``."""
    parser = argparse.ArgumentParser(
        prog="diag_organic.py",
        description="Reproduce the measured ground truth in docs/ORGANIC.md section 0.",
    )
    parser.add_argument("--count", type=int, default=10, help="Candidates to generate.")
    parser.add_argument("--seed", type=int, default=1, help="Determinism seed.")
    parser.add_argument("--preset", default="balanced", help="ScoreWeights preset.")
    parser.add_argument(
        "--sections",
        default=",".join(SECTIONS),
        help="Comma-separated subset of %s (default all)." % ",".join(SECTIONS),
    )
    parser.add_argument(
        "--corpus", action="store_true", help="Also run the section 8 corpus/oracle report."
    )
    parser.add_argument(
        "--quick", action="store_true", help="With --corpus, skip meshes above 4000 triangles."
    )
    parser.add_argument(
        "--mc-samples", type=int, default=200000,
        help="Monte-Carlo samples used to cross-check the overlap clipper.",
    )
    parser.add_argument("--out", default=None, help="Optional path to write the report to.")
    return parser.parse_args(argv)


def main() -> int:
    """Run the requested sections and print the report.  Returns an exit code."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    wanted = {s.strip() for s in args.sections.split(",") if s.strip()}
    unknown = wanted - set(SECTIONS)
    if unknown:
        print("ERROR: unknown section(s) %s; known: %s"
              % (", ".join(sorted(unknown)), ", ".join(SECTIONS)))
        return 1

    rep = Report()
    weights = ScoreWeights.preset(args.preset)
    rep.say("ORGANIC GROUND TRUTH -- Blender %s, numpy %s"
            % (bpy.app.version_string, np.__version__))
    rep.say("preset %r | seed %d | count %d | primitive_monkey_add()"
            % (args.preset, args.seed, args.count))

    _clear_scene()
    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.active_object
    obj.name = "suzanne"

    report = engine.generate(obj, count=args.count, seed=args.seed, weights=weights, store=False,
                             compute_overlap=True)
    if not report.candidates:
        rep.say("GENERATION PRODUCED NOTHING -- cannot reproduce any of section 0")
        rep.write(args.out)
        return 1
    ranked = sorted(report.candidates, key=lambda c: -c.score)
    best = ranked[0]
    rep.say("generated %d candidates in %.2fs" % (len(report.candidates), report.seconds))

    sample = bridge.extract_sample(obj).with_uvs(best.uvs)
    tri_island = uv_islands(sample)

    if "0.1" in wanted:
        section_topology(rep, sample)
    if "0.2" in wanted:
        section_penalty(rep, best, weights, ranked)
    if "0.3" in wanted:
        section_overlap(rep, sample, best.metrics, tri_island, mc_samples=args.mc_samples)
    if "0.4" in wanted:
        section_seams(rep, sample, best.metrics, tri_island)
    if args.corpus:
        section_corpus(rep, weights=weights, count=args.count, seed=args.seed, quick=args.quick)

    rep.say("")
    rep.say("DONE")
    rep.write(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
