"""A deliberately naive, loop-based reference implementation of the UV metrics.

This is the *oracle*.  It exists so the vectorised evaluator in
``unwrap_variations.core.evaluator`` can be cross-checked against something
obviously correct: plain Python loops, textbook formulae written out verbatim,
no clever numpy.  When the two disagree, this file is the one to trust.

The stretch math is Sander et al., *Texture Mapping Progressive Meshes*
(SIGGRAPH 2001), section 3.  Given a triangle with 3D corners ``q1, q2, q3`` and
texture corners ``p1 = (s1, t1) .. p3``::

    A  = ((s2-s1)(t3-t1) - (s3-s1)(t2-t1)) / 2

    Ss = (q1(t2-t3) + q2(t3-t1) + q3(t1-t2)) / (2A)
    St = (q1(s3-s2) + q2(s1-s3) + q3(s2-s1)) / (2A)

    a = Ss.Ss    b = Ss.St    c = St.St

    Gamma = sqrt(((a+c) + sqrt((a-c)^2 + 4b^2)) / 2)   # max singular value
    gamma = sqrt(((a+c) - sqrt((a-c)^2 + 4b^2)) / 2)   # min singular value

    L2(T)      = sqrt((a + c) / 2)
    Linf(T)    = Gamma
    L2(mesh)   = sqrt(sum(L2(T)^2 * A3d(T)) / sum(A3d(T)))
    Linf(mesh) = max Gamma(T)

**Normalisation.**  Texture coordinates are scaled so the total texture area
equals the total surface area, i.e. the singular values are divided by
``k = sqrt(sum A3d / sum Auv)``.  After that an isometric map scores exactly
1.0, and the metric is invariant to a global UV scale.

**The invariant.**  Mesh-wide normalised L2 stretch is *provably* ``>= 1.0``.
Per triangle AM-GM gives ``L2(T)^2 = (G^2+g^2)/2 >= G*g = A3d/Auv``; summing
area-weighted and applying Cauchy-Schwarz gives
``sum(A3d^2/Auv) >= (sum A3d)^2 / sum Auv = sum A3d``.  A value below 1.0 is
therefore always a bug, never a good unwrap.  The usual cause is computing the
normalisation constant ``k`` over all triangles while accumulating the sums over
only the non-degenerate subset: the two triangle sets must match exactly.

Nothing here may import :mod:`bpy`, and nothing here may import the module it is
supposed to be checking.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

__all__ = [
    "tri_area_3d",
    "tri_area_uv_signed",
    "stretch_singular_values",
    "tri_angles",
    "tri_overlap_area",
    "reference_metrics",
    "UnionFind",
    # -- the injectivity oracle (docs/ORGANIC.md sections 4.1 and 8.2) --------
    "naive_orient2d_sign",
    "naive_segments_cross",
    "naive_boundary_segments",
    "naive_crossing_pairs",
    "naive_boundary_crossings",
    "naive_weld_uv_vertices",
    "naive_chart_labels",
    "naive_boundary_loops",
    "naive_boundary_loops_by_chart",
    "naive_overlapping_triangle_pairs",
    "naive_certify",
]

#: Relative area floor below which a triangle is unusable for distortion maths.
#: Matches :func:`unwrap_variations.core.meshdata.degenerate_mask`: the epsilon
#: is scaled by the mesh's own mean triangle area, *per domain*, so degeneracy is
#: scale invariant.  An absolute floor would call every triangle of a 0.1 mm CAD
#: part degenerate -- their areas are legitimately around 5e-15 -- and score a
#: perfectly good unwrap of it at zero.
EPS_AREA = 1e-12


def area_floor(areas) -> float:
    """Return the degeneracy floor for one domain, from its own finite areas.

    Multiplies, and deliberately does not take a maximum against the bare
    epsilon: clamping the floor up to 1e-12 would reinstate the absolute test on
    exactly the sub-millimetre models the relative one exists to rescue.
    """
    finite = [a for a in areas if math.isfinite(a)]
    if not finite:
        return EPS_AREA
    scale = sum(finite) / len(finite)
    if not math.isfinite(scale) or scale <= 0.0:
        return EPS_AREA
    return EPS_AREA * scale


# ---------------------------------------------------------------------------
# Tiny vector helpers -- written out longhand on purpose.
# ---------------------------------------------------------------------------


def _sub(a, b):
    """Return ``a - b`` for two 3-tuples."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    """Return the dot product of two 3-tuples."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    """Return the cross product of two 3-tuples."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def tri_area_3d(q1, q2, q3):
    """Return the unsigned area of a 3D triangle."""
    c = _cross(_sub(q2, q1), _sub(q3, q1))
    return 0.5 * math.sqrt(_dot(c, c))


def tri_area_uv_signed(p1, p2, p3):
    """Return the **signed** area of a UV triangle; negative means flipped."""
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def stretch_singular_values(q, p, uv_area_floor: float = 1e-14):
    """Return ``(Gamma, gamma)`` for one triangle, or ``None`` when degenerate.

    Args:
        q: Three 3D corner positions.
        p: Three UV corner positions, in the same corner order.
        uv_area_floor: Absolute UV-area threshold below which the Jacobian
            inverse is not numerically meaningful.  Callers that classify
            degeneracy relatively (see :func:`area_floor`) pass their own floor
            so this guard cannot silently reclassify a triangle they already
            accepted.

    Returns:
        The larger and smaller singular value of the parametrisation Jacobian,
        **unnormalised**, or ``None`` if the UV triangle has no area.
    """
    (q1, q2, q3), (p1, p2, p3) = q, p
    s1, t1 = p1
    s2, t2 = p2
    s3, t3 = p3
    area = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2.0
    if abs(area) < uv_area_floor:
        return None
    inv = 1.0 / (2.0 * area)
    ss = tuple(
        inv * (q1[i] * (t2 - t3) + q2[i] * (t3 - t1) + q3[i] * (t1 - t2)) for i in range(3)
    )
    st = tuple(
        inv * (q1[i] * (s3 - s2) + q2[i] * (s1 - s3) + q3[i] * (s2 - s1)) for i in range(3)
    )
    a = _dot(ss, ss)
    b = _dot(ss, st)
    c = _dot(st, st)
    disc = math.sqrt(max((a - c) ** 2 + 4.0 * b * b, 0.0))
    big = math.sqrt(max((a + c + disc) / 2.0, 0.0))
    small = math.sqrt(max((a + c - disc) / 2.0, 0.0))
    return big, small


def tri_angles(a, b, c):
    """Return the three interior angles (radians) of a triangle of any dimension."""
    out = []
    pts = [a, b, c]
    for i in range(3):
        p0 = pts[i]
        p1 = pts[(i + 1) % 3]
        p2 = pts[(i + 2) % 3]
        u = [p1[k] - p0[k] for k in range(len(p0))]
        v = [p2[k] - p0[k] for k in range(len(p0))]
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(x * x for x in v))
        if nu < 1e-15 or nv < 1e-15:
            out.append(0.0)
            continue
        cs = sum(u[k] * v[k] for k in range(len(u))) / (nu * nv)
        out.append(math.acos(max(-1.0, min(1.0, cs))))
    return out


class UnionFind:
    """Textbook union-find with path halving, used for island counting."""

    def __init__(self, n: int) -> None:
        """Create ``n`` singleton sets."""
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Return the representative of ``x``'s set."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge the sets containing ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Exact polygon-clipping overlap.
# ---------------------------------------------------------------------------


def _poly_area(poly):
    """Return the absolute area of a simple polygon by the shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def _ccw(tri):
    """Return the triangle wound counter-clockwise."""
    if tri_area_uv_signed(*tri) < 0:
        return [tri[0], tri[2], tri[1]]
    return list(tri)


def _line_isect(p1, p2, a, b):
    """Return the intersection of segment ``p1-p2`` with the infinite line ``a-b``."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-18:
        return p1
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def tri_overlap_area(t1, t2):
    """Return the exact area of the intersection of two UV triangles.

    Sutherland-Hodgman clipping of ``t1`` against each half-plane of ``t2``.
    Measuring true intersection **area** rather than running a separating-axis
    boolean is what makes this trustworthy: triangles that merely share an edge
    or touch at a single vertex intersect in a set of measure zero and correctly
    yield ``0.0``, so no ad-hoc adjacency special-casing is needed.  A boolean
    SAT test reported 8773 false positives on a 2048-triangle flat grid whose
    true answer is zero.

    **Degenerate triangles are rejected up front, and must be.**  A collapsed
    clip triangle has zero-length edges, and a zero-length edge's half-plane
    test reduces to ``0 >= -1e-15`` -- vacuously true, so that edge clips
    nothing.  Collapse all three and the subject survives the clip untouched:
    ``tri_overlap_area`` used to return a *unit triangle's whole area* against a
    collapsed triangle five units away.  Nothing caught it because every
    ``reference_metrics`` call in the suite passed ``do_overlap=False``.  The
    guard is exact, not a tolerance: a triangle with zero area covers no area,
    so its intersection with anything has none either.
    """
    if tri_area_uv_signed(*t1) == 0.0 or tri_area_uv_signed(*t2) == 0.0:
        return 0.0
    subject = _ccw(t1)
    clip = _ccw(t2)
    for i in range(3):
        a = clip[i]
        b = clip[(i + 1) % 3]
        ex, ey = b[0] - a[0], b[1] - a[1]

        def inside(p, a=a, ex=ex, ey=ey):
            return ex * (p[1] - a[1]) - ey * (p[0] - a[0]) >= -1e-15

        out = []
        n = len(subject)
        if n == 0:
            return 0.0
        for j in range(n):
            cur = subject[j]
            prv = subject[(j - 1) % n]
            cin, pin = inside(cur), inside(prv)
            if cin:
                if not pin:
                    out.append(_line_isect(prv, cur, a, b))
                out.append(cur)
            elif pin:
                out.append(_line_isect(prv, cur, a, b))
        subject = out
    return _poly_area(subject)


# ---------------------------------------------------------------------------
# The reference metric set.
# ---------------------------------------------------------------------------


def reference_metrics(sample, *, do_overlap: bool = True, overlap_limit: int = 4000) -> dict:
    """Compute the naive reference metrics for a :class:`UVMeshSample`.

    Args:
        sample: Any object exposing ``verts``, ``tri_verts``, ``tri_loops``,
            ``uvs``, ``edge_verts`` and ``edge_seam`` as array-likes.
        do_overlap: Whether to run the O(T^2) exact overlap pass.
        overlap_limit: Triangle count above which overlap is skipped entirely
            (the naive pass is quadratic and would take minutes).

    Returns:
        A dict of scalar metrics.  ``overlap_pairs`` and ``overlap_area`` are
        ``None`` when the overlap pass was skipped.
    """
    verts = np.asarray(sample.verts, dtype=float)
    tv = np.asarray(sample.tri_verts)
    tl = np.asarray(sample.tri_loops)
    uvs = np.asarray(sample.uvs, dtype=float)
    ev = np.asarray(sample.edge_verts)
    es = np.asarray(sample.edge_seam)

    n_tris = len(tv)
    a3_list: list[float] = []
    auv_list: list[float] = []
    for i in range(n_tris):
        q = [tuple(verts[tv[i][k]]) for k in range(3)]
        p = [tuple(uvs[tl[i][k]]) for k in range(3)]
        a3_list.append(tri_area_3d(*q))
        auv_list.append(tri_area_uv_signed(*p))

    # The normalisation constant MUST be computed over exactly the same subset
    # of triangles that contributes to the sums below (see the module docstring).
    # The floor is relative to each domain's own mean area, over the triangles
    # with representable geometry -- see :func:`area_floor`.
    finite3 = [i for i in range(n_tris) if _finite3(verts, tv[i])]
    finite2 = [i for i in range(n_tris) if _finite2(uvs, tl[i])]
    floor_3d = area_floor([a3_list[i] for i in finite3])
    floor_uv = area_floor([abs(auv_list[i]) for i in finite2])

    good = [
        i
        for i in range(n_tris)
        if _finite3(verts, tv[i])
        and _finite2(uvs, tl[i])
        and a3_list[i] > floor_3d
        and abs(auv_list[i]) > floor_uv
    ]
    good_set = set(good)
    sum_a3 = sum(a3_list[i] for i in good)
    sum_auv = sum(abs(auv_list[i]) for i in good)
    k = math.sqrt(sum_a3 / sum_auv) if sum_auv > 1e-15 and sum_a3 > 1e-15 else 1.0

    l2_num = l2_den = 0.0
    linf = 0.0
    ang_num = ang_den = 0.0
    ang_max = 0.0
    area_num = area_den = 0.0
    conf_num = 0.0
    flipped = 0
    flipped_usable = 0
    degenerate = 0
    # docs/ARCHITECTURE.md pins `angle_distortion` as area-weighted but leaves
    # `area_distortion` and `conformal_error` as an unqualified "mean".  Both
    # readings are computed so a test can accept either without going loose.
    area_terms: list[float] = []
    conf_terms: list[float] = []

    for i in range(n_tris):
        q = [tuple(verts[tv[i][k]]) for k in range(3)]
        p = [tuple(uvs[tl[i][k]]) for k in range(3)]
        a3 = a3_list[i]
        auv = auv_list[i]
        # A flipped triangle is flipped whether or not it is usable for the
        # distortion means, so this count runs over every triangle.
        if auv < 0:
            flipped += 1
        if i not in good_set:
            degenerate += 1
            continue

        sv = stretch_singular_values(q, p, uv_area_floor=floor_uv)
        if sv is None:
            degenerate += 1
            continue
        flipped_usable += 1 if auv < 0 else 0
        big, small = sv[0] / k, sv[1] / k
        l2_num += ((big * big + small * small) / 2.0) * a3
        l2_den += a3
        linf = max(linf, big)
        conf = (big / small - 1.0) if small > 1e-12 else 0.0
        conf_num += conf * a3
        conf_terms.append(conf)

        ang3 = tri_angles(*q)
        anguv = tri_angles(*p)
        dev = sum(abs(math.degrees(ang3[j] - anguv[j])) for j in range(3)) / 3.0
        ang_num += dev * a3
        ang_den += a3
        ang_max = max(ang_max, max(abs(math.degrees(ang3[j] - anguv[j])) for j in range(3)))

        ratio = (abs(auv) / sum_auv) / (a3 / sum_a3)
        area_num += abs(math.log2(ratio)) * a3
        area_den += a3
        area_terms.append(abs(math.log2(ratio)))

    islands = _count_islands(tv, tl, uvs)

    seam_len = 0.0
    for e in range(len(ev)):
        if es[e]:
            a, b = verts[ev[e][0]], verts[ev[e][1]]
            seam_len += math.dist(a, b)

    overlap_pairs = None
    overlap_area = None
    if do_overlap and n_tris <= overlap_limit:
        overlap_pairs, overlap_area = _overlap_bruteforce(tv, tl, uvs)

    return {
        "tris": n_tris,
        "stretch_l2": math.sqrt(l2_num / l2_den) if l2_den else 0.0,
        "stretch_linf": linf,
        "angle_distortion": ang_num / ang_den if ang_den else 0.0,
        "angle_distortion_max": ang_max,
        "area_distortion": area_num / area_den if area_den else 0.0,
        "area_distortion_unweighted": (sum(area_terms) / len(area_terms)) if area_terms else 0.0,
        "conformal_error": conf_num / l2_den if l2_den else 0.0,
        "conformal_error_unweighted": (sum(conf_terms) / len(conf_terms)) if conf_terms else 0.0,
        "island_count": islands,
        "seam_length": seam_len,
        "flipped_tris": flipped,
        "flipped_usable_tris": flipped_usable,
        "degenerate_tris": degenerate,
        "uv_area_sum": sum_auv,
        "area_3d_sum": sum_a3,
        "overlap_pairs": overlap_pairs,
        "overlap_area": overlap_area,
    }


def _finite3(verts, idx) -> bool:
    """Return whether all three 3D corners are finite."""
    return all(math.isfinite(float(verts[idx[k]][j])) for k in range(3) for j in range(3))


def _finite2(uvs, idx) -> bool:
    """Return whether all three UV corners are finite."""
    return all(math.isfinite(float(uvs[idx[k]][j])) for k in range(3) for j in range(2))


def _count_islands(tv, tl, uvs) -> int:
    """Count connected UV islands by welding loops on ``(vertex, uv)``."""
    n_tris = len(tv)
    if n_tris == 0:
        return 0
    key_of: dict = {}
    keys: dict[int, int] = {}
    for i in range(n_tris):
        for c in range(3):
            li = int(tl[i][c])
            vi = int(tv[i][c])
            kk = (vi, round(float(uvs[li][0]), 9), round(float(uvs[li][1]), 9))
            if kk not in key_of:
                key_of[kk] = len(key_of)
            keys[li] = key_of[kk]
    uf = UnionFind(max(len(key_of), 1))
    for i in range(n_tris):
        ids = [keys[int(tl[i][c])] for c in range(3)]
        uf.union(ids[0], ids[1])
        uf.union(ids[1], ids[2])
    roots = {uf.find(keys[int(tl[i][0])]) for i in range(n_tris)}
    return len(roots)


def _overlap_bruteforce(tv, tl, uvs):
    """Return ``(pair_count, total_area)`` of UV self-overlap, the slow way."""
    n_tris = len(tv)
    tris = [[tuple(uvs[tl[i][c]]) for c in range(3)] for i in range(n_tris)]
    boxes = [
        (
            min(p[0] for p in t),
            min(p[1] for p in t),
            max(p[0] for p in t),
            max(p[1] for p in t),
        )
        for t in tris
    ]
    pairs = 0
    area = 0.0
    for i in range(n_tris):
        bi = boxes[i]
        if not all(math.isfinite(v) for v in bi):
            continue
        for j in range(i + 1, n_tris):
            bj = boxes[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            a = tri_overlap_area(tris[i], tris[j])
            if a > 1e-13:
                pairs += 1
                area += a
    return pairs, area


# ---------------------------------------------------------------------------
# The injectivity oracle.
# ---------------------------------------------------------------------------
#
# ``docs/ORGANIC.md`` section 8.2 makes this a deliverable, not scaffolding.
# ``unwrap_variations.core.injectivity`` is a spatial grid plus a filtered
# floating-point predicate, and the only way to trust it is to have a second
# implementation that shares no code, no data structure and no trick with it.
#
# Everything below is written to be *obviously correct* and slow: plain Python
# loops, dictionaries instead of sorted index arrays, and
# :class:`fractions.Fraction` for every orientation test, so the arithmetic is
# exact by construction rather than exact by an error-bound argument.  The
# determinant is even written in a different algebraic arrangement --
# ``(b-a) x (c-a)`` here against Shewchuk's ``(a-c) x (b-c)`` in the fast path --
# so a sign slip cannot be common to both.  When the two disagree, this file is
# the one to trust.
#
# Conventions, matching the module under test exactly (see its docstring):
#
# * segments crossing **transversally** count; shared endpoints, T-junctions and
#   collinear overlap do not;
# * loops weld on ``(mesh vertex, exact UV)``;
# * a boundary half-edge is a directed triangle edge whose opposite direction
#   does not occur in the same chart;
# * segments are numbered loop by loop, in order, each loop's closing segment
#   last -- so the two implementations' segment ids are directly comparable and
#   the tests can check *which* pairs cross, not merely how many.


def naive_orient2d_sign(a, b, c) -> int:
    """Return the exact sign of the orientation determinant of ``a, b, c``.

    ``+1`` when ``c`` lies to the left of the directed line ``a -> b``, ``-1``
    to the right, ``0`` when the three points are exactly collinear.

    Every float64 is a rational number, so :class:`fractions.Fraction` makes the
    determinant exact with no error analysis at all.  This is the slow, obvious
    counterpart to the filtered predicate in ``core/injectivity.py``.
    """
    ax, ay = Fraction(float(a[0])), Fraction(float(a[1]))
    bx, by = Fraction(float(b[0])), Fraction(float(b[1]))
    cx, cy = Fraction(float(c[0])), Fraction(float(c[1]))
    det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if det > 0:
        return 1
    if det < 0:
        return -1
    return 0


def naive_segments_cross(p1, q1, p2, q2) -> bool:
    """Return whether segment ``p1-q1`` transversally crosses segment ``p2-q2``.

    A crossing requires each segment to strictly separate the other's endpoints,
    so all four orientation signs must be non-zero.  A zero anywhere means the
    two merely touch -- a shared endpoint, a T-junction or a collinear overlap --
    and touching is not crossing.
    """
    d1 = naive_orient2d_sign(p1, q1, p2)
    d2 = naive_orient2d_sign(p1, q1, q2)
    d3 = naive_orient2d_sign(p2, q2, p1)
    d4 = naive_orient2d_sign(p2, q2, q1)
    if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
        return False
    return d1 != d2 and d3 != d4


def naive_boundary_segments(loops) -> list:
    """Return ``[(from, to), ...]`` for every segment of every closed polyline.

    Loop by loop, in order, with each loop's closing segment last.  Polylines
    shorter than two vertices contribute nothing.
    """
    segments = []
    for loop in loops:
        idx = [int(v) for v in loop]
        if len(idx) < 2:
            continue
        for k in range(len(idx)):
            segments.append((idx[k], idx[(k + 1) % len(idx)]))
    return segments


def naive_crossing_pairs(uv, loops, *, use_bbox: bool = True, limit: int = 6000) -> list:
    """Return every crossing pair of segment ids, the slow way.

    Args:
        uv: ``(N, 2)`` coordinates the loop indices refer to.
        loops: Closed polylines of indices into ``uv``.
        use_bbox: When True, pairs whose bounding boxes do not strictly overlap
            are skipped before the exact predicate.  That rejection is a
            theorem, not a heuristic -- a transversal crossing puts each
            segment's endpoints strictly either side of the other's supporting
            line, so both boxes must strictly overlap on both axes -- and the
            test suite pins it by running small cases both ways.
        limit: Refuse above this many segments.  The pass is quadratic in exact
            rational arithmetic and would take hours on a real atlas.

    Returns:
        A sorted list of ``(i, j)`` segment-id pairs with ``i < j``.
    """
    points = [(float(p[0]), float(p[1])) for p in np.asarray(uv, dtype=float).reshape(-1, 2)]
    segments = naive_boundary_segments(loops)
    n = len(segments)
    if n > limit:
        raise ValueError("naive_crossing_pairs refuses %d segments (limit %d)" % (n, limit))

    ends: list = []
    for a, b in segments:
        if a < 0 or b < 0 or a >= len(points) or b >= len(points):
            ends.append(None)
            continue
        pa, pb = points[a], points[b]
        finite = all(math.isfinite(v) for v in (pa[0], pa[1], pb[0], pb[1]))
        ends.append((pa, pb) if finite else None)

    out = []
    for i in range(n):
        if ends[i] is None:
            continue
        pi, qi = ends[i]
        ai, bi = segments[i]
        for j in range(i + 1, n):
            if ends[j] is None:
                continue
            pj, qj = ends[j]
            aj, bj = segments[j]
            # Segments sharing an endpoint touch, and touching is not crossing.
            if ai in (aj, bj) or bi in (aj, bj):
                continue
            if use_bbox and (
                max(pi[0], qi[0]) <= min(pj[0], qj[0])
                or max(pj[0], qj[0]) <= min(pi[0], qi[0])
                or max(pi[1], qi[1]) <= min(pj[1], qj[1])
                or max(pj[1], qj[1]) <= min(pi[1], qi[1])
            ):
                continue
            if naive_segments_cross(pi, qi, pj, qj):
                out.append((i, j))
    return out


def naive_boundary_crossings(uv, loops, **kwargs) -> int:
    """Return how many pairs of boundary segments cross, the slow way."""
    return len(naive_crossing_pairs(uv, loops, **kwargs))


def naive_weld_uv_vertices(sample) -> list:
    """Return one welded UV-vertex id per loop, keyed on ``(vertex, exact UV)``.

    A plain dictionary over exact keys.  Two loops share a UV vertex only when
    they sit on the same mesh vertex *and* carry a bit-identical UV, which is
    what a real unwrapper writes and what makes a genuine difference read as a
    genuine crack in the chart.
    """
    uvs = np.asarray(sample.uvs, dtype=float).reshape(-1, 2)
    loop_verts = np.asarray(sample.loop_verts).reshape(-1)
    ids: dict = {}
    node_of_loop = []
    for li in range(len(uvs)):
        vert = int(loop_verts[li]) if li < len(loop_verts) else 0
        key = (max(vert, 0), float(uvs[li][0]), float(uvs[li][1]))
        if key not in ids:
            ids[key] = len(ids)
        node_of_loop.append(ids[key])
    return node_of_loop


def _naive_valid_tris(sample) -> list:
    """Return the indices of triangles whose loop indices are all in range."""
    tl = np.asarray(sample.tri_loops).reshape(-1, 3)
    n_loops = len(np.asarray(sample.uvs, dtype=float).reshape(-1, 2))
    good = []
    for t in range(len(tl)):
        corner = [int(tl[t][k]) for k in range(3)]
        if all(0 <= c < n_loops for c in corner):
            good.append(t)
    return good


def naive_chart_labels(sample) -> list:
    """Return one chart label per **valid** triangle, by naive union-find.

    Charts are UV islands: triangles join when they share a welded UV vertex.

    Labels are numbered **in the order their first triangle appears**, which is
    the rule ``core/injectivity.chart_labels`` states.  Anything derived from
    internal node ids would not be comparable: this file numbers welded UV
    vertices in the order it happens to meet them and the module under test
    numbers them by a sorted key, so "the component's smallest node id" means
    two different things.  First-triangle order is a property of the partition
    itself, so both can reach it independently.
    """
    node_of_loop = naive_weld_uv_vertices(sample)
    tl = np.asarray(sample.tri_loops).reshape(-1, 3)
    valid = _naive_valid_tris(sample)
    if not valid or not node_of_loop:
        return []
    uf = UnionFind(max(node_of_loop) + 1)
    for t in valid:
        nodes = [node_of_loop[int(tl[t][k])] for k in range(3)]
        uf.union(nodes[0], nodes[1])
        uf.union(nodes[1], nodes[2])
    roots = [uf.find(node_of_loop[int(tl[t][0])]) for t in valid]
    return _naive_compact_by_first_appearance(roots)


def _naive_compact_by_first_appearance(raw) -> list:
    """Renumber arbitrary labels to ``0 .. k-1`` in order of first appearance."""
    rank: dict = {}
    for value in raw:
        if value not in rank:
            rank[value] = len(rank)
    return [rank[value] for value in raw]


def _naive_labels_in_force(sample, tri_chart) -> tuple:
    """Return ``(valid_tri_indices, label_per_valid_tri, label_per_tri)``.

    A caller-supplied labelling is renumbered by first appearance the same way a
    derived one is, so the chart keys of the two implementations line up.  A
    supplied labelling of the wrong length is ignored, exactly as the module
    under test ignores it.
    """
    tl = np.asarray(sample.tri_loops).reshape(-1, 3)
    valid = _naive_valid_tris(sample)
    if tri_chart is not None and len(list(tri_chart)) == len(tl):
        labels = _naive_compact_by_first_appearance([int(tri_chart[t]) for t in valid])
    else:
        labels = naive_chart_labels(sample)
    full = [0] * len(tl)
    for i, t in enumerate(valid):
        full[t] = labels[i]
    return valid, labels, full


def naive_boundary_loops_by_chart(sample, tri_chart=None) -> dict:
    """Return ``{chart: [closed polyline of loop indices, ...]}``, the slow way.

    Directed triangle edges are counted in a dictionary; a half-edge is on the
    boundary when its opposite direction never occurs in the same chart.  The
    survivors are chained head to tail.  Self-loop edges -- both endpoints
    welded to the same UV vertex, which only a collapsed triangle produces --
    are dropped, because their own reverse is themselves and they would
    otherwise be their own twin.

    Each cycle is rotated to start at its smallest loop index and the loops of a
    chart are ordered by that index, matching the canonical form the module
    under test promises.  A cycle has no intrinsic starting point, so the two
    walks pick different ones; without canonicalising, the segment numbering
    would differ and the crossing *pairs* could not be compared, only counted.
    """
    node_of_loop = naive_weld_uv_vertices(sample)
    tl = np.asarray(sample.tri_loops).reshape(-1, 3)
    valid, labels, _ = _naive_labels_in_force(sample, tri_chart)
    if not valid:
        return {}
    charts = dict(zip(valid, labels, strict=False))

    directed: dict = {}
    half_edges = []  # (chart, from_node, to_node, from_loop)
    for t in valid:
        chart = charts[t]
        nodes = [node_of_loop[int(tl[t][k])] for k in range(3)]
        for k in range(3):
            a, b = nodes[k], nodes[(k + 1) % 3]
            if a == b:
                continue
            directed[(chart, a, b)] = directed.get((chart, a, b), 0) + 1
            half_edges.append((chart, a, b, int(tl[t][k])))

    boundary = [h for h in half_edges if directed.get((h[0], h[2], h[1]), 0) == 0]

    outgoing: dict = {}
    for i, (chart, a, _b, _loop) in enumerate(boundary):
        outgoing.setdefault((chart, a), []).append(i)

    used = [False] * len(boundary)
    result: dict = {}
    for start in range(len(boundary)):
        if used[start]:
            continue
        chain = []
        here = start
        while not used[here]:
            used[here] = True
            chain.append(boundary[here][3])
            key = (boundary[here][0], boundary[here][2])
            nxt = -1
            for candidate in outgoing.get(key, []):
                if not used[candidate]:
                    nxt = candidate
                    break
            if nxt < 0:
                break
            here = nxt
        if len(chain) >= 3:
            pivot = chain.index(min(chain))
            result.setdefault(boundary[start][0], []).append(chain[pivot:] + chain[:pivot])
    for chart in result:
        result[chart].sort(key=lambda chain: chain[0])
    return result


def naive_boundary_loops(sample, tri_chart=None) -> list:
    """Return every chart's boundary polylines as one flat list."""
    grouped = naive_boundary_loops_by_chart(sample, tri_chart)
    out = []
    for chart in sorted(grouped):
        out.extend(grouped[chart])
    return out


def naive_overlapping_triangle_pairs(sample, *, limit: int = 4000) -> tuple:
    """Return ``(pairs, total_area)`` over **all** triangle pairs, no filtering.

    Unlike :func:`_overlap_bruteforce` this skips the bounding-box rejection
    entirely, so it is the crudest possible statement of "do any two UV
    triangles share area".  It exists to pin the bounding-box rejection itself.
    """
    tl = np.asarray(sample.tri_loops).reshape(-1, 3)
    uvs = np.asarray(sample.uvs, dtype=float).reshape(-1, 2)
    valid = _naive_valid_tris(sample)
    if len(valid) > limit:
        raise ValueError("naive_overlapping_triangle_pairs refuses %d triangles" % len(valid))
    tris = {t: [tuple(uvs[int(tl[t][c])]) for c in range(3)] for t in valid}
    pairs = []
    area = 0.0
    for i, ti in enumerate(valid):
        if not all(math.isfinite(v) for p in tris[ti] for v in p):
            continue
        for tj in valid[i + 1 :]:
            if not all(math.isfinite(v) for p in tris[tj] for v in p):
                continue
            overlap = tri_overlap_area(tris[ti], tris[tj])
            if overlap > 1e-13:
                pairs.append((ti, tj))
                area += overlap
    return pairs, area


def naive_certify(sample, tri_chart=None, *, do_overlap: bool = True) -> dict:
    """Return the naive counterpart of ``InjectivityReport``, as a dict.

    Every field is computed with loops and dictionaries straight from the
    definitions in ``docs/ORGANIC.md`` section 4.1, sharing nothing with the
    module under test.

    Args:
        sample: The mesh plus the UV layer to certify.
        tri_chart: Optional chart label per triangle; UV islands are derived
            when omitted.
        do_overlap: Whether to run the all-pairs triangle overlap pass.

    Returns:
        A dict carrying the report's fields plus ``chart_crossings`` and
        ``euler`` per chart, for localisation.
    """
    tl = np.asarray(sample.tri_loops).reshape(-1, 3)
    uvs = np.asarray(sample.uvs, dtype=float).reshape(-1, 2)
    node_of_loop = naive_weld_uv_vertices(sample)
    valid, labels, full = _naive_labels_in_force(sample, tri_chart)
    dropped = len(tl) - len(valid)
    if not valid:
        return {
            "flipped": 0,
            "degenerate": dropped,
            "boundary_crossings": 0,
            "overlap_area": 0.0,
            "locally_injective": dropped == 0,
            "globally_injective": dropped == 0,
            "charts_checked": 0,
            "non_disk_charts": 0,
            "chart_crossings": {},
            "euler": {},
        }
    charts = sorted(set(labels))

    signed: dict = {}
    finite: dict = {}
    for t in valid:
        p = [tuple(uvs[int(tl[t][c])]) for c in range(3)]
        finite[t] = all(math.isfinite(v) for corner in p for v in corner)
        signed[t] = tri_area_uv_signed(*p) if finite[t] else 0.0
    floor = area_floor([abs(signed[t]) for t in valid if finite[t]])

    degenerate = dropped
    usable = set()
    for t in valid:
        if not finite[t] or abs(signed[t]) <= floor:
            degenerate += 1
        else:
            usable.add(t)

    flipped = 0
    for chart in charts:
        mine = [t for i, t in enumerate(valid) if labels[i] == chart and t in usable]
        positive = sum(1 for t in mine if signed[t] > 0)
        negative = sum(1 for t in mine if signed[t] < 0)
        wanted = 1 if positive >= negative else -1
        flipped += sum(1 for t in mine if (1 if signed[t] > 0 else -1) != wanted)

    grouped = naive_boundary_loops_by_chart(sample, full)
    chart_crossings = {}
    total_crossings = 0
    for chart in charts:
        loops = grouped.get(chart, [])
        count = naive_boundary_crossings(uvs, loops) if loops else 0
        chart_crossings[chart] = count
        total_crossings += count

    euler = {}
    non_disk = 0
    for chart in charts:
        faces = 0
        nodes = set()
        edges = set()
        incident: dict = {}
        for i, t in enumerate(valid):
            if labels[i] != chart:
                continue
            faces += 1
            corner = [node_of_loop[int(tl[t][k])] for k in range(3)]
            nodes.update(corner)
            for k in range(3):
                a, b = corner[k], corner[(k + 1) % 3]
                if a != b:
                    key = (min(a, b), max(a, b))
                    edges.add(key)
                    incident[key] = incident.get(key, 0) + 1
        euler[chart] = len(nodes) - len(edges) + faces
        # A disk needs all four: chi == 1, one boundary loop, no edge with more
        # than two incident triangles, and no boundary loop through the same UV
        # vertex twice.  The last rules out a bowtie, which has chi == 1 and a
        # single boundary cycle and is still not a disk.
        loops_here = grouped.get(chart, [])
        pinched = any(
            len({node_of_loop[i] for i in chain}) != len(chain) for chain in loops_here
        )
        non_manifold = any(count > 2 for count in incident.values())
        if euler[chart] != 1 or len(loops_here) != 1 or pinched or non_manifold:
            non_disk += 1

    overlap_area = 0.0
    if do_overlap:
        _, overlap_area = naive_overlapping_triangle_pairs(sample)

    locally = flipped == 0 and degenerate == 0
    return {
        "flipped": flipped,
        "degenerate": degenerate,
        "boundary_crossings": total_crossings,
        "overlap_area": overlap_area,
        "locally_injective": locally,
        "globally_injective": (
            locally and total_crossings == 0 and non_disk == 0 and not overlap_area > 0.0
        ),
        "charts_checked": len(charts),
        "non_disk_charts": non_disk,
        "chart_crossings": chart_crossings,
        "euler": euler,
    }


# ===========================================================================
# Topology oracle -- docs/ORGANIC.md section 8.2
#
# Independent shell count and Euler characteristic from the triangle soup, and
# nothing else.  ORGANIC.md section 0.1 -- "Suzanne is three connected shells,
# not one, and the head is not a closed sphere" -- was produced by exactly this
# computation, and every downstream decision in that contract rests on it.  So
# it gets an oracle of its own: plain dicts and sets, one triangle at a time,
# no numpy cleverness and no shared code with the vectorised version in
# ``tools/organic_corpus.py`` that it exists to check.
#
# Kept in its own clearly delimited section, with names that cannot collide
# with the injectivity oracle, so this file can be extended from two
# directions at once without a merge conflict.  The exports are appended to
# ``__all__`` here rather than edited into the literal at the top of the file,
# for the same reason.
# ===========================================================================

__all__ += [
    "reference_shell_topology",
    "reference_euler_characteristic",
    "reference_union_rect_area",
    "reference_layout_metrics",
]



def reference_shell_topology(tri_verts, n_verts: int | None = None) -> list[dict]:
    """Return the per-shell Euler data of a triangle soup, the slow obvious way.

    Args:
        tri_verts: ``(T, 3)`` vertex indices, any array-like of ints.
        n_verts: Accepted for signature symmetry and deliberately unused.
            Unreferenced vertices belong to no shell and are excluded, exactly
            as they are by the vectorised implementation.

    Returns:
        One dict per connected component with keys ``faces``, ``edges``,
        ``verts``, ``chi``, ``boundary_edges`` and ``nonmanifold_edges``, sorted
        by descending face count then descending vertex count -- the same order
        ``tools.organic_corpus.mesh_topology`` produces, so the two are directly
        comparable element by element.

    Note:
        ``chi = V - E + F`` is reported raw and never converted to a genus.
        ``chi = 0`` is a torus *or* a genus-0 surface with two boundary loops,
        and Suzanne's head is the second of those; collapsing the two into one
        number is how a head gets mistaken for a doughnut.
    """
    tris = [tuple(int(v) for v in t) for t in tri_verts]
    _ = n_verts

    # Connected components by flood fill over a vertex adjacency dict.
    adjacency: dict[int, set[int]] = {}
    for a, b, c in tris:
        for x, y in ((a, b), (b, c), (c, a)):
            adjacency.setdefault(x, set()).add(y)
            adjacency.setdefault(y, set()).add(x)

    component: dict[int, int] = {}
    next_id = 0
    for start in sorted(adjacency):
        if start in component:
            continue
        stack = [start]
        component[start] = next_id
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if nxt not in component:
                    component[nxt] = next_id
                    stack.append(nxt)
        next_id += 1

    # Edge use counts, per undirected edge.
    edge_uses: dict[tuple[int, int], int] = {}
    for a, b, c in tris:
        for x, y in ((a, b), (b, c), (c, a)):
            key = (x, y) if x < y else (y, x)
            edge_uses[key] = edge_uses.get(key, 0) + 1

    shells: dict[int, dict] = {}
    for a, b, c in tris:
        shell = shells.setdefault(
            component[a], {"faces": 0, "edges": set(), "verts": set()}
        )
        shell["faces"] += 1
        for x, y in ((a, b), (b, c), (c, a)):
            shell["edges"].add((x, y) if x < y else (y, x))
            shell["verts"].add(x)
            shell["verts"].add(y)

    out = []
    for shell in shells.values():
        edges = shell["edges"]
        out.append(
            {
                "faces": shell["faces"],
                "edges": len(edges),
                "verts": len(shell["verts"]),
                "chi": len(shell["verts"]) - len(edges) + shell["faces"],
                "boundary_edges": sum(1 for e in edges if edge_uses[e] == 1),
                "nonmanifold_edges": sum(1 for e in edges if edge_uses[e] >= 3),
            }
        )
    out.sort(key=lambda s: (-s["faces"], -s["verts"]))
    return out


def reference_euler_characteristic(tri_verts) -> int:
    """Return ``V - E + F`` summed over every shell of the triangle soup."""
    return sum(s["chi"] for s in reference_shell_topology(tri_verts))


# ===========================================================================
# Metric oracle, layout half -- docs/ORGANIC.md section 8.2
#
# :func:`reference_metrics` above covers distortion, islands, seams and
# overlap.  The organic work leans on the *layout* fields too -- packing
# efficiency is 5.6% of Suzanne's penalty and UV coverage sets the denominator
# of the overlap ratio that is another 41.7% (ORGANIC.md section 0.2) -- and
# those had no oracle at all.  These are them.
# ===========================================================================


def reference_union_rect_area(rects) -> float:
    """Return the exact area of the union of axis-aligned rectangles.

    Coordinate-compression sweep: cut the plane into vertical slabs at every
    distinct x, and inside each slab union the y-intervals of the rectangles
    that span it.  Quadratic and obviously correct, which is the entire
    specification for an oracle.  The evaluator computes the same quantity by
    rasterising onto a bounded lattice, so the two may disagree by about one
    cell -- a real difference worth being able to see rather than one to paper
    over with a loose tolerance.

    Args:
        rects: Iterable of ``(x0, y0, x1, y1)`` with ``x0 <= x1``, ``y0 <= y1``.

    Returns:
        The union area.  Zero-width and zero-height rectangles contribute
        nothing, as they should: a single-triangle island has a real bounding
        box, but a degenerate one covers no area.
    """
    boxes = [tuple(float(v) for v in r) for r in rects]
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 0.0
    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    total = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        width = x1 - x0
        if width <= 0.0:
            continue
        spans = sorted((b[1], b[3]) for b in boxes if b[0] <= x0 and b[2] >= x1)
        if not spans:
            continue
        covered = 0.0
        cur_lo, cur_hi = spans[0]
        for lo, hi in spans[1:]:
            if lo > cur_hi:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        covered += cur_hi - cur_lo
        total += width * covered
    return total


def reference_layout_metrics(sample) -> dict:
    """Compute the naive reference values of the *layout* metrics.

    Companion to :func:`reference_metrics`, covering the fields that one does
    not: packing efficiency, UV coverage, seam ratio, texel-density spread, UV
    bounds and the out-of-bounds count, plus per-island areas so chart work has
    something independent to check against.  Everything is a Python loop over a
    Python list; nothing here is trying to be fast.

    Args:
        sample: Any object exposing ``verts``, ``tri_verts``, ``tri_loops``,
            ``uvs``, ``edge_verts`` and ``edge_seam`` as array-likes.

    Returns:
        A dict of scalar metrics plus ``island_areas_uv`` and
        ``island_areas_3d``, both keyed by island id.
    """
    verts = np.asarray(sample.verts, dtype=float)
    tv = np.asarray(sample.tri_verts)
    tl = np.asarray(sample.tri_loops)
    uvs = np.asarray(sample.uvs, dtype=float)
    n_tris = len(tv)

    a3: list[float] = []
    auv: list[float] = []
    corners: list[list[tuple[float, float]]] = []
    for i in range(n_tris):
        q = [tuple(verts[tv[i][k]]) for k in range(3)]
        p = [tuple(uvs[tl[i][k]]) for k in range(3)]
        corners.append(p)
        a3.append(tri_area_3d(*q))
        auv.append(abs(tri_area_uv_signed(*p)))

    finite = [i for i in range(n_tris) if _finite3(verts, tv[i]) and _finite2(uvs, tl[i])]
    floor_3d = area_floor([a3[i] for i in finite])
    floor_uv = area_floor([auv[i] for i in finite])
    good = [i for i in finite if a3[i] > floor_3d and auv[i] > floor_uv]

    sum_a3 = sum(a3[i] for i in good)
    sum_auv = sum(auv[i] for i in good)

    labels = _reference_island_labels(tv, tl, uvs)
    island_uv: dict[int, float] = {}
    island_3d: dict[int, float] = {}
    island_box: dict[int, list[float]] = {}
    for i in range(n_tris):
        lab = labels[i]
        island_uv[lab] = island_uv.get(lab, 0.0) + auv[i]
        island_3d[lab] = island_3d.get(lab, 0.0) + a3[i]
        us = [c[0] for c in corners[i]]
        vs = [c[1] for c in corners[i]]
        if not all(math.isfinite(x) for x in us + vs):
            continue
        box = island_box.get(lab)
        if box is None:
            island_box[lab] = [min(us), min(vs), max(us), max(vs)]
        else:
            box[0] = min(box[0], *us)
            box[1] = min(box[1], *vs)
            box[2] = max(box[2], *us)
            box[3] = max(box[3], *vs)

    union = reference_union_rect_area(island_box.values())
    # Clamped at 1.0 for the same reason the evaluator clamps it: anything above
    # 1.0 is overlap wearing packing efficiency as a disguise.
    efficiency = min(1.0, sum_auv / union) if union > 0.0 and sum_auv > 0.0 else 0.0

    density = [math.sqrt(auv[i] / a3[i]) for i in good]
    weights = [a3[i] for i in good]
    wsum = sum(weights)
    if wsum > 0.0 and density:
        mean = sum(d * w for d, w in zip(density, weights, strict=True)) / wsum
        var = sum((d - mean) ** 2 * w for d, w in zip(density, weights, strict=True)) / wsum
        texel_cv = math.sqrt(max(var, 0.0)) / mean if mean > 0.0 else 0.0
    else:
        texel_cv = 0.0

    seam_len = 0.0
    ev = np.asarray(sample.edge_verts)
    es = np.asarray(sample.edge_seam)
    for e in range(len(ev)):
        if es[e]:
            seam_len += math.dist(verts[ev[e][0]], verts[ev[e][1]])
    seam_ratio = seam_len / math.sqrt(sum_a3) if sum_a3 > 0.0 else 0.0

    flat = [c for tri in corners for c in tri if math.isfinite(c[0]) and math.isfinite(c[1])]
    if flat:
        bounds = (
            min(c[0] for c in flat),
            min(c[1] for c in flat),
            max(c[0] for c in flat),
            max(c[1] for c in flat),
        )
    else:
        bounds = (0.0, 0.0, 0.0, 0.0)
    out_of_bounds = sum(
        1
        for tri in corners
        if any(not (0.0 <= c[0] <= 1.0 and 0.0 <= c[1] <= 1.0) for c in tri)
    )

    return {
        "island_count": len(island_uv),
        "packing_efficiency": efficiency,
        "packing_union_area": union,
        "uv_coverage": sum_auv,
        "seam_length": seam_len,
        "seam_ratio": seam_ratio,
        "texel_density_cv": texel_cv,
        "uv_bounds": bounds,
        "out_of_bounds_tris": out_of_bounds,
        "island_areas_uv": island_uv,
        "island_areas_3d": island_3d,
    }


def _reference_island_labels(tv, tl, uvs) -> list[int]:
    """Return one dense island id per triangle, welding loops on ``(vertex, uv)``."""
    n_tris = len(tv)
    if n_tris == 0:
        return []
    key_of: dict = {}
    keys: dict[int, int] = {}
    for i in range(n_tris):
        for c in range(3):
            li = int(tl[i][c])
            vi = int(tv[i][c])
            kk = (vi, round(float(uvs[li][0]), 9), round(float(uvs[li][1]), 9))
            if kk not in key_of:
                key_of[kk] = len(key_of)
            keys[li] = key_of[kk]
    uf = UnionFind(max(len(key_of), 1))
    for i in range(n_tris):
        ids = [keys[int(tl[i][c])] for c in range(3)]
        uf.union(ids[0], ids[1])
        uf.union(ids[1], ids[2])
    dense: dict[int, int] = {}
    out = []
    for i in range(n_tris):
        root = uf.find(keys[int(tl[i][0])])
        if root not in dense:
            dense[root] = len(dense)
        out.append(dense[root])
    return out
