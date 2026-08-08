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

import numpy as np

__all__ = [
    "tri_area_3d",
    "tri_area_uv_signed",
    "stretch_singular_values",
    "tri_angles",
    "tri_overlap_area",
    "reference_metrics",
    "UnionFind",
]

#: Areas at or below this are treated as degenerate.  Matches the epsilon used
#: by :func:`unwrap_variations.core.meshdata.degenerate_mask`.
EPS_AREA = 1e-12


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


def stretch_singular_values(q, p):
    """Return ``(Gamma, gamma)`` for one triangle, or ``None`` when degenerate.

    Args:
        q: Three 3D corner positions.
        p: Three UV corner positions, in the same corner order.

    Returns:
        The larger and smaller singular value of the parametrisation Jacobian,
        **unnormalised**, or ``None`` if the UV triangle has no area.
    """
    (q1, q2, q3), (p1, p2, p3) = q, p
    s1, t1 = p1
    s2, t2 = p2
    s3, t3 = p3
    area = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2.0
    if abs(area) < 1e-14:
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
    """
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
    good = [
        i
        for i in range(n_tris)
        if _finite3(verts, tv[i])
        and _finite2(uvs, tl[i])
        and a3_list[i] > EPS_AREA
        and abs(auv_list[i]) > EPS_AREA
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

        sv = stretch_singular_values(q, p)
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
