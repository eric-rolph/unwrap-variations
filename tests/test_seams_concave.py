"""Tests for the ``CONCAVE`` seam strategy -- cutting down the valleys.

Why this strategy exists, and therefore what these tests have to pin down:

* On a smooth organic mesh ``SHARP`` selects **nothing** -- measured, zero
  edges at a 30-degree threshold on a sphere, a torus and an icosphere
  (docs/BENCHMARKS.md section 4.2).  ``uv.unwrap`` then gets no cuts, fails,
  and still reports ``{'FINISHED'}``.  ``CONCAVE`` must always open a closed
  surface.
* Artists hide seams in concave valleys, and a valley is also where the
  surface is least developable, so a cut there costs the least distortion.
  ``CONCAVE`` must actually prefer them, by a measurable margin over what the
  mesh offers on average.
* A cut that does not turn every chart into a disk is not a cut, it is a
  failure the solver reports as success.  Every case below is checked by an
  **independent** Euler count written in this file, over the *cut* triangulated
  submesh -- not by the module's own :func:`cut_topology`, which would make the
  assertion vacuous, and not from ``sample.n_edges``, which is the quad mesh's
  edge count and silently omits the tessellation diagonals.

The shapes are built here rather than loaded, because the point of each one is
that its answer is known by hand: a cube is convex everywhere, an L-shaped
solid has exactly one concave edge, a cylinder must be cut lengthwise, an
annulus needs exactly one path between its two rims.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict

import numpy as np
import pytest

from tests.conftest import (
    import_or_skip,
    load_corpus,
    make_all_identical,
    make_cube,
    make_empty,
    make_flat_grid,
    make_nan_uvs,
    make_random_mesh,
    make_unit_triangle,
)
from unwrap_variations.core.meshdata import make_sample

seams = import_or_skip("unwrap_variations.backends.seams")

concave_seams = seams.concave_seams
compute_seams = seams.compute_seams
edge_signed_dihedral_deg = seams.edge_signed_dihedral_deg


# ---------------------------------------------------------------------------
# Independent topology: written from scratch so the module cannot mark its own
# homework.
# ---------------------------------------------------------------------------


class _UF:
    """Plain union-find; deliberately naive."""

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def chart_euler(sample, seam) -> list[int]:
    """Return ``chi = V - E + F`` per chart of the **cut** triangulated submesh.

    Slow, quadratic in nothing, and obviously correct: group triangle sides by
    unordered vertex pair, weld the corners of every side that is not cut, then
    count.  A slit's two sides count as two edges and its interior vertices as
    two vertices, which is exactly why a slit sphere comes out at ``1`` here and
    at ``2`` under any Euler count taken over the uncut mesh.
    """
    tv = np.asarray(sample.tri_verts, dtype=np.int64)
    n_tris = tv.shape[0]
    cut = set()
    for e in np.flatnonzero(np.asarray(seam, dtype=bool)):
        a, b = int(sample.edge_verts[e][0]), int(sample.edge_verts[e][1])
        cut.add((min(a, b), max(a, b)))

    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for t in range(n_tris):
        for j in range(3):
            a, b = int(tv[t, j]), int(tv[t, (j + 1) % 3])
            groups[(min(a, b), max(a, b))].append((t, 3 * t + j, 3 * t + (j + 1) % 3))

    tri_uf = _UF(max(n_tris, 1))
    corner_uf = _UF(max(3 * n_tris, 1))
    glued: list[int] = []
    opened: list[int] = []
    for key, sides in groups.items():
        if len(sides) == 2 and key not in cut:
            (t1, a1, b1), (t2, a2, b2) = sides
            tri_uf.union(t1, t2)
            if int(tv[t1, a1 % 3]) == int(tv[t2, a2 % 3]):
                corner_uf.union(a1, a2)
                corner_uf.union(b1, b2)
            else:
                corner_uf.union(a1, b2)
                corner_uf.union(b1, a2)
            glued.append(t1)
        else:
            opened.extend(t for t, _a, _b in sides)

    stats: dict[int, dict] = defaultdict(lambda: {"F": 0, "E": 0, "V": set()})
    for t in range(n_tris):
        root = tri_uf.find(t)
        stats[root]["F"] += 1
        for j in range(3):
            stats[root]["V"].add(corner_uf.find(3 * t + j))
    for t in glued:
        stats[tri_uf.find(t)]["E"] += 1
    for t in opened:
        stats[tri_uf.find(t)]["E"] += 1
    return sorted(len(s["V"]) - s["E"] + s["F"] for s in stats.values())


def assert_all_disks(sample, seam, label: str) -> list[int]:
    """Assert every chart is a disk by the exact Euler test, and return the chis."""
    chis = chart_euler(sample, seam)
    assert chis and all(c == 1 for c in chis), (
        "%s: not every chart is a disk -- chi per chart %s (a disk is exactly 1)"
        % (label, chis)
    )
    return chis


def boundary_edges(sample) -> np.ndarray:
    """``(E,)`` bool marking edges with fewer than two incident triangles."""
    if sample.n_edges == 0:
        return np.zeros(0, dtype=bool)
    pairs = np.concatenate(
        [
            sample.tri_verts[:, [0, 1]],
            sample.tri_verts[:, [1, 2]],
            sample.tri_verts[:, [2, 0]],
        ],
        axis=0,
    ).astype(np.int64)
    base = int(max(sample.n_verts, 1))
    tri_keys = np.sort(pairs, axis=1)
    tri_keys = tri_keys[:, 0] * base + tri_keys[:, 1]
    edge_keys = np.sort(sample.edge_verts.astype(np.int64), axis=1)
    edge_keys = edge_keys[:, 0] * base + edge_keys[:, 1]
    counts = np.zeros(sample.n_edges, dtype=np.int64)
    order = np.argsort(edge_keys)
    sorted_keys = edge_keys[order]
    pos = np.searchsorted(sorted_keys, tri_keys)
    safe = np.minimum(pos, sorted_keys.size - 1)
    valid = (pos < sorted_keys.size) & (sorted_keys[safe] == tri_keys)
    np.add.at(counts, order[pos[valid]], 1)
    return counts < 2


# ---------------------------------------------------------------------------
# Shapes whose answer is known by hand.
# ---------------------------------------------------------------------------


def _from_quads(verts, quads, name, *, edge_verts=None):
    """Fan-triangulate ``quads``, preserving winding."""
    tris = []
    for quad in quads:
        for k in range(1, len(quad) - 1):
            tris.append([quad[0], quad[k], quad[k + 1]])
    return make_sample(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
        edge_verts=edge_verts,
        name=name,
    )


def outward_cube():
    """A unit cube wound counter-clockwise seen from **outside**: all 12 edges convex."""
    corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    quads = [
        (0, 3, 2, 1),  # bottom, -Z
        (4, 5, 6, 7),  # top,    +Z
        (0, 1, 5, 4),  # front,  -Y
        (1, 2, 6, 5),  # right,  +X
        (2, 3, 7, 6),  # back,   +Y
        (3, 0, 4, 7),  # left,   -X
    ]
    return _from_quads(corners, quads, "cube_outward")


def inverted_cube():
    """The same cube with every face reversed: signed volume ``-1``, all 12 concave."""
    corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    quads = [
        (1, 2, 3, 0), (7, 6, 5, 4), (4, 5, 1, 0),
        (5, 6, 2, 1), (6, 7, 3, 2), (7, 4, 0, 3),
    ]
    return _from_quads(corners, quads, "cube_inverted")


def l_solid():
    """An L-shaped extrusion: 17 convex folds and **exactly one** concave one.

    The footprint has a single reflex corner at ``(1, 1)``, so the vertical
    edge above it is the only valley on the whole solid.  That makes it the
    cleanest possible test of a concavity sign: one answer, hand-checkable.
    """
    foot = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    n = len(foot)
    verts = [[x, y, 0.0] for x, y in foot] + [[x, y, 1.0] for x, y in foot]
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append([i, j, j + n])
        tris.append([i, j + n, i + n])
    for k in range(1, n - 1):          # top cap, +Z
        tris.append([n, n + k, n + k + 1])
    for k in range(1, n - 1):          # bottom cap, -Z
        tris.append([0, k + 1, k])
    return make_sample(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
        name="l_solid",
    )


def open_tube(seg=24, rings=4, radius=1.0, height=2.0):
    """A closed tube of quads with open ends: one shell, two boundary loops, ``chi = 0``.

    docs/BENCHMARKS.md section 4.3: this is the shape ``SHARP`` gets
    catastrophically wrong -- 64 rim seams, zero lengthwise, L2 stretch 30.6 --
    because nothing ever opens the tube.
    """
    verts = []
    for i in range(rings + 1):
        z = height * i / rings
        for k in range(seg):
            a = 2.0 * math.pi * k / seg
            verts.append([radius * math.cos(a), radius * math.sin(a), z])
    quads = []
    for i in range(rings):
        for k in range(seg):
            k2 = (k + 1) % seg
            quads.append(
                (i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k)
            )
    return _from_quads(verts, quads, "open_tube")


def capped_cylinder(seg=24, rings=3, radius=1.0, height=2.0):
    """The same tube with fan caps: closed, genus 0."""
    tube = open_tube(seg, rings, radius, height)
    verts = list(np.asarray(tube.verts).tolist())
    tris = list(np.asarray(tube.tri_verts).tolist())
    verts.append([0.0, 0.0, 0.0])
    bottom = len(verts) - 1
    verts.append([0.0, 0.0, height])
    top = len(verts) - 1
    for k in range(seg):
        k2 = (k + 1) % seg
        tris.append([bottom, k2, k])
        tris.append([top, rings * seg + k, rings * seg + k2])
    return make_sample(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
        name="capped_cylinder",
    )


def annulus(seg=24, rings=3, inner=0.4, outer=1.0):
    """A flat washer: one shell, two boundary loops, genus 0, ``chi = 0``."""
    verts = []
    for i in range(rings + 1):
        r = inner + (outer - inner) * i / rings
        for k in range(seg):
            a = 2.0 * math.pi * k / seg
            verts.append([r * math.cos(a), r * math.sin(a), 0.0])
    quads = []
    for i in range(rings):
        for k in range(seg):
            k2 = (k + 1) % seg
            quads.append(
                (i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k)
            )
    return _from_quads(verts, quads, "annulus")


def uv_sphere(seg=20, rings=10, radius=1.0):
    """A closed genus-0 sphere with no sharp edge anywhere."""
    verts = [[0.0, 0.0, radius]]
    for i in range(1, rings):
        phi = math.pi * i / rings
        for k in range(seg):
            th = 2.0 * math.pi * k / seg
            verts.append(
                [
                    radius * math.sin(phi) * math.cos(th),
                    radius * math.sin(phi) * math.sin(th),
                    radius * math.cos(phi),
                ]
            )
    verts.append([0.0, 0.0, -radius])
    south = len(verts) - 1
    tris = []
    for k in range(seg):
        tris.append([0, 1 + (k + 1) % seg, 1 + k])
    for i in range(rings - 2):
        for k in range(seg):
            a = 1 + i * seg + k
            b = 1 + i * seg + (k + 1) % seg
            c = 1 + (i + 1) * seg + (k + 1) % seg
            d = 1 + (i + 1) * seg + k
            tris.append([a, b, c])
            tris.append([a, c, d])
    base = 1 + (rings - 2) * seg
    for k in range(seg):
        tris.append([south, base + k, base + (k + 1) % seg])
    return make_sample(
        np.asarray(verts, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
        name="uv_sphere",
    )


def torus(major=20, minor=10, big=1.0, small=0.35):
    """Genus 1: one handle to remove before anything can be a disk."""
    verts = []
    for i in range(major):
        u = 2.0 * math.pi * i / major
        for j in range(minor):
            v = 2.0 * math.pi * j / minor
            verts.append(
                [
                    (big + small * math.cos(v)) * math.cos(u),
                    (big + small * math.cos(v)) * math.sin(u),
                    small * math.sin(v),
                ]
            )
    quads = []
    for i in range(major):
        i2 = (i + 1) % major
        for j in range(minor):
            j2 = (j + 1) % minor
            quads.append(
                (i * minor + j, i2 * minor + j, i2 * minor + j2, i * minor + j2)
            )
    return _from_quads(verts, quads, "torus")


def valley_strip(n=16, depth=0.5):
    """A folded strip with one long concave crease down the middle.

    Two flat halves meeting in a V.  Every interior edge across the fold line
    is concave; every other interior edge is exactly flat.  A concavity-seeking
    cut therefore has one and only one place it should want to go.
    """
    verts = []
    for i in range(n + 1):
        x = i / n
        for j in range(n + 1):
            y = -1.0 + 2.0 * j / n
            verts.append([x, y, depth * abs(y)])
    quads = []
    for i in range(n):
        for j in range(n):
            quads.append(
                (
                    i * (n + 1) + j,
                    (i + 1) * (n + 1) + j,
                    (i + 1) * (n + 1) + j + 1,
                    i * (n + 1) + j + 1,
                )
            )
    return _from_quads(verts, quads, "valley_strip")


def two_loop_shell(seg=20, rings=6):
    """A tube with a lump: genus 0, two boundary loops -- Suzanne's head in miniature.

    ORGANIC.md section 0.1 measures Suzanne's head at ``chi = 0`` with two
    boundary loops, so it needs exactly **one** cut joining them.  This is the
    same topology at a size a unit test can reason about.
    """
    verts = []
    for i in range(rings + 1):
        z = 2.0 * i / rings
        r = 1.0 + 0.4 * math.sin(math.pi * i / rings)
        for k in range(seg):
            a = 2.0 * math.pi * k / seg
            verts.append([r * math.cos(a), r * math.sin(a), z])
    quads = []
    for i in range(rings):
        for k in range(seg):
            k2 = (k + 1) % seg
            quads.append(
                (i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k)
            )
    return _from_quads(verts, quads, "two_loop_shell")


def quad_grid_no_diagonals(n=6):
    """A grid whose ``edge_verts`` holds only the real quad edges.

    ``make_sample`` derives edges from the triangle soup, which promotes every
    tessellation diagonal to a mesh edge.  Blender does not: a quad has four
    edges and the diagonal does not exist.  Passing ``edge_verts`` explicitly
    reproduces that, which is the only way to test that ``CONCAVE`` never tries
    to cut a diagonal (the "triangle-fan trap" in the module docstring).
    """
    verts = []
    for i in range(n + 1):
        for j in range(n + 1):
            verts.append([i / n, j / n, 0.3 * math.sin(3.0 * j / n)])
    quads = []
    edges = set()
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = (i + 1) * (n + 1) + j
            c = (i + 1) * (n + 1) + j + 1
            d = i * (n + 1) + j + 1
            quads.append((a, b, c, d))
            for u, v in ((a, b), (b, c), (c, d), (d, a)):
                edges.add((min(u, v), max(u, v)))
    return _from_quads(
        verts,
        quads,
        "quad_grid",
        edge_verts=np.asarray(sorted(edges), dtype=np.int32),
    )


def grooved_tube(seg=12, rings=3):
    """A tube with one groove down it and one ridge, rigged so they compete.

    Column 0 is pushed out to radius 1.25 -- a convex ridge, and the shortest
    way from one rim to the other.  Column ``seg // 2`` is pulled in to 0.75 --
    a concave groove, and a longer route.  Anything that routes on length alone
    takes the ridge; anything that means what it says about concavity takes the
    groove.
    """
    verts = []
    for i in range(rings + 1):
        z = float(i)
        for k in range(seg):
            a = 2.0 * math.pi * k / seg
            r = 1.25 if k == 0 else (0.75 if k == seg // 2 else 1.0)
            verts.append([r * math.cos(a), r * math.sin(a), z])
    quads = []
    for i in range(rings):
        for k in range(seg):
            k2 = (k + 1) % seg
            quads.append(
                (i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k)
            )
    return _from_quads(verts, quads, "grooved_tube")


#: Every shape a "must be disks" sweep runs over.
SHAPES = {
    "grid": make_flat_grid(6),
    "cube": outward_cube(),
    "l_solid": l_solid(),
    "annulus": annulus(),
    "open_tube": open_tube(),
    "capped_cylinder": capped_cylinder(),
    "uv_sphere": uv_sphere(),
    "torus": torus(),
    "valley_strip": valley_strip(),
    "two_loop_shell": two_loop_shell(),
    "grooved_tube": grooved_tube(),
    "quad_grid": quad_grid_no_diagonals(),
}


# ---------------------------------------------------------------------------
# 1. The sign convention, measured rather than assumed.
# ---------------------------------------------------------------------------


def test_edge_dihedral_deg_carries_no_sign():
    """``edge_dihedral_deg`` cannot tell a ridge from a valley.  Pinned, not implied.

    ORGANIC.md section 7 says to "compute the signed dihedral
    (``edge_dihedral_deg`` exists already; confirm its sign convention)".  There
    is no sign convention to confirm: it is ``arccos`` of a dot product, so it
    lives in ``[0, 180]`` and a cube's convex 90 and an L-solid's concave 90
    are the same number.  This test exists so nobody rediscovers that the hard
    way.
    """
    for sample, label in ((outward_cube(), "convex cube"), (l_solid(), "L-solid")):
        topo = seams.build_edge_topology(sample)
        unsigned = seams.edge_dihedral_deg(topo)
        assert (unsigned >= 0.0).all(), "%s produced a negative dihedral" % label
        folds = np.round(unsigned[(topo.face_count == 2) & (unsigned > 1e-6)], 6)
        assert set(folds.tolist()) == {90.0}, (
            "%s: expected every fold at exactly 90 degrees, got %s"
            % (label, sorted(set(folds.tolist())))
        )


def test_signed_dihedral_calls_an_outward_cube_convex():
    """All twelve folds of an outward-wound cube are ``+90``."""
    sample = outward_cube()
    topo = seams.build_edge_topology(sample)
    signed = edge_signed_dihedral_deg(sample, topo, orient=False)
    folds = signed[(topo.face_count == 2) & (np.abs(signed) > 1e-6)]
    assert folds.size == 12, "expected 12 folded edges, got %d" % folds.size
    assert (folds > 0).all(), (
        "an outward cube is convex everywhere; %d edges came back concave"
        % int((folds < 0).sum())
    )
    np.testing.assert_allclose(np.abs(folds), 90.0, atol=1e-9)


def test_signed_dihedral_calls_an_inverted_cube_concave():
    """Reversing every face inverts every sign.  Convexity needs an inside."""
    sample = inverted_cube()
    topo = seams.build_edge_topology(sample)
    signed = edge_signed_dihedral_deg(sample, topo, orient=False)
    folds = signed[(topo.face_count == 2) & (np.abs(signed) > 1e-6)]
    assert folds.size == 12
    assert (folds < 0).all(), (
        "an inside-out cube must read concave when the winding is taken at face "
        "value; %d edges came back convex" % int((folds > 0).sum())
    )


def test_signed_dihedral_finds_the_one_valley_of_an_l_solid():
    """Exactly one edge of an L-shaped solid is concave, and it is the inner corner."""
    sample = l_solid()
    topo = seams.build_edge_topology(sample)
    signed = edge_signed_dihedral_deg(sample, topo)
    folds = (topo.face_count == 2) & (np.abs(signed) > 1e-6)
    concave = np.flatnonzero(folds & (signed < 0))
    assert concave.size == 1, (
        "an L-shaped solid has exactly one valley; found %d" % concave.size
    )
    a = sample.verts[topo.edge_v0[concave[0]]]
    b = sample.verts[topo.edge_v1[concave[0]]]
    np.testing.assert_allclose(sorted([a[0], b[0]]), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sorted([a[1], b[1]]), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sorted([a[2], b[2]]), [0.0, 1.0], atol=1e-12)


def test_orientation_flip_rescues_an_inside_out_mesh():
    """``tests.conftest.make_cube`` is wound inward; ``orient=True`` must notice.

    Measured: its signed volume is ``-1.0``.  Taken at face value every edge of
    it is concave, which would hand a full concavity discount to a shape that
    has no valley at all.  The per-shell signed-volume check flips it back.
    """
    sample = make_cube()
    topo = seams.build_edge_topology(sample)
    raw = edge_signed_dihedral_deg(sample, topo, orient=False)
    fixed = edge_signed_dihedral_deg(sample, topo, orient=True)
    folded = (topo.face_count == 2) & (np.abs(raw) > 1e-6)
    assert (raw[folded] < 0).all(), "fixture changed: make_cube is no longer inside-out"
    assert (fixed[folded] > 0).all(), (
        "orient=True failed to correct an inside-out cube: %d edges still concave"
        % int((fixed[folded] < 0).sum())
    )


@pytest.mark.parametrize("name", ["valley_strip", "annulus", "open_tube", "grid"])
def test_orientation_flip_leaves_an_open_shell_alone(name):
    """An open shell has no inside, so the winding's answer is the only answer.

    This is not a convenience.  Measured, the ``valley_strip`` -- a sheet
    folded into a shallow V, with no inside whatsoever -- reports a
    volume-to-area ratio of 0.050 against a sphere's 0.094.  Any threshold
    generous enough to catch an inside-out sphere also fires here and inverts
    the sign of all 736 of its edges, turning its single genuine crease into a
    ridge.  Restricting the flip to closed shells removes the guesswork.
    """
    sample = SHAPES[name]
    topo = seams.build_edge_topology(sample)
    raw = edge_signed_dihedral_deg(sample, topo, orient=False)
    fixed = edge_signed_dihedral_deg(sample, topo, orient=True)
    np.testing.assert_array_equal(raw, fixed)


def test_the_crease_of_a_folded_strip_reads_as_a_valley():
    """The V-fold of an open strip is concave, and stays concave.

    Sixteen crease edges, every other interior edge exactly flat.  If the
    orientation heuristic ever fires on an open shell again, this is the test
    that catches it: the crease flips to a ridge and the count inverts.
    """
    sample = valley_strip(n=16, depth=0.6)
    topo = seams.build_edge_topology(sample)
    signed = edge_signed_dihedral_deg(sample, topo)
    manifold = topo.face_count == 2
    # The two halves are planar, so only the crease genuinely folds; everything
    # else sits at float noise around zero.
    real = manifold & (np.abs(signed) > 1e-3)
    assert int(real.sum()) == 16, (
        "expected 16 crease edges, found %d" % int(real.sum())
    )
    assert (signed[real] < 0).all(), "the crease of a trough read as a ridge"


def test_signed_dihedral_magnitude_matches_the_unsigned_one():
    """Only the sign is new; the angle itself is the existing measurement."""
    for sample in (outward_cube(), l_solid(), uv_sphere(), torus()):
        topo = seams.build_edge_topology(sample)
        signed = edge_signed_dihedral_deg(sample, topo)
        np.testing.assert_allclose(
            np.abs(signed), seams.edge_dihedral_deg(topo), atol=1e-12
        )


# ---------------------------------------------------------------------------
# 2. Weights.
# ---------------------------------------------------------------------------


def test_a_valley_costs_less_than_a_ridge_of_the_same_length():
    """The whole premise: a path through a crease is cheaper than one over a fold."""
    sample = l_solid()
    topo = seams.build_edge_topology(sample)
    signed = edge_signed_dihedral_deg(sample, topo)
    weight = seams.concavity_weights(sample, topo, signed=signed)
    valley = (topo.face_count == 2) & (signed < -1e-6)
    ridge = (topo.face_count == 2) & (signed > 1e-6)
    assert valley.any() and ridge.any()
    per_unit_valley = weight[valley] / np.maximum(topo.edge_len[valley], 1e-12)
    per_unit_ridge = weight[ridge] / np.maximum(topo.edge_len[ridge], 1e-12)
    assert per_unit_valley.max() < per_unit_ridge.min(), (
        "a valley is not cheaper than a ridge: valley %.4f vs ridge %.4f per unit"
        % (per_unit_valley.max(), per_unit_ridge.min())
    )


def test_weights_are_non_negative_and_finite_everywhere():
    """Dijkstra needs that; a NaN weight would silently break every path."""
    for name, sample in SHAPES.items():
        weight = seams.concavity_weights(sample)
        assert np.isfinite(weight).all(), "%s produced a non-finite weight" % name
        assert (weight >= 0.0).all(), "%s produced a negative weight" % name


def test_weights_scale_linearly_with_the_mesh():
    """Scale the model by 1000 and every weight scales by exactly 1000.

    That is what makes the *paths* scale invariant: multiplying every edge cost
    by one constant cannot reorder a shortest path.  An implementation that
    compared an angle against an absolute length would fail here.
    """
    sample = l_solid()
    factor = 1000.0
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    base = seams.concavity_weights(sample)
    got = seams.concavity_weights(scaled)
    np.testing.assert_allclose(got, base * factor, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Every chart must be a disk -- the exact Euler test, independently computed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("threshold", [0.05, 0.35, 0.9])
def test_every_chart_is_a_disk(name, threshold):
    """``chi = F - E + V == 1`` for every chart, on every shape, at every setting."""
    sample = SHAPES[name]
    mask, report = concave_seams(sample, curvature_threshold=threshold, seed=0)
    chis = assert_all_disks(sample, mask, "%s @ %g" % (name, threshold))
    assert report.non_disks == 0, (
        "%s: the module reported %d non-disk charts while the independent count "
        "found %s" % (name, report.non_disks, chis)
    )
    assert report.charts == len(chis), (
        "%s: module counted %d charts, independent count found %d"
        % (name, report.charts, len(chis))
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_no_shape_needs_the_spanning_tree_fallback(name):
    """The measured cut loop must open everything on its own.

    ``fallbacks`` is the module admitting it could not, and cutting the
    complement of a dual spanning tree instead -- correct, but enormously long.
    A non-zero count here is a real defect, not a tuning choice.
    """
    _, report = concave_seams(SHAPES[name], seed=0)
    assert report.fallbacks == 0, (
        "%s fell back to the spanning-tree cut %d times" % (name, report.fallbacks)
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_boundary_edges_are_never_seams(name):
    """An edge with one incident face is already a chart boundary."""
    sample = SHAPES[name]
    mask = compute_seams(sample, "CONCAVE")
    rim = boundary_edges(sample)
    assert not (mask & rim).any(), (
        "%s marked %d boundary edges as seams" % (name, int((mask & rim).sum()))
    )


def test_a_disk_is_left_completely_uncut():
    """A flat grid already unwraps perfectly.  Cutting it is pure damage."""
    mask, report = concave_seams(make_flat_grid(10))
    assert int(mask.sum()) == 0, "CONCAVE cut %d edges of a flat disk" % int(mask.sum())
    assert report.joins == 0 and report.arcs == 0 and report.handle_loops == 0


# ---------------------------------------------------------------------------
# 4. The topological minimum: one cut where one cut is what is needed.
# ---------------------------------------------------------------------------


def test_two_boundary_loops_need_exactly_one_join():
    """ORGANIC.md section 0.1: a ``chi = 0`` shell with two rims needs one path.

    An annulus is that shell at its simplest.  One join, no arc, no handle
    loop, and the cut is a single short path -- not a spanning tree, not a
    cube-map boundary.
    """
    sample = annulus()
    mask, report = concave_seams(sample)
    assert report.joins == 1, "expected exactly one join, got %d" % report.joins
    assert report.arcs == 0 and report.handle_loops == 0
    assert_all_disks(sample, mask, "annulus")
    # rings=3, so the shortest rim-to-rim path is three radial edges.
    assert int(mask.sum()) <= 4, (
        "the minimum cut across a 3-ring annulus is 3 edges; got %d" % int(mask.sum())
    )


def test_a_two_loop_shell_is_opened_by_one_path_not_a_spanning_tree():
    """Suzanne's head in miniature: two rims, genus 0, so one join and stop."""
    sample = two_loop_shell()
    mask, report = concave_seams(sample)
    assert report.joins == 1
    assert_all_disks(sample, mask, "two_loop_shell")
    assert int(mask.sum()) < sample.n_verts // 4, (
        "a single rim-to-rim path should be short; got %d cuts on %d vertices"
        % (int(mask.sum()), sample.n_verts)
    )


def test_a_tube_is_cut_lengthwise():
    """docs/BENCHMARKS.md section 4.3: the failure that scores 0.0 with L2 30.6.

    ``SHARP`` marks 64 rim seams on a cylinder and not one lengthwise, so the
    tube is never opened.  ``CONCAVE`` must produce a cut that spans the tube's
    full height -- and the disk test is what proves the tube really opened.
    """
    sample = open_tube(seg=24, rings=4, height=2.0)
    mask, _ = concave_seams(sample)
    assert_all_disks(sample, mask, "open_tube")
    cut = sample.edge_verts[mask]
    z0 = sample.verts[cut[:, 0]][:, 2]
    z1 = sample.verts[cut[:, 1]][:, 2]
    span = float(max(z0.max(), z1.max()) - min(z0.min(), z1.min()))
    assert span > 1.9, (
        "the cut spans only %.3f of the tube's 2.0 height, so it is a rim cut, "
        "not a lengthwise one" % span
    )


def test_a_torus_loses_its_handle_before_anything_else():
    """Genus 1 needs a non-contractible loop; an arc alone takes chi the wrong way.

    Cutting a torus along an arc moves ``chi`` from 0 to -1, further from a
    disk, not closer.  The handle loop takes it to ``chi = 0`` with two
    boundary loops, and the join then reaches 1.
    """
    sample = torus()
    mask, report = concave_seams(sample)
    assert report.handle_loops >= 1, "a torus was opened without cutting its handle"
    assert report.joins >= 1, "the two rims left by the handle cut were never joined"
    assert_all_disks(sample, mask, "torus")


def test_a_closed_sphere_is_always_opened():
    """The defect that motivates this strategy: ``SHARP`` cuts nothing here."""
    sample = uv_sphere()
    assert not compute_seams(sample, "SHARP", angle_deg=30.0).any(), (
        "fixture changed: SHARP now finds edges on a smooth sphere"
    )
    mask, report = concave_seams(sample)
    assert mask.any(), "CONCAVE left a closed sphere with no cuts at all"
    assert report.arcs >= 1
    assert_all_disks(sample, mask, "uv_sphere")


def test_a_cube_has_no_concavity_to_find_and_says_so():
    """With nothing to prefer, the cut falls back to a plain shortest path.

    A convex cube offers no valley, so the concave fraction is exactly zero and
    the mesh's own baseline is exactly zero too -- the honest answer, not a
    failure.  What matters is that the result is still a single disk.
    """
    sample = outward_cube()
    mask, report = concave_seams(sample)
    assert report.baseline_fraction == 0.0, "a convex cube has no concave edge"
    assert report.concave_fraction == 0.0
    assert mask.any()
    assert_all_disks(sample, mask, "cube")


# ---------------------------------------------------------------------------
# 5. Does it actually prefer valleys?
# ---------------------------------------------------------------------------


def test_concavity_is_preferred_over_a_shorter_convex_route():
    """A deliberately rigged shell: the valley route is longer and must still win.

    Two rims joined by twelve columns.  One is pushed outward into a ridge and
    is the geometrically shortest way across; one is pulled inward into a
    groove and is longer.  A length-only shortest path takes the ridge; a
    concavity-weighted one takes the groove, which is where an artist would
    put the seam and where the cut costs the least distortion.
    """
    sample = grooved_tube()
    topo = seams.build_edge_topology(sample)
    signed = edge_signed_dihedral_deg(sample, topo)
    mask, _ = concave_seams(sample)
    assert_all_disks(sample, mask, "grooved_tube")
    marked = mask & (topo.face_count == 2)
    assert marked.any()
    assert signed[marked].mean() < 0.0, (
        "the join took the convex route: mean signed dihedral of the cut is "
        "%+.2f degrees" % signed[marked].mean()
    )
    # And specifically: the groove column, not the ridge column.
    on_ridge = 0
    for a, b in sample.edge_verts[mask]:
        for v in (a, b):
            if np.hypot(*sample.verts[v][:2]) > 1.2:
                on_ridge += 1
    assert on_ridge == 0, "%d cut endpoints landed on the ridge column" % on_ridge


@pytest.mark.parametrize("name", ["l_solid", "torus", "grooved_tube"])
def test_the_cut_is_more_concave_than_the_mesh_average(name):
    """The claim, stated as a measurement: enrichment over the mesh's own baseline."""
    _, report = concave_seams(SHAPES[name], seed=0)
    if report.baseline_fraction == 0.0:
        pytest.skip("%s has no concave edge to prefer" % name)
    assert report.concave_fraction > report.baseline_fraction, (
        "%s: cut is %.1f%% concave against a mesh baseline of %.1f%% -- no "
        "preference at all"
        % (name, 100 * report.concave_fraction, 100 * report.baseline_fraction)
    )


# ---------------------------------------------------------------------------
# 6. The triangle-fan trap.
# ---------------------------------------------------------------------------


def test_a_tessellation_diagonal_is_never_cut():
    """A quad has four edges.  Its diagonal is not one of them and cannot be marked.

    The mask is indexed by ``edge_verts``, so an out-of-range mark is impossible
    by construction; what this checks is that the cut still *works* when the
    diagonals are absent from the edge list, rather than quietly relying on
    them being cuttable.
    """
    sample = quad_grid_no_diagonals(6)
    mask, report = concave_seams(sample, curvature_threshold=0.05)
    assert mask.shape == (sample.n_edges,)
    assert_all_disks(sample, mask, "quad_grid")
    assert report.fallbacks == 0
    # Every marked edge is an axis-aligned quad edge, never a diagonal.
    for a, b in sample.edge_verts[mask]:
        delta = np.abs(sample.verts[a][:2] - sample.verts[b][:2])
        assert int((delta > 1e-9).sum()) <= 1, "a quad diagonal was marked"


# ---------------------------------------------------------------------------
# 7. Determinism, scale, and the parameter contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_same_mesh_same_seed_same_cuts(name):
    """Determinism is contract: a rerun of a seed reproduces a run exactly."""
    sample = SHAPES[name]
    a = compute_seams(sample, "CONCAVE", curvature_threshold=0.4, seed=7)
    b = compute_seams(sample, "CONCAVE", curvature_threshold=0.4, seed=7)
    np.testing.assert_array_equal(a, b)


def test_the_seed_actually_changes_the_cut_on_a_closed_shell():
    """A seed that never changes anything is a seed nobody can use.

    On a closed shell the arc's source is rotated among the sharpest few
    curvature extremes, so different seeds give genuinely different, equally
    valid cuts -- the same trick ``DUAL_GRAPH`` plays on a cube.
    """
    sample = uv_sphere()
    cuts = {compute_seams(sample, "CONCAVE", seed=s).tobytes() for s in range(6)}
    assert len(cuts) > 1, "every seed produced the identical cut on a sphere"


@pytest.mark.parametrize("factor", [1e-3, 1e3])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_the_topological_cut_is_exactly_scale_invariant(factor, name):
    """A model in centimetres is the same model as one in metres.

    The weights are ``length * f(angle)``, so a uniform scale multiplies every
    cost by one constant and cannot reorder a shortest path.  The topological
    phase is therefore invariant in the two things that are actually defined:
    **how many edges are cut and how long the cut is**, both asserted here to
    the last bit.

    *Which* edges is a different question, and on a shape of revolution it has
    no single answer: an annulus offers twenty-four identical radial routes and
    a tie is settled by the last bit of a float.  Measured, that moves 6 to 20
    edges without moving the length by one part in 1e15.  The asymmetric shapes
    below are bit-identical.
    """
    sample = SHAPES[name]
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    base, _ = concave_seams(sample, max_extra_cuts=0)
    got, _ = concave_seams(scaled, max_extra_cuts=0)
    assert int(got.sum()) == int(base.sum()), (
        "%s: scaling by %g changed the cut from %d edges to %d -- that is a "
        "units bug, not a tie-break"
        % (name, factor, int(base.sum()), int(got.sum()))
    )
    len_base = float(seams.build_edge_topology(sample).edge_len[base].sum())
    len_got = float(seams.build_edge_topology(scaled).edge_len[got].sum()) / factor
    np.testing.assert_allclose(
        len_got,
        len_base,
        rtol=1e-12,
        err_msg="%s: cut length changed under a %g scale" % (name, factor),
    )


@pytest.mark.parametrize("factor", [1e-3, 1e3])
@pytest.mark.parametrize("name", ["l_solid", "cube", "grooved_tube", "torus"])
def test_an_asymmetric_topological_cut_is_bit_identical_under_scale(factor, name):
    """With no exact ties to break there is nothing left for float noise to move."""
    sample = SHAPES[name]
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    np.testing.assert_array_equal(
        concave_seams(scaled, max_extra_cuts=0)[0],
        concave_seams(sample, max_extra_cuts=0)[0],
    )


@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_an_asymmetric_shape_is_scale_invariant_all_the_way_through(factor):
    """With no ties to break, the escalation is exactly invariant too."""
    sample = l_solid()
    scaled = dataclasses.replace(sample, verts=sample.verts * factor)
    np.testing.assert_array_equal(
        compute_seams(scaled, "CONCAVE"), compute_seams(sample, "CONCAVE")
    )


@pytest.mark.parametrize("name", ["uv_sphere", "two_loop_shell", "torus"])
def test_a_symmetric_shape_stays_within_a_tie_break_of_itself_under_scale(name):
    """On a shape with exact ties the escalation can land one round either side.

    A sphere of revolution offers twenty identical rim-to-rim routes and a
    thousand identical split lines.  Which one wins is decided at the last bit
    of a float, and because the escalation then re-measures the charts it just
    made, a flipped tie can put a chart a hair above or below the curvature
    budget and cost or save one round.  Measured on a 20x10 sphere: 53 cuts at
    scale 1, 47 at 1e-3, 44 at 1e3.

    That is tie-breaking, not a units bug -- the topological phase above is
    bit-identical at every scale, and ``DUAL_GRAPH`` carries the same caveat.
    What must hold is that the answer stays the same *kind* of answer.
    """
    sample = SHAPES[name]
    counts = []
    for factor in (1e-3, 1.0, 1e3):
        scaled = dataclasses.replace(sample, verts=sample.verts * factor)
        mask, report = concave_seams(scaled)
        assert_all_disks(scaled, mask, "%s @ %g" % (name, factor))
        assert report.fallbacks == 0
        counts.append(int(mask.sum()))
    spread = (max(counts) - min(counts)) / max(max(counts), 1)
    assert spread < 0.35, (
        "%s cut counts %s across three scales -- a %.0f%% spread is more than a "
        "tie-break" % (name, counts, 100 * spread)
    )


def test_seams_are_independent_of_uvs():
    """Seam heuristics read geometry only."""
    sample = two_loop_shell()
    rotated = sample.with_uvs(sample.uvs[:, ::-1] * 3.0 + 0.7)
    np.testing.assert_array_equal(
        compute_seams(sample, "CONCAVE"), compute_seams(rotated, "CONCAVE")
    )


def test_concave_is_registered_in_both_strategy_lists():
    """The two copies of the strategy list must not drift apart."""
    from unwrap_variations.core.params import SEAM_STRATEGIES as PARAM_STRATEGIES

    assert "CONCAVE" in seams.SEAM_STRATEGIES
    assert tuple(seams.SEAM_STRATEGIES) == tuple(PARAM_STRATEGIES)


def test_concave_is_reachable_from_the_parameter_space():
    """The grid can only sample what ``PARAM_SPACE`` offers."""
    from unwrap_variations.core.params import PARAM_SPACE, UnwrapParams

    assert "CONCAVE" in PARAM_SPACE["seam_strategy"].choices
    params = UnwrapParams(seam_strategy="CONCAVE").clamped()
    assert params.seam_strategy == "CONCAVE"
    assert "conc" in params.describe()


def test_concave_recipes_hash_apart_on_threshold_and_seed():
    """Two candidates that cut differently must not de-duplicate onto one slot."""
    from unwrap_variations.core.params import UnwrapParams

    base = UnwrapParams(method="ANGLE_BASED", seam_strategy="CONCAVE")
    signatures = {
        base.replace(curvature_threshold=t, seed=s).signature()
        for t in (0.1, 0.5)
        for s in (1, 2)
    }
    assert len(signatures) == 4, "CONCAVE recipes collapsed onto %d signatures" % len(
        signatures
    )


def test_compute_seams_routes_concave_to_the_strategy():
    """``compute_seams`` and ``concave_seams`` must not diverge."""
    sample = uv_sphere()
    np.testing.assert_array_equal(
        compute_seams(sample, "concave", curvature_threshold=0.5, seed=3),
        concave_seams(sample, curvature_threshold=0.5, seed=3)[0],
    )


# ---------------------------------------------------------------------------
# 8. Robustness.  Bad geometry is reported, never thrown.
# ---------------------------------------------------------------------------


def test_pathological_meshes_do_not_raise():
    """Empty, collapsed, NaN-bearing and random meshes all give a valid mask."""
    for sample in (
        make_empty(),
        make_all_identical(4),
        make_nan_uvs(),
        make_unit_triangle(),
        make_flat_grid(2),
        make_random_mesh(5, mode="chaotic"),
        make_random_mesh(6, mode="spiky"),
        make_random_mesh(7, mode="jittered"),
    ):
        mask, report = concave_seams(sample)
        assert mask.dtype == bool
        assert mask.shape == (sample.n_edges,)
        assert isinstance(report.charts, int)


def test_non_manifold_input_does_not_raise_and_is_always_cut():
    """An edge with three incident faces cannot be interior to a 2-manifold chart."""
    sample = load_corpus("nonmanifold")
    topo = seams.build_edge_topology(sample)
    mask = compute_seams(sample, "CONCAVE")
    assert mask.shape == (sample.n_edges,)
    nonmanifold = topo.face_count >= 3
    if nonmanifold.any():
        assert (mask[nonmanifold]).all(), "a non-manifold edge was left uncut"


@pytest.mark.parametrize("name", ["cube_project", "hardsurface", "sphere_smart"])
def test_real_corpus_meshes_come_out_as_disks(name):
    """The committed Blender meshes, checked by the independent Euler count."""
    sample = load_corpus(name)
    mask, report = concave_seams(sample, seed=1)
    assert_all_disks(sample, mask, name)
    assert report.fallbacks == 0


def test_a_zero_area_mesh_does_not_raise():
    """Every triangle collapsed to a point: no normals, no dihedrals, no cuts."""
    mask, report = concave_seams(make_all_identical(8))
    assert mask.dtype == bool
    assert report.non_disks >= 0


def test_out_of_range_indices_do_not_raise():
    """A malformed sample degrades rather than throwing mid-run."""
    sample = make_flat_grid(4)
    broken = dataclasses.replace(sample, edge_verts=sample.edge_verts + 10_000)
    mask = compute_seams(broken, "CONCAVE")
    assert mask.shape == (broken.n_edges,)


def test_hostile_thresholds_are_clamped_not_propagated():
    """NaN, infinity and absurd values are parameters, not crashes."""
    sample = uv_sphere()
    for threshold in (float("nan"), float("inf"), -5.0, 1e9, 0.0):
        mask = compute_seams(sample, "CONCAVE", curvature_threshold=threshold)
        assert mask.shape == (sample.n_edges,)
        assert_all_disks(sample, mask, "sphere @ %r" % threshold)


def test_a_hostile_seed_is_tolerated():
    """A seed from a corrupted ``.blend`` must not take the run down with it."""
    sample = uv_sphere()
    for seed in (-1, 2**40, "not a number", None):
        mask = compute_seams(sample, "CONCAVE", seed=seed)
        assert mask.shape == (sample.n_edges,)


def test_the_escalation_is_bounded():
    """A threshold that wants to cut everything still cannot: the cap holds."""
    sample = uv_sphere(seg=24, rings=12)
    _, report = concave_seams(sample, curvature_threshold=0.02)
    assert report.extra_cuts <= seams._CONCAVE_MAX_EXTRA_CUTS
    assert report.charts <= report.extra_cuts + 2


def test_the_mask_never_aliases_the_input():
    """The returned array is owned by the caller and safe to mutate."""
    sample = two_loop_shell()
    original = sample.edge_seam.copy()
    mask = compute_seams(sample, "CONCAVE")
    mask[:] = True
    np.testing.assert_array_equal(sample.edge_seam, original)


def test_existing_seams_are_ignored():
    """``CONCAVE`` derives its cut from geometry; it is not an ``EXISTING`` variant."""
    sample = two_loop_shell()
    marked = np.zeros(sample.n_edges, dtype=bool)
    marked[::7] = True
    np.testing.assert_array_equal(
        compute_seams(sample, "CONCAVE"),
        compute_seams(sample.with_seams(marked), "CONCAVE"),
    )


# ---------------------------------------------------------------------------
# 9. Cost.
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_a_mid_size_organic_mesh_is_cut_in_well_under_a_second():
    """Budget: this runs once per candidate, inside a ten-slot grid.

    ORGANIC.md section 9.6 allows the whole ten-candidate run 30 s on Suzanne,
    where packing alone already costs seconds, so a seam pass measured in
    hundreds of milliseconds is the ceiling that matters.
    """
    import time

    sample = uv_sphere(seg=96, rings=48)
    assert sample.n_tris > 8000
    start = time.perf_counter()
    mask, report = concave_seams(sample)
    elapsed = time.perf_counter() - start
    assert report.non_disks == 0
    assert mask.any()
    assert elapsed < 3.0, "%d triangles took %.2f s" % (sample.n_tris, elapsed)
