"""Measure the quality of a UV layout.

:func:`evaluate` turns a :class:`~unwrap_variations.core.meshdata.UVMeshSample`
into a :class:`~unwrap_variations.core.metrics.UVMetrics`.  Everything here is
pure numpy -- no :mod:`bpy` -- so the whole measurement stack runs (and is
tested) in plain CPython.

Three properties are non-negotiable and shape most of the code below.

**It never raises.**  The evaluator sits in the middle of a generate loop that
must survive whatever a third-party unwrapper produces: NaN UVs, collapsed
triangles, index arrays that no longer match the mesh, an empty mesh.  Every
one of those returns a well-formed record with the damage reported in
``degenerate_tris`` rather than propagating an exception into a Blender
operator.

**It is fully vectorised.**  The target is 500k triangles in well under a
quarter second (excluding overlap detection), which rules out any per-triangle
Python and rules out a second pass over the big ``(T, 3, 3)`` arrays whenever
the same number can be recovered from a ``(T,)`` one.  That is why almost every
metric here is expressed through the triangle's *first fundamental form* -- see
below -- instead of through its corner positions.

**It is deterministic.**  Same arrays in, bit-identical floats out.  No
randomness, no hash iteration order, no parallel reduction.

The first fundamental form
--------------------------

With ``u1 = q2-q1`` and ``u2 = q3-q1`` the 3D edge vectors and
``E = u1.u1``, ``F = u1.u2``, ``G = u2.u2``, a triangle's entire intrinsic
geometry is those three scalars: its area is ``0.5*sqrt(EG - F^2)``, its squared
side lengths are ``E``, ``G`` and ``E - 2F + G``, and its corner angles follow
from the cosine rule.  The same holds in UV space with ``E'``, ``F'``, ``G'``
built from the UV edges.  Collapsing to six scalars per triangle up front turns
what would be a dozen passes over ``(T, 3, 3)`` float64 arrays into a dozen
passes over ``(T,)`` ones, which is roughly a 3x saving on the whole evaluation.

Stretch (Sander et al. 2001)
----------------------------

``stretch_l2`` / ``stretch_linf`` implement Sander et al., *Texture Mapping
Progressive Meshes* (SIGGRAPH 2001), section 3.  The affine map from texture
space to the surface has partial derivatives::

    Ss = (q1*(t2-t3) + q2*(t3-t1) + q3*(t1-t2)) / (2A)
    St = (q1*(s3-s2) + q2*(s1-s3) + q3*(s2-s1)) / (2A)

with ``A`` the signed UV area.  Written in terms of the UV edge components
``(ds1,dt1) = p2-p1`` and ``(ds2,dt2) = p3-p1`` and with ``D = 2A``, that is
``Ss = (u1*dt2 - u2*dt1)/D`` and ``St = (u2*ds1 - u1*ds2)/D``, so with
``a = Ss.Ss``, ``b = Ss.St``, ``c = St.St``::

    a = ( dt2^2 E       - 2 dt1 dt2 F              + dt1^2 G      ) / D^2
    b = (-ds2 dt2 E     + (ds1 dt2 + ds2 dt1) F    - ds1 dt1 G    ) / D^2
    c = ( ds2^2 E       - 2 ds1 ds2 F              + ds1^2 G      ) / D^2

-- three scalars per triangle, no 3-vectors.  The singular values of the
Jacobian are then::

    sigma_max = sqrt(((a+c) + sqrt((a-c)^2 + 4b^2)) / 2)
    sigma_min = sqrt(((a+c) - sqrt((a-c)^2 + 4b^2)) / 2)

and the per-triangle norms are ``L2(T) = sqrt((a+c)/2)`` (which equals
``sqrt((sigma_max^2 + sigma_min^2)/2)``) and ``Linf(T) = sigma_max``.  The mesh
norms are the surface-area-weighted RMS and the maximum respectively.

Both are scale dependent -- doubling every UV halves them -- so, exactly as the
paper prescribes, the parametrisation is normalised to equal total area before
the norms are reported.  That is a single multiply by
``sqrt(sum|uv_area| / sum surface_area)``, and it makes a perfectly isometric
unwrap score exactly ``1.0`` regardless of the UV scale it happens to use.
A parametrisation stretched 2x along one axis scores ``sqrt(5)/2 = 1.118034``
in L2 and ``sqrt(2) = 1.414214`` in Linf, which is what the unit tests check.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from .meshdata import UVMeshSample
from .metrics import UVMetrics

__all__ = ["evaluate"]


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Area below which a triangle is unusable for distortion maths.  Matches the
#: default of :func:`unwrap_variations.core.meshdata.degenerate_mask` so the two
#: never disagree about which triangles are countable.
_DEGENERATE_EPS = 1e-12

#: Relative tolerance for calling two UVs "the same UV vertex" when welding
#: loops into islands.  Expressed as a fraction of the layout's own extent (or
#: of 1.0, whichever is larger) so it behaves the same for a 0..1 layout and a
#: sprawling UDIM one.
_WELD_EPS = 1e-6

#: When True, welding runs a second pass on a half-cell-offset lattice and
#: merges the two groupings.  Quantisation alone can split a pair of UVs that
#: are within tolerance but straddle a cell boundary; two offset lattices cannot
#: both split the same pair.  The second pass runs over UV *vertices* rather
#: than loops, so it costs a sort over a few hundred thousand entries at most.
_WELD_DUAL_LATTICE = True

#: Hard cap on the union-of-AABBs rasterisation used by ``packing_efficiency``.
#: Below it the rasterisation uses compressed coordinates and is exact; above it
#: it falls back to a uniform lattice and is approximate.
_RECT_GRID_CELLS = 4_000_000

#: Overlap guards.  A pathological layout (every triangle stacked on the same
#: spot) would otherwise generate a quadratic number of candidate pairs.
_OVERLAP_MAX_INSERTS = 8_000_000
_OVERLAP_MAX_PAIRS = 2_000_000
_OVERLAP_CHUNK = 65_536
_OVERLAP_MAX_GRID_DIM = 4096

#: Tolerance for "outside the unit square", in UV units.
_OOB_TOL = 1e-6

_RAD_TO_DEG = 180.0 / math.pi
_TINY = 1e-300


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(sample: UVMeshSample, *, compute_overlap: bool = True) -> UVMetrics:
    """Measure every quality metric of ``sample``'s UV layout.

    Args:
        sample: The mesh plus the UV layer to evaluate.  Malformed samples are
            tolerated: triangles referencing out-of-range indices are dropped
            and counted as degenerate, non-finite coordinates are neutralised.
        compute_overlap: When False, ``overlap_area`` is reported as ``0.0`` and
            the spatial-hash pass is skipped entirely.  Overlap detection is the
            only super-linear step, so the interactive path turns it off while
            sweeping a large parameter grid and turns it back on for the final
            ranking.

    Returns:
        A fully populated :class:`~unwrap_variations.core.metrics.UVMetrics`.
        This function never raises; on catastrophic failure it returns a neutral
        record with ``n_tris`` and ``degenerate_tris`` filled in as best it can.
    """
    started = time.perf_counter()
    try:
        # Non-finite input is expected and handled explicitly by the masks
        # below, so numpy's warnings about it are noise -- and in Blender they
        # would land in the user's system console once per candidate.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            metrics = _evaluate_impl(sample, bool(compute_overlap))
    except Exception:  # noqa: BLE001 - containment is the whole point
        metrics = _fallback_metrics(sample)
    return replace(metrics, eval_seconds=time.perf_counter() - started)


def _fallback_metrics(sample: object) -> UVMetrics:
    """Return the most honest record we can build when evaluation blew up."""
    try:
        tris = np.asarray(sample.tri_verts)  # type: ignore[attr-defined]
        n_tris = int(tris.reshape(-1, 3).shape[0])
    except Exception:  # noqa: BLE001
        n_tris = 0
    return UVMetrics.empty(n_tris=n_tris, degenerate_tris=n_tris)


# ---------------------------------------------------------------------------
# Input conditioning
# ---------------------------------------------------------------------------


class _Clean:
    """Validated, dtype-normalised views of a sample's arrays.

    Bundling the conditioning in one place keeps :func:`_evaluate_impl` free of
    defensive noise.  Index ranges are checked with two reductions rather than
    a full boolean mask, because on a well-formed mesh -- the overwhelmingly
    common case -- that is four passes over the index arrays instead of eight
    plus an allocation.
    """

    __slots__ = (
        "verts",
        "tri_verts",
        "tri_loops",
        "uvs",
        "loop_verts",
        "edge_verts",
        "edge_seam",
        "n_tris",
        "n_dropped",
    )

    def __init__(self, sample: UVMeshSample) -> None:
        verts = np.asarray(sample.verts, dtype=np.float64)
        self.verts = verts.reshape(-1, 3) if verts.size else np.zeros((0, 3), np.float64)

        tri_verts = np.asarray(sample.tri_verts, dtype=np.int64)
        tri_loops = np.asarray(sample.tri_loops, dtype=np.int64)
        tri_verts = tri_verts.reshape(-1, 3) if tri_verts.size else np.zeros((0, 3), np.int64)
        tri_loops = tri_loops.reshape(-1, 3) if tri_loops.size else np.zeros((0, 3), np.int64)
        # A mismatched pair is corrupt input; keep the common prefix rather than
        # guessing which one is right.
        usable = min(tri_verts.shape[0], tri_loops.shape[0])
        self.n_tris = max(tri_verts.shape[0], tri_loops.shape[0])

        uvs = np.asarray(sample.uvs, dtype=np.float64)
        uvs = uvs.reshape(-1, 2) if uvs.size else np.zeros((0, 2), np.float64)
        self.uvs = uvs

        loop_verts = np.asarray(sample.loop_verts, dtype=np.int64).reshape(-1)
        if loop_verts.shape[0] < uvs.shape[0]:
            loop_verts = np.concatenate(
                [loop_verts, np.zeros(uvs.shape[0] - loop_verts.shape[0], np.int64)]
            )
        self.loop_verts = loop_verts

        n_verts = self.verts.shape[0]
        n_loops = uvs.shape[0]
        tri_verts = tri_verts[:usable]
        tri_loops = tri_loops[:usable]
        if usable:
            in_range = (
                int(tri_verts.min()) >= 0
                and int(tri_verts.max()) < n_verts
                and int(tri_loops.min()) >= 0
                and int(tri_loops.max()) < n_loops
            )
            if not in_range:
                ok = (
                    (tri_verts >= 0).all(axis=1)
                    & (tri_verts < n_verts).all(axis=1)
                    & (tri_loops >= 0).all(axis=1)
                    & (tri_loops < n_loops).all(axis=1)
                )
                tri_verts = tri_verts[ok]
                tri_loops = tri_loops[ok]
        self.tri_verts = tri_verts
        self.tri_loops = tri_loops
        self.n_dropped = self.n_tris - tri_verts.shape[0]

        edge_verts = np.asarray(sample.edge_verts, dtype=np.int64)
        edge_verts = edge_verts.reshape(-1, 2) if edge_verts.size else np.zeros((0, 2), np.int64)
        edge_seam = np.asarray(sample.edge_seam, dtype=bool).reshape(-1)
        n_edges = min(edge_verts.shape[0], edge_seam.shape[0])
        self.edge_verts = edge_verts[:n_edges]
        self.edge_seam = edge_seam[:n_edges]


# ---------------------------------------------------------------------------
# The evaluation itself
# ---------------------------------------------------------------------------


def _evaluate_impl(sample: UVMeshSample, compute_overlap: bool) -> UVMetrics:
    """Do the real work.  See :func:`evaluate` for the contract."""
    data = _Clean(sample)
    n_tris_total = data.n_tris
    if data.tri_verts.shape[0] == 0:
        return UVMetrics.empty(n_tris=n_tris_total, degenerate_tris=n_tris_total)

    # Gather one contiguous array per corner and per component.  A single
    # ``verts[tri_verts]`` gather would move the same bytes, but every
    # downstream op would then read with a 72-byte stride; keeping the corners
    # separate makes the whole pipeline contiguous and is worth several tens of
    # milliseconds at half a million triangles.
    tv = data.tri_verts
    tl = data.tri_loops
    p0 = data.verts[tv[:, 0]]
    p1 = data.verts[tv[:, 1]]
    p2 = data.verts[tv[:, 2]]
    u0 = data.uvs[tl[:, 0], 0]
    v0 = data.uvs[tl[:, 0], 1]
    u1c = data.uvs[tl[:, 1], 0]
    v1c = data.uvs[tl[:, 1], 1]
    u2c = data.uvs[tl[:, 2], 0]
    v2c = data.uvs[tl[:, 2], 1]

    # -- first fundamental forms, 3D and UV ----------------------------------
    d1 = p1 - p0
    d2 = p2 - p0
    e3 = np.einsum("ij,ij->i", d1, d1)
    f3 = np.einsum("ij,ij->i", d1, d2)
    g3 = np.einsum("ij,ij->i", d2, d2)
    del d1, d2, p0, p1, p2

    ds1 = u1c - u0
    dt1 = v1c - v0
    ds2 = u2c - u0
    dt2 = v2c - v0
    e2d = ds1 * ds1 + dt1 * dt1
    f2d = ds1 * ds2 + dt1 * dt2
    g2d = ds2 * ds2 + dt2 * dt2

    area3 = 0.5 * np.sqrt(np.maximum(e3 * g3 - f3 * f3, 0.0))
    signed_uv = 0.5 * (ds1 * dt2 - dt1 * ds2)
    abs_uv = np.abs(signed_uv)

    tri_lo_u = np.minimum(np.minimum(u0, u1c), u2c)
    tri_hi_u = np.maximum(np.maximum(u0, u1c), u2c)
    tri_lo_v = np.minimum(np.minimum(v0, v1c), v2c)
    tri_hi_v = np.maximum(np.maximum(v0, v1c), v2c)

    # A NaN or Inf anywhere in a corner necessarily reaches at least one of the
    # three fundamental-form scalars, so six finiteness tests on (T,) arrays
    # replace a pass over the (T, 3, 3) and (T, 3, 2) coordinate blocks.
    finite3 = np.isfinite(e3) & np.isfinite(f3) & np.isfinite(g3)
    finite2 = np.isfinite(e2d) & np.isfinite(f2d) & np.isfinite(g2d)
    # Same rule as meshdata.degenerate_mask, expressed on the scalars.
    good = finite3 & finite2 & (area3 > _DEGENERATE_EPS) & (abs_uv > _DEGENERATE_EPS)
    n_good = int(good.sum())
    degenerate_tris = int(good.shape[0] - n_good) + data.n_dropped

    total_area3 = float(area3[finite3].sum()) if finite3.any() else 0.0
    flipped_tris = int(np.count_nonzero(good & (signed_uv < 0.0)))

    seam_length, seam_ratio = _seam_metrics(data, total_area3)
    uv_bounds, out_of_bounds = _bounds_metrics(tri_lo_u, tri_lo_v, tri_hi_u, tri_hi_v, finite2)
    if not finite2.all():
        # A single NaN corner would otherwise poison its whole island's bounding
        # box and drop that island from the count; collapse it to a point
        # instead, which contributes nothing to the union.
        tri_lo_u = np.where(finite2, tri_lo_u, 0.0)
        tri_lo_v = np.where(finite2, tri_lo_v, 0.0)
        tri_hi_u = np.where(finite2, tri_hi_u, 0.0)
        tri_hi_v = np.where(finite2, tri_hi_v, 0.0)
    island_count, packing_efficiency = _island_metrics(
        data, abs_uv, good, uv_bounds, tri_lo_u, tri_lo_v, tri_hi_u, tri_hi_v
    )
    uv_coverage = float(abs_uv[good].sum()) if n_good else 0.0

    if n_good == 0:
        return UVMetrics(
            island_count=island_count,
            seam_length=seam_length,
            seam_ratio=seam_ratio,
            packing_efficiency=packing_efficiency,
            uv_coverage=uv_coverage,
            overlap_area=0.0,
            flipped_tris=flipped_tris,
            degenerate_tris=degenerate_tris,
            uv_bounds=uv_bounds,
            out_of_bounds_tris=out_of_bounds,
            n_tris=n_tris_total,
        )

    # Restrict the distortion maths to measurable triangles.  ``good.all()`` is
    # the overwhelmingly common case, and skipping the fancy-index copies there
    # is worth the branch on a 500k-triangle mesh.
    sel = slice(None) if n_good == good.shape[0] else np.flatnonzero(good)
    g_e3, g_f3, g_g3 = e3[sel], f3[sel], g3[sel]
    g_e2, g_f2, g_g2 = e2d[sel], f2d[sel], g2d[sel]
    g_area3 = area3[sel]
    g_abs_uv = abs_uv[sel]
    g_signed = signed_uv[sel]

    stretch_l2, stretch_linf, conformal_error = _stretch_metrics(
        g_e3,
        g_f3,
        g_g3,
        ds1[sel],
        dt1[sel],
        ds2[sel],
        dt2[sel],
        g_area3,
        g_abs_uv,
        g_signed,
    )
    angle_mean, angle_max = _angle_metrics(g_e3, g_f3, g_g3, g_e2, g_f2, g_g2, g_area3)
    area_distortion = _area_distortion(g_area3, g_abs_uv)
    texel_cv = _texel_density_cv(g_area3, g_abs_uv)

    overlap_area = 0.0
    if compute_overlap:
        corners = np.empty((g_area3.shape[0], 3, 2), dtype=np.float64)
        corners[:, 0, 0] = u0[sel]
        corners[:, 0, 1] = v0[sel]
        corners[:, 1, 0] = u1c[sel]
        corners[:, 1, 1] = v1c[sel]
        corners[:, 2, 0] = u2c[sel]
        corners[:, 2, 1] = v2c[sel]
        overlap_area = _overlap_area(
            corners,
            np.stack([tri_lo_u[sel], tri_lo_v[sel]], axis=1),
            np.stack([tri_hi_u[sel], tri_hi_v[sel]], axis=1),
            float(g_abs_uv.sum()),
        )

    return UVMetrics(
        angle_distortion=angle_mean,
        angle_distortion_max=angle_max,
        area_distortion=area_distortion,
        stretch_l2=stretch_l2,
        stretch_linf=stretch_linf,
        conformal_error=conformal_error,
        island_count=island_count,
        seam_length=seam_length,
        seam_ratio=seam_ratio,
        packing_efficiency=packing_efficiency,
        uv_coverage=uv_coverage,
        overlap_area=overlap_area,
        flipped_tris=flipped_tris,
        degenerate_tris=degenerate_tris,
        texel_density_cv=texel_cv,
        uv_bounds=uv_bounds,
        out_of_bounds_tris=out_of_bounds,
        n_tris=n_tris_total,
    )


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------


def _stretch_metrics(
    e3: np.ndarray,
    f3: np.ndarray,
    g3: np.ndarray,
    ds1: np.ndarray,
    dt1: np.ndarray,
    ds2: np.ndarray,
    dt2: np.ndarray,
    area3: np.ndarray,
    abs_uv: np.ndarray,
    signed_uv: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(stretch_l2, stretch_linf, conformal_error)``.

    Implements Sander et al. 2001 section 3 exactly; the module docstring
    carries the derivation of the ``(E, F, G)`` form used here and the argument
    for the area normalisation.  ``conformal_error`` -- the area-weighted mean
    of ``sigma_max/sigma_min - 1`` -- falls out of the same singular values for
    free and is already scale invariant, so it is deliberately *not* normalised.

    The ``1/D`` factor is folded into the UV edge components *before* they are
    squared rather than applied as a ``1/D^2`` factor afterwards.  The two forms
    are algebraically identical, but ``D^2`` overflows to infinity once the UV
    coordinates pass roughly ``1e77`` -- well inside the range where ``e2d`` and
    ``g2d`` are still finite, so the triangle is not rejected as degenerate.
    That drove ``a``, ``b`` and ``c`` to zero and reported a *zero* stretch for
    a layout that was merely scaled up, which is below the provable floor of
    1.0.  Dividing first keeps every intermediate on the order of the stretch
    itself and costs one array pass less than the old form; ``D`` is safe to
    divide by because the caller only passes triangles whose absolute UV area
    clears ``_DEGENERATE_EPS``.
    """
    inv_d = 0.5 / signed_uv  # 1 / D, with D = 2 * signed UV area
    x1 = ds1 * inv_d
    y1 = dt1 * inv_d
    x2 = ds2 * inv_d
    y2 = dt2 * inv_d
    a = y2 * y2 * e3 - 2.0 * y1 * y2 * f3 + y1 * y1 * g3
    c = x2 * x2 * e3 - 2.0 * x1 * x2 * f3 + x1 * x1 * g3
    b = (x1 * y2 + x2 * y1) * f3 - x2 * y2 * e3 - x1 * y1 * g3

    half_sum = 0.5 * (a + c)
    half_diff = 0.5 * (a - c)
    root = np.sqrt(half_diff * half_diff + b * b)
    sig_max_sq = np.maximum(half_sum + root, 0.0)
    sig_min_sq = np.maximum(half_sum - root, 0.0)

    total_uv = float(abs_uv.sum())
    total_3d = float(area3.sum())
    k_norm = math.sqrt(total_uv / total_3d) if total_uv > 0.0 and total_3d > 0.0 else 1.0

    if total_3d > 0.0:
        l2_sq = float(np.dot(half_sum, area3) / total_3d)
    else:
        l2_sq = float(half_sum.mean()) if half_sum.size else 1.0
    stretch_l2 = k_norm * math.sqrt(max(l2_sq, 0.0))
    stretch_linf = k_norm * math.sqrt(float(sig_max_sq.max())) if sig_max_sq.size else 1.0

    ratio = np.sqrt(sig_max_sq / np.maximum(sig_min_sq, _TINY))
    if total_3d > 0.0:
        conformal = float(np.dot(np.where(np.isfinite(ratio), ratio, 0.0), area3) / total_3d)
        conformal -= 1.0
    else:
        conformal = 0.0

    # Both bounds below are theorems about the quantity being measured, not
    # taste, so enforcing them can only ever remove floating-point noise.
    # Per triangle AM-GM gives L2(T)^2 = (Gamma^2 + gamma^2)/2 >= Gamma*gamma =
    # A3d/Auv; summing area-weighted and applying Cauchy-Schwarz gives
    # sum(A3d^2/Auv) >= (sum A3d)^2 / sum Auv = sum A3d, hence L2 >= 1 -- which
    # holds precisely because ``total_uv`` and ``total_3d`` here are summed over
    # exactly the triangles that feed ``l2_sq``.  Computing the normalisation
    # over a wider set than the accumulation is the classic way to break this.
    # Linf is a maximum over the same per-triangle values whose area-weighted
    # RMS is L2, so Linf >= L2 likewise.
    l2_out = max(_finite(stretch_l2, 1.0), 1.0)
    return (
        l2_out,
        max(_finite(stretch_linf, 1.0), l2_out),
        max(0.0, _finite(conformal, 0.0)),
    )


def _triangle_angles(
    e: np.ndarray, f: np.ndarray, g: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three interior angles (radians) of triangles given ``E,F,G``.

    Squared side lengths are ``E`` (corner 0 to 1), ``G`` (corner 0 to 2) and
    ``E - 2F + G`` (corner 1 to 2), so the cosine rule reduces to two
    normalised dot products.  The third angle is ``pi`` minus the other two --
    exact for a planar triangle, and one fewer ``arccos`` per triangle per
    space, which at 500k triangles is worth having.
    """
    side12 = np.maximum(e - 2.0 * f + g, _TINY)
    cos0 = f / np.sqrt(np.maximum(e * g, _TINY))
    cos1 = (e - f) / np.sqrt(np.maximum(e * side12, _TINY))
    ang0 = np.arccos(np.clip(cos0, -1.0, 1.0))
    ang1 = np.arccos(np.clip(cos1, -1.0, 1.0))
    return ang0, ang1, math.pi - ang0 - ang1


def _angle_metrics(
    e3: np.ndarray,
    f3: np.ndarray,
    g3: np.ndarray,
    e2d: np.ndarray,
    f2d: np.ndarray,
    g2d: np.ndarray,
    area3: np.ndarray,
) -> tuple[float, float]:
    """Return ``(area-weighted mean, max)`` corner-angle deviation in degrees."""
    a0, a1, a2 = _triangle_angles(e3, f3, g3)
    b0, b1, b2 = _triangle_angles(e2d, f2d, g2d)
    d0 = np.abs(a0 - b0)
    d1 = np.abs(a1 - b1)
    d2 = np.abs(a2 - b2)
    per_tri = (d0 + d1 + d2) * (_RAD_TO_DEG / 3.0)
    worst = (
        max(
            float(d0.max()) if d0.size else 0.0,
            float(d1.max()) if d1.size else 0.0,
            float(d2.max()) if d2.size else 0.0,
        )
        * _RAD_TO_DEG
    )

    total = float(area3.sum())
    if total > 0.0:
        mean = float(np.dot(per_tri, area3) / total)
    else:
        mean = float(per_tri.mean()) if per_tri.size else 0.0
    return _finite(mean, 0.0), _finite(worst, 0.0)


def _area_distortion(area3: np.ndarray, abs_uv: np.ndarray) -> float:
    """Return the mean ``|log2(uv share / surface share)|`` over triangles.

    Each triangle's area is expressed as a *share* of its own domain's total,
    which makes the metric independent of both the model scale and the UV scale;
    only the relative distribution of texture across the surface matters.  The
    architecture specifies a plain mean here (unlike the angle metric, which is
    explicitly area-weighted), so that is what this computes.
    """
    total_uv = float(abs_uv.sum())
    total_3d = float(area3.sum())
    if total_uv <= 0.0 or total_3d <= 0.0 or area3.size == 0:
        return 0.0
    ratio = (abs_uv * (total_3d / total_uv)) / area3
    ratio = np.where(ratio > 0.0, ratio, 1.0)
    return _finite(float(np.abs(np.log2(ratio)).mean()), 0.0)


def _texel_density_cv(area3: np.ndarray, abs_uv: np.ndarray) -> float:
    """Return the area-weighted coefficient of variation of texel density.

    Texel density is ``sqrt(uv_area / surface_area)`` -- texels per world unit.
    The coefficient of variation (weighted sigma over weighted mean) is the
    scale-free way to say "how evenly is texture resolution spread", which is
    exactly the question a game artist asks of an atlas.
    """
    if area3.size == 0:
        return 0.0
    density = np.sqrt(abs_uv / area3)
    wsum = float(area3.sum())
    if wsum <= 0.0:
        return 0.0
    mean = float(np.dot(density, area3) / wsum)
    if not math.isfinite(mean) or mean <= 0.0:
        return 0.0
    delta = density - mean
    var = float(np.dot(delta * delta, area3) / wsum)
    return _finite(math.sqrt(max(var, 0.0)) / mean, 0.0)


# ---------------------------------------------------------------------------
# Seams and bounds
# ---------------------------------------------------------------------------


def _seam_metrics(data: _Clean, total_area3: float) -> tuple[float, float]:
    """Return ``(seam_length, seam_ratio)``.

    ``seam_ratio`` divides by ``sqrt(total surface area)`` because seam length
    is a 1D quantity and area is 2D: the ratio is then dimensionless, so the
    same model imported in centimetres and in metres scores identically.
    """
    ev = data.edge_verts
    if ev.shape[0] == 0 or data.verts.shape[0] == 0:
        return 0.0, 0.0
    seam = data.edge_seam
    if not seam.any():
        return 0.0, 0.0
    n_verts = data.verts.shape[0]
    ok = (ev >= 0).all(axis=1) & (ev < n_verts).all(axis=1) & seam
    if not ok.any():
        return 0.0, 0.0
    picked = ev[ok]
    delta = data.verts[picked[:, 1]] - data.verts[picked[:, 0]]
    lengths = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    length = float(lengths[np.isfinite(lengths)].sum())
    ratio = length / math.sqrt(total_area3) if total_area3 > 0.0 else 0.0
    return _finite(length, 0.0), _finite(ratio, 0.0)


def _bounds_metrics(
    lo_u: np.ndarray,
    lo_v: np.ndarray,
    hi_u: np.ndarray,
    hi_v: np.ndarray,
    finite2: np.ndarray,
) -> tuple[tuple[float, float, float, float], int]:
    """Return ``(uv_bounds, out_of_bounds_tris)`` from per-triangle UV extents.

    The extents are computed once by the caller and reused here, by the island
    pass and by overlap detection.  Recomputing them with ``uv.min(axis=1)`` on
    the ``(T, 3, 2)`` block instead would cost more than every other metric in
    this module combined -- reducing over the middle axis of a three-dimensional
    array defeats numpy's contiguous fast paths.
    """
    if lo_u.shape[0] == 0 or not finite2.any():
        return (0.0, 0.0, 0.0, 0.0), 0
    if finite2.all():
        s_lo_u, s_lo_v, s_hi_u, s_hi_v = lo_u, lo_v, hi_u, hi_v
    else:
        s_lo_u, s_lo_v = lo_u[finite2], lo_v[finite2]
        s_hi_u, s_hi_v = hi_u[finite2], hi_v[finite2]
    bounds = (
        float(s_lo_u.min()),
        float(s_lo_v.min()),
        float(s_hi_u.max()),
        float(s_hi_v.max()),
    )
    oob = (
        (s_lo_u < -_OOB_TOL)
        | (s_lo_v < -_OOB_TOL)
        | (s_hi_u > 1.0 + _OOB_TOL)
        | (s_hi_v > 1.0 + _OOB_TOL)
    )
    return bounds, int(np.count_nonzero(oob))


# ---------------------------------------------------------------------------
# Islands and packing
# ---------------------------------------------------------------------------


def _island_metrics(
    data: _Clean,
    abs_uv: np.ndarray,
    good: np.ndarray,
    uv_bounds: tuple[float, float, float, float],
    tri_lo_u: np.ndarray,
    tri_lo_v: np.ndarray,
    tri_hi_u: np.ndarray,
    tri_hi_v: np.ndarray,
) -> tuple[int, float]:
    """Return ``(island_count, packing_efficiency)``.

    Islands are the connected components of the UV graph: two face corners join
    when they belong to the same triangle, or when they share a mesh vertex
    *and* land on the same UV.  Both conditions are required -- welding on the
    vertex alone would merge deliberately split islands, and welding on the UV
    alone would fuse islands that merely touch in texture space.

    Island bounding boxes are accumulated from *triangle* extents rather than
    from individual UV vertices.  All three corners of a triangle are in the
    same island by construction, so the answer is identical, and it saves
    scattering a UV per node.  Component labels are used directly as the
    accumulator index (labels are node ids, so they are already dense enough),
    which avoids a sort purely to renumber them.
    """
    nodes = _uv_vertex_nodes(data, uv_bounds)
    if nodes is None:
        return 0, 0.0
    node_of_loop, n_nodes, extra = nodes

    corners = node_of_loop[data.tri_loops]  # (T, 3)
    ea = corners.reshape(-1)
    eb = corners[:, [1, 2, 0]].reshape(-1)
    if extra is not None:
        ea = np.concatenate([ea, extra[0]])
        eb = np.concatenate([eb, extra[1]])

    labels = _connected_components(n_nodes, ea, eb)
    tri_label = labels[corners[:, 0]]
    if tri_label.size == 0:
        return 0, 0.0

    lo_u = np.full(n_nodes, np.inf, dtype=np.float64)
    lo_v = np.full(n_nodes, np.inf, dtype=np.float64)
    hi_u = np.full(n_nodes, -np.inf, dtype=np.float64)
    hi_v = np.full(n_nodes, -np.inf, dtype=np.float64)
    np.minimum.at(lo_u, tri_label, tri_lo_u)
    np.minimum.at(lo_v, tri_label, tri_lo_v)
    np.maximum.at(hi_u, tri_label, tri_hi_u)
    np.maximum.at(hi_v, tri_label, tri_hi_v)

    live = np.isfinite(lo_u)
    island_count = int(np.count_nonzero(live))
    if island_count == 0:
        return 0, 0.0
    lo = np.stack([lo_u[live], lo_v[live]], axis=1)
    hi = np.stack([hi_u[live], hi_v[live]], axis=1)

    union_area = _union_rect_area(lo, hi)
    used_area = float(abs_uv[good].sum()) if good.any() else 0.0
    # Clamped at 1.0: anything above it is overlap, which is reported separately
    # rather than being allowed to masquerade as super-efficient packing.
    measurable = union_area > 0.0 and used_area > 0.0
    efficiency = min(1.0, used_area / union_area) if measurable else 0.0
    return island_count, _finite(efficiency, 0.0)


def _uv_vertex_nodes(data: _Clean, uv_bounds: tuple[float, float, float, float]):
    """Group loops into UV vertices.

    Returns ``(node_of_loop, n_nodes, extra_edges)``, or ``None`` when there is
    nothing to group.  ``node_of_loop`` maps every loop index to a node id and
    ``extra_edges`` holds merge hints for pairs the lattice would otherwise
    split.

    The grouping exploits the fact that on a real mesh nearly every vertex has
    exactly *one* UV.  One scatter records a representative loop per vertex;
    every loop whose quantised UV key matches its vertex's representative --
    the whole mesh except the seam boundaries -- is that vertex's node with no
    further work.  Only the leftovers go through a sort.  On a fully welded
    mesh the sort disappears entirely, which is what makes island detection
    affordable at half a million triangles.

    Two safeguards cover the one weakness of quantisation, namely a pair of UVs
    within tolerance that happens to straddle a cell boundary.  First, a
    leftover loop is re-admitted to its vertex's node when its UV is within the
    tolerance of the representative's *exact* UV -- no lattice involved.
    Second, the survivors are additionally grouped on a half-cell-offset
    lattice, and the two groupings are merged; two lattices offset by half a
    cell cannot both split the same near-coincident pair.  Both safeguards work
    on subsets that are empty for a well-formed mesh, so they are free in the
    common case.

    Loops not referenced by any triangle are excluded; otherwise a stray loop
    left behind by a modifier would count as its own island.
    """
    n_loops = data.uvs.shape[0]
    if data.tri_loops.size == 0 or n_loops == 0:
        return None

    corner_loops = data.tri_loops.reshape(-1)
    loop_hits = np.bincount(corner_loops, minlength=n_loops)
    if loop_hits.min() > 0:
        used_idx = None
        uv = data.uvs
        vert = data.loop_verts[:n_loops]
    else:
        used_idx = np.flatnonzero(loop_hits)
        if used_idx.size == 0:
            return None
        uv = data.uvs[used_idx]
        vert = data.loop_verts[used_idx]

    lo_u, lo_v, hi_u, hi_v = uv_bounds
    if not all(math.isfinite(x) for x in uv_bounds):
        lo_u = lo_v = 0.0
        hi_u = hi_v = 0.0
    if not np.isfinite(uv).all():
        uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
        lo_u = min(lo_u, float(uv[:, 0].min()))
        lo_v = min(lo_v, float(uv[:, 1].min()))
        hi_u = max(hi_u, float(uv[:, 0].max()))
        hi_v = max(hi_v, float(uv[:, 1].max()))

    n_verts = max(data.verts.shape[0], 1)
    if vert.size and (int(vert.min()) < 0 or int(vert.max()) >= n_verts):
        vert = np.clip(vert, 0, n_verts - 1)
    n_slots = (int(vert.max()) + 1) if vert.size else 1

    # Keep vert * cells^2 inside int64.  1<<20 cells per axis is plenty: with a
    # 1e-6 relative tolerance nothing finer is meaningful.
    budget = (1 << 62) // max(n_slots, 1)
    cells = max(4, min(int(math.isqrt(max(budget, 16))), 1 << 20))
    span = float(max(hi_u - lo_u, hi_v - lo_v, 0.0))
    step = _WELD_EPS * max(span, 1.0)
    if not (step > 0.0) or not math.isfinite(step):
        step = 1.0
    if span / step > cells - 2:
        step = span / (cells - 2)
    # The lattice index is computed as ``uv/step - lo/step``, so ``lo/step`` has
    # to stay inside the range where float64 still resolves integers.  UVs out
    # at 1e15 would otherwise quantise to noise; coarsening the tolerance
    # degrades gracefully instead.
    far = max(abs(lo_u), abs(hi_u), abs(lo_v), abs(hi_v), 0.0)
    if far > 0.0 and far / step > 4.0e15:
        step = far / 4.0e15
    inv_step = 1.0 / step

    # rel is non-negative and bounded by cells-2, so truncation after +0.5 is a
    # correct round-to-nearest and needs no clipping.
    qu = (uv[:, 0] * inv_step + (0.5 - lo_u * inv_step)).astype(np.int64)
    qv = (uv[:, 1] * inv_step + (0.5 - lo_v * inv_step)).astype(np.int64)
    uv_key = qv * cells + qu

    rep_loop = np.zeros(n_slots, dtype=np.int64)
    rep_loop[vert] = np.arange(vert.shape[0], dtype=np.int64)
    ref_of = rep_loop[vert]
    odd = np.flatnonzero(uv_key != uv_key[ref_of])

    n_nodes = n_slots
    if odd.size == 0:
        node_of_used = vert
    else:
        node_of_used = np.array(vert, dtype=np.int64)
        # Re-admit leftovers that are genuinely within tolerance of their
        # vertex's representative: those only fell out because the lattice
        # boundary happened to run between them.
        ref = ref_of[odd]
        near = (np.abs(uv[odd, 0] - uv[ref, 0]) <= 0.5 * step) & (
            np.abs(uv[odd, 1] - uv[ref, 1]) <= 0.5 * step
        )
        rest = odd[~near]
        if rest.size:
            full_key = vert[rest] * (cells * cells) + uv_key[rest]
            uniq, inverse = np.unique(full_key, return_inverse=True)
            node_of_used[rest] = n_slots + inverse.reshape(-1)
            n_nodes = n_slots + int(uniq.size)

    if used_idx is None:
        node_of_loop = node_of_used
    else:
        node_of_loop = np.zeros(n_loops, dtype=np.int64)
        node_of_loop[used_idx] = node_of_used

    extra = None
    if _WELD_DUAL_LATTICE and odd.size and n_nodes > n_slots:
        rest = odd[node_of_used[odd] >= n_slots]
        if rest.size > 1:
            alt_key = (
                vert[rest] * (cells * cells)
                + (uv[rest, 1] * inv_step + (1.0 - lo_v * inv_step)).astype(np.int64) * cells
                + (uv[rest, 0] * inv_step + (1.0 - lo_u * inv_step)).astype(np.int64)
            )
            alt_uniq, alt_inv = np.unique(alt_key, return_inverse=True)
            if alt_uniq.size != rest.size:
                alt_inv = alt_inv.reshape(-1)
                reverse = np.arange(rest.size - 1, -1, -1, dtype=np.int64)
                first = np.zeros(alt_uniq.size, dtype=np.int64)
                first[alt_inv[reverse]] = reverse
                mine = node_of_used[rest]
                partner = mine[first[alt_inv]]
                differs = partner != mine
                if differs.any():
                    extra = (mine[differs], partner[differs])
    return node_of_loop, n_nodes, extra


def _connected_components(n_nodes: int, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Label the connected components of an undirected graph, vectorised.

    This is the Shiloach-Vishkin shape of union-find: hook every node to the
    smallest label in its neighbourhood, fully path-compress by pointer
    doubling, then *contract* the edge list by dropping edges whose endpoints
    now agree and de-duplicating the rest.  The contraction is what makes it
    fast in practice -- the first round removes the overwhelming majority of a
    mesh's edges, so the O(log n) rounds the algorithm guarantees cost
    geometrically less each time.

    A pure-Python union-find would be simpler but needs one interpreted step per
    edge, which is roughly two orders of magnitude too slow at 1.5M edges.

    Args:
        n_nodes: Number of nodes; node ids must lie in ``[0, n_nodes)``.
        ea: ``(E,)`` first endpoint of each edge.
        eb: ``(E,)`` second endpoint of each edge.

    Returns:
        ``(n_nodes,)`` int64 labels; two nodes share a label iff they are
        connected.  The label is the smallest node id in the component.
    """
    labels = np.arange(n_nodes, dtype=np.int64)
    if ea.size == 0 or n_nodes == 0:
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

        # Pointer doubling.  labels[i] <= i always holds, so the pointer graph
        # is acyclic apart from self-loops and this terminates at the roots.
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


def _union_rect_area(lo: np.ndarray, hi: np.ndarray) -> float:
    """Return the area of the union of axis-aligned rectangles.

    Summing the individual rectangle areas would double-count wherever island
    bounding boxes overlap -- which is exactly what a good packer arranges --
    and would make ``packing_efficiency`` read far too low on a tight layout.
    The union is computed by rasterising the rectangles onto the lattice formed
    by their own coordinates (so the result is *exact*, not sampled) using a 2D
    difference array plus a double prefix sum.  Above ``_RECT_GRID_CELLS`` the
    exact lattice is replaced by a uniform one, which slightly over-estimates
    the union and therefore under-states efficiency -- the conservative
    direction.
    """
    if lo.shape[0] == 0:
        return 0.0
    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    if not finite.all():
        if not finite.any():
            return 0.0
        lo = lo[finite]
        hi = hi[finite]
    hi = np.maximum(hi, lo)
    if lo.shape[0] == 1:
        return float((hi[0, 0] - lo[0, 0]) * (hi[0, 1] - lo[0, 1]))

    xs = np.unique(np.concatenate([lo[:, 0], hi[:, 0]]))
    ys = np.unique(np.concatenate([lo[:, 1], hi[:, 1]]))
    nx = xs.size - 1
    ny = ys.size - 1
    if nx <= 0 or ny <= 0:
        return 0.0

    if nx * ny > _RECT_GRID_CELLS:
        side = max(2, int(math.isqrt(_RECT_GRID_CELLS)))
        xs = np.linspace(xs[0], xs[-1], side + 1)
        ys = np.linspace(ys[0], ys[-1], side + 1)
        nx = side
        ny = side
        i0 = np.clip(np.searchsorted(xs, lo[:, 0], side="right") - 1, 0, nx)
        j0 = np.clip(np.searchsorted(ys, lo[:, 1], side="right") - 1, 0, ny)
        i1 = np.clip(np.searchsorted(xs, hi[:, 0], side="left"), 0, nx)
        j1 = np.clip(np.searchsorted(ys, hi[:, 1], side="left"), 0, ny)
        i1 = np.maximum(i1, np.minimum(i0 + 1, nx))
        j1 = np.maximum(j1, np.minimum(j0 + 1, ny))
    else:
        i0 = np.searchsorted(xs, lo[:, 0], side="left")
        j0 = np.searchsorted(ys, lo[:, 1], side="left")
        i1 = np.searchsorted(xs, hi[:, 0], side="left")
        j1 = np.searchsorted(ys, hi[:, 1], side="left")

    span = (i1 > i0) & (j1 > j0)
    if not span.any():
        return 0.0
    i0 = i0[span]
    i1 = i1[span]
    j0 = j0[span]
    j1 = j1[span]

    stride = ny + 1
    flat = np.concatenate([i0 * stride + j0, i1 * stride + j1, i0 * stride + j1, i1 * stride + j0])
    n_rect = i0.size
    weights = np.empty(4 * n_rect, dtype=np.float64)
    weights[: 2 * n_rect] = 1.0
    weights[2 * n_rect :] = -1.0
    diff = np.bincount(flat, weights=weights, minlength=(nx + 1) * stride)
    diff = diff.reshape(nx + 1, stride)
    covered = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:nx, :ny] > 0.5

    dx = np.diff(xs)
    dy = np.diff(ys)
    total = 0.0
    for start in range(0, nx, 2048):
        stop = min(start + 2048, nx)
        block = covered[start:stop].astype(np.float64)
        total += float(dx[start:stop] @ (block @ dy))
    return _finite(total, 0.0)


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


def _overlap_area(uv: np.ndarray, lo: np.ndarray, hi: np.ndarray, total_uv_area: float) -> float:
    """Return the total pairwise overlapping UV area.

    Args:
        uv: ``(T, 3, 2)`` UV triangles to test.
        lo: ``(T, 2)`` per-triangle minimum UV corner.
        hi: ``(T, 2)`` per-triangle maximum UV corner.
        total_uv_area: Summed absolute UV area, used to clamp the result and to
            scale the sliver-rejection epsilon.

    Candidate pairs come from a uniform spatial hash sized to the median
    triangle, so the cost is linear in the number of triangles times the local
    density -- never the O(T^2) that a naive all-pairs test would cost.
    Candidates are then filtered by bounding box and by the separating-axis
    test before anything expensive happens; on a clean layout essentially every
    candidate dies there.

    Whatever survives is resolved *exactly*: the intersection of two convex
    polygons is the convex hull of (a) every edge-edge crossing, (b) the corners
    of A inside B and (c) the corners of B inside A, so the true intersection
    area falls out of a shoelace over those points sorted by angle.  Containment
    needs no special case -- when A is wholly inside B all three of A's corners
    are collected and nothing else is.

    Triangles that merely share an edge are separated by that edge (the test
    treats touching as separated) and therefore contribute nothing, which is why
    no adjacency pre-filter is needed; a genuinely folded-over neighbour, on the
    other hand, does contribute, which is the behaviour an artist wants.

    The returned figure is the sum over pairs, so a spot where three triangles
    stack counts each of the three pairs.  That is the useful reading -- it
    scales with how badly the layout is broken -- and it is clamped to the total
    used UV area so the number stays interpretable.
    """
    n_tris = uv.shape[0]
    if n_tris < 2:
        return 0.0
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return 0.0

    grid_min = lo.min(axis=0)
    extent = float(max(hi[:, 0].max() - grid_min[0], hi[:, 1].max() - grid_min[1]))
    if extent <= 0.0:
        return 0.0

    cell = float(np.median((hi - lo).max(axis=1)))
    if not (cell > 0.0):
        cell = extent / 512.0
    cell = max(cell, extent / _OVERLAP_MAX_GRID_DIM, 1e-12)

    for _ in range(8):
        i0 = np.floor((lo[:, 0] - grid_min[0]) / cell).astype(np.int64)
        i1 = np.floor((hi[:, 0] - grid_min[0]) / cell).astype(np.int64)
        j0 = np.floor((lo[:, 1] - grid_min[1]) / cell).astype(np.int64)
        j1 = np.floor((hi[:, 1] - grid_min[1]) / cell).astype(np.int64)
        counts = (i1 - i0 + 1) * (j1 - j0 + 1)
        total = int(counts.sum())
        if total <= _OVERLAP_MAX_INSERTS:
            break
        cell *= 2.0
    else:
        return 0.0
    if total <= 1:
        return 0.0

    grid_w = int(i1.max()) + 2
    starts = np.zeros(n_tris, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    tri_of = np.repeat(np.arange(n_tris, dtype=np.int64), counts)
    offset = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    width = np.repeat(i1 - i0 + 1, counts)
    cell_id = (np.repeat(j0, counts) + offset // width) * grid_w + (
        np.repeat(i0, counts) + offset % width
    )

    pair_a, pair_b = _cell_pairs(cell_id, tri_of)
    if pair_a.size == 0:
        return 0.0
    packed = np.unique(pair_a * n_tris + pair_b)
    pair_a = packed // n_tris
    pair_b = packed - pair_a * n_tris

    # Cheap bounding-box rejection before any geometry.  The comparison is
    # strict: boxes that merely touch along a line bound triangles that can at
    # most touch, and touching contributes no area.
    keep = (
        (lo[pair_a, 0] < hi[pair_b, 0])
        & (lo[pair_b, 0] < hi[pair_a, 0])
        & (lo[pair_a, 1] < hi[pair_b, 1])
        & (lo[pair_b, 1] < hi[pair_a, 1])
    )
    pair_a = pair_a[keep]
    pair_b = pair_b[keep]
    if pair_a.size == 0:
        return 0.0

    oriented = _orient_ccw(uv)
    tri_scale = np.maximum(hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1])
    area_eps = 1e-12 * max(total_uv_area, _TINY)
    accumulated = 0.0
    for start in range(0, pair_a.size, _OVERLAP_CHUNK):
        stop = min(start + _OVERLAP_CHUNK, pair_a.size)
        sub_a = pair_a[start:stop]
        sub_b = pair_b[start:stop]
        areas = _pair_intersection_area(
            oriented[sub_a],
            oriented[sub_b],
            scale=np.maximum(tri_scale[sub_a], tri_scale[sub_b]),
        )
        accumulated += float(areas[areas > area_eps].sum())
        # The answer is clamped to the used area, so once it saturates no
        # further pair can change it.  This is what keeps a catastrophic layout
        # (every triangle stacked on one spot) from costing seconds.
        if total_uv_area > 0.0 and accumulated >= total_uv_area:
            return total_uv_area

    if total_uv_area > 0.0:
        accumulated = min(accumulated, total_uv_area)
    return _finite(accumulated, 0.0)


def _cell_pairs(cell_id: np.ndarray, tri_of: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(a, b)`` triangle pairs that share at least one grid cell.

    Pairs are generated by sorting the (cell, triangle) insertions and then
    walking a widening stride within each cell run, which is exact and needs no
    per-cell Python loop.  Generation stops once ``_OVERLAP_MAX_PAIRS`` is
    reached: a layout dense enough to hit that is already catastrophically
    broken, and the reported overlap saturates anyway.
    """
    order = np.argsort(cell_id)
    sorted_cells = cell_id[order]
    sorted_tris = tri_of[order]

    total = sorted_cells.size
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
        first = sorted_tris[picked]
        second = sorted_tris[picked + stride]
        keep = first != second
        if keep.any():
            chunks_a.append(np.minimum(first[keep], second[keep]))
            chunks_b.append(np.maximum(first[keep], second[keep]))
            produced += int(keep.sum())
        if produced >= _OVERLAP_MAX_PAIRS:
            break
    if not chunks_a:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(chunks_a), np.concatenate(chunks_b)


def _orient_ccw(uv: np.ndarray) -> np.ndarray:
    """Return a copy of ``uv`` with every triangle wound counter-clockwise.

    The inside/outside tests below assume a consistent winding.  Flipping a
    mirrored triangle changes nothing about the region it covers, so this is
    safe for an area query even though the flip itself is a defect reported
    separately as ``flipped_tris``.
    """
    out = np.array(uv, dtype=np.float64, copy=True)
    e1 = out[:, 1] - out[:, 0]
    e2 = out[:, 2] - out[:, 0]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flip = signed < 0.0
    if flip.any():
        out[flip] = out[flip][:, [0, 2, 1], :]
    return out


def _pair_intersection_area(
    tri_a: np.ndarray, tri_b: np.ndarray, scale: np.ndarray | None = None
) -> np.ndarray:
    """Return the exact intersection area of matched CCW triangle pairs.

    Args:
        tri_a: ``(N, 3, 2)`` counter-clockwise triangles.
        tri_b: ``(N, 3, 2)`` counter-clockwise triangles.
        scale: Optional ``(N,)`` characteristic length per pair, used to set the
            relative tolerance.  Callers that already know each triangle's
            extent pass it in to avoid a reduction over the ``(N, 3, 2)`` block.

    Returns:
        ``(N,)`` intersection areas, ``0.0`` where the pair does not overlap.
    """
    n = tri_a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    da = tri_a[:, [1, 2, 0], :] - tri_a
    db = tri_b[:, [1, 2, 0], :] - tri_b
    if scale is None:
        scale = np.maximum(np.abs(da).max(axis=(1, 2)), np.abs(db).max(axis=(1, 2)))
    scale = np.where(scale > 0.0, scale, 1.0)
    tol = 1e-12 * scale * scale

    live = _sat_survivors(tri_a, tri_b, da, db, tol)
    if live.size == 0:
        return np.zeros(n, dtype=np.float64)
    if live.size == n:
        return _clip_area(tri_a, tri_b, da, db, tol)
    out = np.zeros(n, dtype=np.float64)
    out[live] = _clip_area(tri_a[live], tri_b[live], da[live], db[live], tol[live])
    return out


def _sat_survivors(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    da: np.ndarray,
    db: np.ndarray,
    tol: np.ndarray,
) -> np.ndarray:
    """Return indices of pairs that no edge of either triangle separates.

    Two convex polygons are disjoint if and only if some edge of one of them is
    a separating axis, so six half-plane tests settle the question -- far
    cheaper than constructing the intersection polygon.  Touching counts as
    separated (the comparison is ``<= +tol``), which is what makes two triangles
    sharing an edge cost almost nothing and report exactly zero overlap.

    Each axis is applied to the survivors of the previous one, so on a clean
    layout -- where nearly every candidate pair is disjoint -- the working set
    collapses geometrically and the later axes are nearly free.
    """
    n = tri_a.shape[0]
    idx: np.ndarray | None = None
    for tri, other, edge in ((tri_a, tri_b, da), (tri_b, tri_a, db)):
        for corner in range(3):
            if idx is None:
                ox = tri[:, corner, 0]
                oy = tri[:, corner, 1]
                ex = edge[:, corner, 0]
                ey = edge[:, corner, 1]
                far = other
                limit = tol
            else:
                if idx.size == 0:
                    return idx
                ox = tri[idx, corner, 0]
                oy = tri[idx, corner, 1]
                ex = edge[idx, corner, 0]
                ey = edge[idx, corner, 1]
                far = other[idx]
                limit = tol[idx]
            separated = ex * (far[:, 0, 1] - oy) - ey * (far[:, 0, 0] - ox) <= limit
            separated &= ex * (far[:, 1, 1] - oy) - ey * (far[:, 1, 0] - ox) <= limit
            separated &= ex * (far[:, 2, 1] - oy) - ey * (far[:, 2, 0] - ox) <= limit
            if not separated.any():
                continue
            keep = ~separated
            idx = np.flatnonzero(keep) if idx is None else idx[keep]
    return np.arange(n, dtype=np.int64) if idx is None else idx


def _edge_cross(pts: np.ndarray, origin: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Return ``(N, 3 points, 3 edges)`` cross products ``edge x (pt - origin)``.

    Positive means the point is to the left of the directed edge, i.e. inside a
    counter-clockwise triangle.  Written as four ``(N, 3, 3)`` broadcasts rather
    than one ``(N, 3, 3, 2)`` difference, which halves the peak footprint.
    """
    ex = edge[:, None, :, 0]
    ey = edge[:, None, :, 1]
    ox = origin[:, None, :, 0]
    oy = origin[:, None, :, 1]
    return ex * (pts[:, :, None, 1] - oy) - ey * (pts[:, :, None, 0] - ox)


def _clip_area(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    da: np.ndarray,
    db: np.ndarray,
    tol: np.ndarray,
) -> np.ndarray:
    """Return the intersection area of pairs already known to be non-separated."""
    n = tri_a.shape[0]
    tol = tol[:, None]
    a_in_b = (_edge_cross(tri_a, tri_b, db) >= -tol[:, :, None]).all(axis=2)
    b_in_a = (_edge_cross(tri_b, tri_a, da) >= -tol[:, :, None]).all(axis=2)
    ea = da[:, :, None, :]
    eb = db[:, None, :, :]
    rel = tri_b[:, None, :, :] - tri_a[:, :, None, :]
    denom = ea[..., 0] * eb[..., 1] - ea[..., 1] * eb[..., 0]
    usable = np.abs(denom) > tol[:, :, None] * 1e-2
    safe = np.where(usable, denom, 1.0)
    t = (rel[..., 0] * eb[..., 1] - rel[..., 1] * eb[..., 0]) / safe
    u = (rel[..., 0] * ea[..., 1] - rel[..., 1] * ea[..., 0]) / safe
    crossing = usable & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    points = tri_a[:, :, None, :] + t[..., None] * ea

    candidates = np.concatenate([points.reshape(n, 9, 2), tri_a, tri_b], axis=1)
    valid = np.concatenate([crossing.reshape(n, 9), a_in_b, b_in_a], axis=1)
    valid &= np.isfinite(candidates).all(axis=2)
    candidates = np.where(valid[:, :, None], candidates, 0.0)

    count = valid.sum(axis=1)
    centroid = candidates.sum(axis=1) / np.maximum(count, 1)[:, None]
    offset = candidates - centroid[:, None, :]
    angle = np.where(valid, np.arctan2(offset[..., 1], offset[..., 0]), np.inf)

    order = np.argsort(angle, axis=1)
    ordered = np.take_along_axis(candidates, order[:, :, None], axis=1)
    ordered_valid = np.take_along_axis(valid, order, axis=1)
    # Collapse the unused slots onto the first real vertex: repeated vertices
    # contribute nothing to the shoelace sum and close the polygon for free.
    ordered = np.where(ordered_valid[:, :, None], ordered, ordered[:, 0:1, :])

    nxt = np.roll(ordered, -1, axis=1)
    twice = (ordered[..., 0] * nxt[..., 1] - ordered[..., 1] * nxt[..., 0]).sum(axis=1)
    area = 0.5 * np.abs(twice)
    return np.where((count >= 3) & np.isfinite(area), area, 0.0)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value: float, fallback: float) -> float:
    """Return ``value`` as a float, substituting ``fallback`` if non-finite."""
    number = float(value)
    return number if math.isfinite(number) else fallback
