#!/usr/bin/env python3
"""Visual regression for the organic corpus -- docs/ORGANIC.md section 8.3.

Numbers are necessary and they are not sufficient.  A layout can score well and
still be one nobody would ship, and the clearest evidence for that in this
project's own history is the 276-island Suzanne recorded in
``docs/BENCHMARKS.md`` section 4.1: excellent distortion, and confetti.  No
metric in the suite caught it.  Somebody looking at it would have, in a second.

So this script renders what the harness measures, for every corpus mesh and
every variant:

* **The UV layout**, software-rasterised: islands filled with distinct colours,
  island boundaries drawn, seam-flagged edges drawn, and every triangle
  involved in a real overlap outlined in **red**.  Rasterised in numpy and
  written through ``bpy.data.images`` rather than drawn with the ``gpu``
  module, because ``docs/ARCHITECTURE.md`` section 0 records that GPU drawing
  raises ``SystemError`` in background mode.  A software rasteriser also has
  the advantage of being exactly reproducible.
* **The mesh itself**, textured with Blender's own UV grid, from three fixed
  camera angles.  Stretch shows up as squashed squares, seams as breaks in the
  grid, and a fold as the grid running over itself.
* **An index page** tying the pictures to the numbers that produced them, so a
  critic can go from "that looks wrong" to "and here is its overlap area".

Determinism: fixed seed, cameras placed analytically from the mesh bounds,
fixed resolution, fixed sample count, EEVEE.  Two caveats, both measured rather
than assumed.  The engine identifier differs between the supported builds
(``BLENDER_EEVEE_NEXT`` on 4.5.4, ``BLENDER_EEVEE`` on 5.1.2) and is resolved by
probing the enum, never by branching on a version number.  And the *layouts*
being rendered come from Blender's non-deterministic UV operators, so two runs
produce visibly similar but not identical pictures -- which is a property of
the subject, not of the camera.

Run::

    blender -b --factory-startup --python tools/render_uv_report.py -- --out renders/

Exit code is 0 when every requested mesh rendered, 1 otherwise.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import bpy
except ImportError:  # pragma: no cover - the whole script needs Blender.
    print("ERROR: this script must be run inside Blender:")
    print("  blender -b --factory-startup --python tools/render_uv_report.py")
    raise SystemExit(1) from None

from unwrap_variations.backends import seams as seam_mod  # noqa: E402
from unwrap_variations.core import bridge, engine  # noqa: E402
from unwrap_variations.core.evaluator import evaluate  # noqa: E402
from unwrap_variations.core.meshdata import tri_areas_uv  # noqa: E402
from unwrap_variations.core.metrics import ScoreWeights, score  # noqa: E402

#: Blender's own Smart UV Project default, in radians (66 degrees).
SMART_ANGLE_LIMIT_RAD = 1.15192

#: Camera directions, as (azimuth degrees, elevation degrees).  Three fixed
#: angles: a front elevation, a three-quarter view that shows silhouette and
#: surface at once, and a top-down that catches limbs a front view hides.
VIEWS = (("front", 0.0, 8.0), ("three_quarter", 52.0, 26.0), ("top", 15.0, 74.0))

#: Island fill colours.  A fixed cycle, not a hash of the island id: a stable
#: palette makes two renders of the same mesh comparable at a glance, and the
#: order islands come out in is itself deterministic.
PALETTE = (
    (0.31, 0.55, 0.83), (0.94, 0.64, 0.27), (0.42, 0.72, 0.44), (0.78, 0.45, 0.72),
    (0.36, 0.74, 0.76), (0.87, 0.51, 0.44), (0.62, 0.60, 0.30), (0.55, 0.49, 0.80),
    (0.30, 0.66, 0.61), (0.85, 0.38, 0.53), (0.48, 0.58, 0.28), (0.70, 0.70, 0.42),
)


def _load(module_name: str, filename: str):
    """Import a sibling ``tools/`` script by path (``tools/`` is not a package)."""
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
# Software rasteriser for the UV layout
# ---------------------------------------------------------------------------


class Canvas:
    """A float RGB image with triangle fill and line drawing, in numpy.

    Deliberately tiny and deliberately not anti-aliased where it would cost
    determinism.  Supersampling is done by rendering at ``scale`` times the
    output size and box-filtering down, which is exact integer arithmetic and
    therefore reproducible on any machine.
    """

    def __init__(self, size: int, scale: int = 2, background=(0.09, 0.09, 0.11)) -> None:
        """Create a ``size * scale`` square canvas filled with ``background``."""
        self.size = int(size)
        self.scale = max(1, int(scale))
        self.res = self.size * self.scale
        self.buf = np.zeros((self.res, self.res, 3), dtype=np.float64)
        self.buf[:] = np.asarray(background, dtype=np.float64)

    def _to_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Map UV space to pixel space with a small margin, V pointing up."""
        margin = 0.04
        span = self.res * (1.0 - 2.0 * margin)
        out = np.empty_like(uv)
        out[..., 0] = self.res * margin + uv[..., 0] * span
        out[..., 1] = self.res * margin + (1.0 - uv[..., 1]) * span
        return out

    def fill_triangles(self, tri_uv: np.ndarray, colours: np.ndarray) -> None:
        """Fill each triangle with its colour, using a barycentric coverage test.

        Args:
            tri_uv: ``(T, 3, 2)`` UV corners.
            colours: ``(T, 3)`` RGB per triangle.
        """
        pix = self._to_pixels(tri_uv)
        lo = np.floor(pix.min(axis=1)).astype(np.int64)
        hi = np.ceil(pix.max(axis=1)).astype(np.int64) + 1
        np.clip(lo, 0, self.res, out=lo)
        np.clip(hi, 0, self.res, out=hi)
        for t in range(pix.shape[0]):
            x0, y0 = lo[t]
            x1, y1 = hi[t]
            if x1 <= x0 or y1 <= y0:
                continue
            ax, ay = pix[t, 0]
            bx, by = pix[t, 1]
            cx, cy = pix[t, 2]
            det = (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)
            if abs(det) < 1e-12:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            px = xs + 0.5
            py = ys + 0.5
            w0 = ((by - ay) * (px - ax) - (bx - ax) * (py - ay)) / det
            w1 = ((cy - by) * (px - bx) - (cx - bx) * (py - by)) / det
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not inside.any():
                continue
            self.buf[y0:y1, x0:x1][inside] = colours[t]

    def draw_segments(self, segments: np.ndarray, colour, width: int = 1) -> None:
        """Draw UV-space line segments by dense point sampling.

        Args:
            segments: ``(N, 2, 2)`` endpoint pairs in UV space.
            colour: RGB triple.
            width: Half-thickness in supersampled pixels.
        """
        if segments.shape[0] == 0:
            return
        pix = self._to_pixels(segments)
        delta = pix[:, 1] - pix[:, 0]
        steps = int(np.ceil(np.abs(delta).max())) + 2
        t = np.linspace(0.0, 1.0, steps)[None, :, None]
        pts = pix[:, 0][:, None, :] + delta[:, None, :] * t
        xs = np.rint(pts[..., 0]).astype(np.int64).reshape(-1)
        ys = np.rint(pts[..., 1]).astype(np.int64).reshape(-1)
        rgb = np.asarray(colour, dtype=np.float64)
        for dy in range(-width, width + 1):
            for dx in range(-width, width + 1):
                gx = np.clip(xs + dx, 0, self.res - 1)
                gy = np.clip(ys + dy, 0, self.res - 1)
                self.buf[gy, gx] = rgb

    def draw_unit_square(self, colour=(0.34, 0.35, 0.42)) -> None:
        """Outline the 0..1 UV square, so it is visible when a layout leaves it.

        Without it a layout packed into [0,1] and one spilling far outside look
        identical -- the rasteriser normalises neither, on purpose.
        """
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        segs = np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)
        self.draw_segments(segs, colour, width=1)

    def resolve(self) -> np.ndarray:
        """Box-filter the supersampled buffer down to the output resolution."""
        s = self.scale
        if s == 1:
            return self.buf
        view = self.buf.reshape(self.size, s, self.size, s, 3)
        return view.mean(axis=(1, 3))


def save_png(rgb: np.ndarray, path: str) -> None:
    """Write a float RGB array to a PNG through Blender's image system.

    ``bpy.data.images`` is the only PNG writer available inside Blender without
    a third-party dependency, and it works headless -- unlike the ``gpu``
    module.  The buffer is flipped because Blender's pixel origin is
    bottom-left and the rasteriser above works top-down.
    """
    height, width = rgb.shape[:2]
    name = "uvvar_render_%d" % (len(bpy.data.images) + 1)
    image = bpy.data.images.new(name, width=width, height=height, alpha=False)
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb, 0.0, 1.0)[::-1]
    image.pixels.foreach_set(rgba.reshape(-1))
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


# ---------------------------------------------------------------------------
# What to draw in red
# ---------------------------------------------------------------------------


def overlapping_triangles(tri_uv: np.ndarray, *, cells: int = 128) -> np.ndarray:
    """Return a ``(T,)`` bool mask of triangles that really overlap another.

    A uniform grid over triangle bounding boxes, then an exact convex clip on
    each surviving pair.  The exact clip matters: a separating-axis boolean
    reported 8773 false positives on a flat grid whose true answer is zero (see
    ``tests/reference.py``), and a picture full of false red is worse than no
    picture at all.
    """
    n = tri_uv.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    lo = tri_uv.min(axis=1)
    hi = tri_uv.max(axis=1)
    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    if not finite.any():
        return mask
    gmin = lo[finite].min(axis=0)
    gmax = hi[finite].max(axis=0)
    span = np.maximum(gmax - gmin, 1e-9)
    cell_lo = np.clip(((lo - gmin) / span * cells).astype(np.int64), 0, cells - 1)
    cell_hi = np.clip(((hi - gmin) / span * cells).astype(np.int64), 0, cells - 1)

    buckets: dict[int, list[int]] = {}
    for t in range(n):
        if not finite[t]:
            continue
        for cy in range(cell_lo[t, 1], cell_hi[t, 1] + 1):
            for cx in range(cell_lo[t, 0], cell_hi[t, 0] + 1):
                buckets.setdefault(cy * cells + cx, []).append(t)

    tested: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            i = members[a]
            for b in range(a + 1, len(members)):
                j = members[b]
                key = (i, j)
                if key in tested:
                    continue
                tested.add(key)
                if hi[i, 0] < lo[j, 0] or hi[j, 0] < lo[i, 0]:
                    continue
                if hi[i, 1] < lo[j, 1] or hi[j, 1] < lo[i, 1]:
                    continue
                if _clip_area(tri_uv[i], tri_uv[j]) > 1e-13:
                    mask[i] = True
                    mask[j] = True
    return mask


def _clip_area(a: np.ndarray, b: np.ndarray) -> float:
    """Return the exact intersection area of two UV triangles, both re-wound CCW."""
    a = _ccw(a)
    b = _ccw(b)
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


def _ccw(tri: np.ndarray) -> np.ndarray:
    """Return the triangle wound counter-clockwise."""
    area = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (
        tri[2][0] - tri[0][0]
    )
    return tri if area >= 0 else np.asarray([tri[0], tri[2], tri[1]])


def island_labels(sample) -> np.ndarray:
    """Return one island label per triangle, welded on ``(vertex, uv)``."""
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


def island_boundary_segments(sample, labels: np.ndarray) -> np.ndarray:
    """Return ``(N, 2, 2)`` UV segments along every island boundary.

    An edge of the UV mesh is a boundary when the two triangles either side put
    it in different places -- which is what a seam *is*, seen from UV space.
    Reading it off the layout rather than off the mesh's seam flags matters:
    ``smart_project`` invents its own charts and marks nothing, so asking the
    mesh would draw no boundaries at all on the very layouts that need them.
    """
    edges: dict[tuple[int, int, int], list[int]] = {}
    for t in range(sample.n_tris):
        lab = int(labels[t])
        tv = sample.tri_verts[t]
        for k in range(3):
            a, b = int(tv[k]), int(tv[(k + 1) % 3])
            edges.setdefault((lab, min(a, b), max(a, b)), []).append(t)

    out = []
    for (_lab, a, b), tris in edges.items():
        if len(tris) != 1:
            continue
        t = tris[0]
        uv_a = uv_b = None
        for k in range(3):
            if int(sample.tri_verts[t, k]) == a:
                uv_a = sample.uvs[int(sample.tri_loops[t, k])]
            elif int(sample.tri_verts[t, k]) == b:
                uv_b = sample.uvs[int(sample.tri_loops[t, k])]
        if uv_a is not None and uv_b is not None:
            out.append([uv_a, uv_b])
    return np.asarray(out, dtype=np.float64).reshape(-1, 2, 2)


def seam_flag_segments(sample) -> np.ndarray:
    """Return ``(N, 2, 2)`` UV segments for edges the mesh flags as seams."""
    if sample.n_edges == 0 or not bool(np.asarray(sample.edge_seam).any()):
        return np.zeros((0, 2, 2), dtype=np.float64)
    uv_of: dict[int, np.ndarray] = {}
    for t in range(sample.n_tris):
        for k in range(3):
            uv_of.setdefault(int(sample.tri_verts[t, k]), sample.uvs[int(sample.tri_loops[t, k])])
    out = []
    for e in range(sample.n_edges):
        if not sample.edge_seam[e]:
            continue
        a, b = int(sample.edge_verts[e, 0]), int(sample.edge_verts[e, 1])
        if a in uv_of and b in uv_of:
            out.append([uv_of[a], uv_of[b]])
    return np.asarray(out, dtype=np.float64).reshape(-1, 2, 2)


def render_uv_layout(sample, path: str, *, size: int) -> dict:
    """Draw the UV layout to ``path`` and return what was highlighted.

    Layers, painted in this order so the important things end on top:
    island fills, island boundaries, seam-flagged edges, then defects in red.
    """
    labels = island_labels(sample)
    tri_uv = sample.tri_uvs()
    canvas = Canvas(size)

    colours = np.asarray([PALETTE[int(lab) % len(PALETTE)] for lab in labels], dtype=np.float64)
    signed = tri_areas_uv(sample)
    flipped = signed < 0.0
    overlapping = overlapping_triangles(tri_uv)
    # Dim the fill of a defective triangle so the red outline reads against it.
    colours[flipped | overlapping] *= 0.45
    canvas.fill_triangles(tri_uv, colours)

    canvas.draw_unit_square()
    canvas.draw_segments(island_boundary_segments(sample, labels), (0.95, 0.95, 0.98), width=1)
    canvas.draw_segments(seam_flag_segments(sample), (1.00, 0.85, 0.20), width=1)

    defect = flipped | overlapping
    if defect.any():
        picked = tri_uv[defect]
        segs = np.concatenate(
            [picked[:, [0, 1]], picked[:, [1, 2]], picked[:, [2, 0]]], axis=0
        )
        canvas.draw_segments(segs, (1.0, 0.13, 0.13), width=1)

    save_png(canvas.resolve(), path)
    return {
        "islands": int(labels.max()) + 1 if labels.size else 0,
        "flipped": int(flipped.sum()),
        "overlapping": int(overlapping.sum()),
    }


# ---------------------------------------------------------------------------
# Textured 3D renders
# ---------------------------------------------------------------------------


def _eevee_engine() -> str:
    """Return this build's EEVEE identifier, probed from the enum.

    Measured: ``BLENDER_EEVEE_NEXT`` on 4.5.4 and ``BLENDER_EEVEE`` on 5.1.2.
    Probing the enum rather than branching on ``bpy.app.version`` means a third
    rename costs nothing.
    """
    items = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in items:
            return candidate
    return "BLENDER_WORKBENCH"


def _checker_material():
    """Return a material carrying Blender's own UV grid image.

    The stock ``UV_GRID`` generated image is used rather than a procedural
    checker because its coloured, numbered squares make it possible to tell
    *which* part of the layout you are looking at, and because it is generated
    by Blender itself and therefore identical on both supported builds.
    """
    image = bpy.data.images.get("uvvar_uv_grid")
    if image is None:
        image = bpy.data.images.new("uvvar_uv_grid", width=1024, height=1024)
        image.generated_type = "UV_GRID"

    mat = bpy.data.materials.new("uvvar_checker")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Closest"
    tex.location = (-320, 0)
    if bsdf is not None:
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.25
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.25
    return mat


def _setup_lighting() -> None:
    """Place two fixed suns and a neutral world, so every mesh is lit the same."""
    world = bpy.data.worlds.new("uvvar_world")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (0.10, 0.105, 0.125, 1.0)
        background.inputs[1].default_value = 1.0
    bpy.context.scene.world = world
    for name, rotation, energy in (
        ("key", (math.radians(52.0), 0.0, math.radians(38.0)), 5.5),
        ("fill", (math.radians(108.0), 0.0, math.radians(-125.0)), 2.4),
    ):
        light = bpy.data.lights.new("uvvar_%s" % name, type="SUN")
        light.energy = energy
        obj = bpy.data.objects.new("uvvar_%s" % name, light)
        obj.rotation_euler = rotation
        bpy.context.collection.objects.link(obj)


def _place_camera(target, azimuth_deg: float, elevation_deg: float):
    """Aim a camera at ``target``'s bounding sphere from a fixed direction.

    Framing is computed from the mesh's own bounds rather than fixed in world
    units, so a small mesh and a large one both fill the frame and the pictures
    stay comparable.  Nothing here depends on a Blender operator, so the result
    is identical on both supported builds.
    """
    verts = np.asarray([v.co[:] for v in target.data.vertices], dtype=np.float64)
    centre = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    radius = float(np.linalg.norm(verts - centre, axis=1).max()) or 1.0

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    direction = np.array(
        [math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)]
    )
    distance = radius * 3.1
    location = centre + direction * distance

    cam_data = bpy.data.cameras.new("uvvar_cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("uvvar_cam", cam_data)
    cam.location = location.tolist()
    # Point the camera down -direction: yaw about Z, then pitch from horizontal.
    cam.rotation_euler = (math.radians(90.0) - el, 0.0, az)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def render_views(obj, prefix: str, *, size: int, samples: int) -> list[str]:
    """Render ``obj`` from every fixed view; return the paths written."""
    scene = bpy.context.scene
    scene.render.engine = _eevee_engine()
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    eevee = getattr(scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = int(samples)

    obj.data.materials.clear()
    obj.data.materials.append(_checker_material())

    written = []
    for name, azimuth, elevation in VIEWS:
        cam = _place_camera(obj, azimuth, elevation)
        path = "%s_%s.png" % (prefix, name)
        scene.render.filepath = path
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:  # noqa: BLE001 -- a missing GPU must not abort the run
            print("  render failed for %s: %r" % (path, exc))
        else:
            written.append(path)
        bpy.data.objects.remove(cam, do_unlink=True)
    return written


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def variant_baseline(obj) -> str:
    """Apply Blender's stock Smart UV Project; return a label for the report."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=SMART_ANGLE_LIMIT_RAD, island_margin=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    return "smart_project (Blender defaults)"


def variant_bestofn(obj, *, count: int, seed: int, weights, timeout: float) -> tuple[str, object]:
    """Generate and apply the top-ranked candidate; return ``(label, report)``.

    ``timeout`` goes through the engine's own ``should_cancel`` probe.  A
    reporting script that hangs produces nothing at all, which is strictly worse
    than a report with one honest "generation timed out" row in it.
    """
    deadline = time.perf_counter() + timeout if timeout > 0.0 else None
    report = engine.generate(
        obj, count=count, seed=seed, weights=weights, store=False, compute_overlap=True,
        should_cancel=(lambda: time.perf_counter() > deadline) if deadline else None,
    )
    if not report.candidates:
        return "best-of-%d (GENERATED NOTHING)" % count, report
    best = max(report.candidates, key=lambda c: c.score)
    bridge.apply_uvs(obj, best.uvs, layer_name=obj.data.uv_layers.active.name)
    suffix = " [timed out, %d slots]" % len(report.candidates) if getattr(
        report, "cancelled", False) else ""
    return "best-of-%d%s" % (count, suffix), report


def measure(obj, weights) -> tuple[object, float]:
    """Return ``(metrics, score)`` of whatever layout ``obj`` currently carries.

    Seams are derived from the layout, not read from the mesh flags, for the
    reason ``tools/benchmark_bestofn.py`` documents: ``smart_project`` marks
    nothing and would otherwise report a seam length of zero.
    """
    sample = bridge.extract_sample(obj)
    topo = seam_mod.build_edge_topology(sample)
    seams = engine.derive_uv_seams(sample, topo, sample.uvs)
    metrics = evaluate(sample.with_uvs(sample.uvs).with_seams(seams), compute_overlap=True)
    return metrics, score(metrics, weights)


# ---------------------------------------------------------------------------
# The index page
# ---------------------------------------------------------------------------

_CSS = """
body { background:#15161a; color:#e8e8ee; font:14px/1.5 system-ui,sans-serif; margin:24px; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:17px; margin:34px 0 2px; border-bottom:1px solid #333; padding-bottom:4px; }
p.why { color:#a9a9b8; margin:2px 0 12px; }
table { border-collapse:collapse; margin:10px 0 22px; }
td, th { padding:5px 8px; text-align:left; vertical-align:top; }
th { color:#9d9dad; font-weight:600; }
tr.v td { border-top:1px solid #2a2b33; }
img { display:block; border:1px solid #2f303a; background:#0d0e11; }
.imgs { display:flex; gap:8px; flex-wrap:wrap; }
.cap { color:#8b8b99; font-size:12px; margin:3px 0 0; }
.num { font-variant-numeric:tabular-nums; }
.bad { color:#ff6b6b; font-weight:600; }
.good { color:#69d67f; }
.legend { color:#a9a9b8; background:#1c1d23; padding:10px 14px; border-radius:6px; }
"""


def _fmt(value: float, bad: bool) -> str:
    """Return an HTML span, red when the value is a defect."""
    text = "%.6g" % value if isinstance(value, float) else str(value)
    return '<span class="%s">%s</span>' % ("bad" if bad else "good", html.escape(text))


def write_index(path: str, entries: list[dict], meta: dict) -> None:
    """Write the index page tying every picture to the numbers behind it."""
    parts = [
        "<title>UnwrapVariations -- organic corpus visual report</title>",
        "<style>%s</style>" % _CSS,
        "<h1>Organic corpus visual report</h1>",
        '<p class="why">Blender %s | seed %d | %d candidates | preset %s | %s</p>'
        % (
            html.escape(meta["blender"]),
            meta["seed"],
            meta["count"],
            html.escape(meta["preset"]),
            html.escape(meta["stamp"]),
        ),
        '<p class="legend">In the UV panels: each island has its own fill colour; '
        "white lines are island boundaries read back from the layout; yellow lines are "
        "edges the mesh flags as seams; <b>red outlines mark every triangle that really "
        "overlaps another or is flipped</b>, confirmed by exact convex clipping rather "
        "than a separating-axis test. Blender's UV operators are non-deterministic "
        "(BENCHMARKS.md section 3), so a re-run gives a visibly similar but not "
        "identical layout.</p>",
    ]
    for entry in entries:
        parts.append("<h2>%s</h2>" % html.escape(entry["mesh"]))
        parts.append('<p class="why">%s &mdash; %d triangles, %d shells, chi %s</p>'
                     % (html.escape(entry["why"]), entry["tris"], entry["shells"],
                        html.escape(entry["chi"])))
        parts.append("<table><tr><th>variant</th><th>score</th><th>islands</th>"
                     "<th>seam ratio</th><th>stretch L2</th><th>overlap area</th>"
                     "<th>flipped</th></tr>")
        for variant in entry["variants"]:
            parts.append(
                '<tr class="v"><td>%s</td><td class="num">%.2f</td><td class="num">%d</td>'
                '<td class="num">%.3f</td><td class="num">%.4f</td><td class="num">%s</td>'
                '<td class="num">%s</td></tr>'
                % (
                    html.escape(variant["label"]),
                    variant["score"],
                    variant["islands"],
                    variant["seam_ratio"],
                    variant["stretch_l2"],
                    _fmt(variant["overlap_area"], variant["overlap_area"] > 0.0),
                    _fmt(variant["flipped"], variant["flipped"] > 0),
                )
            )
        parts.append("</table>")
        for variant in entry["variants"]:
            parts.append('<p class="cap">%s</p><div class="imgs">'
                         % html.escape(variant["label"]))
            for img, caption in variant["images"]:
                parts.append(
                    '<div><img src="%s" width="300"><p class="cap">%s</p></div>'
                    % (html.escape(os.path.basename(img)), html.escape(caption))
                )
            parts.append("</div>")
    with open(path, "w", encoding="ascii", errors="xmlcharrefreplace") as fh:
        fh.write("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv):
    """Parse the arguments Blender passed through after a bare ``--``."""
    parser = argparse.ArgumentParser(
        prog="render_uv_report.py",
        description="Render the organic corpus and its UV layouts to an HTML report.",
    )
    parser.add_argument("--out", default="renders", help="Output directory (default renders/).")
    parser.add_argument("--count", type=int, default=10, help="Candidates per mesh.")
    parser.add_argument("--seed", type=int, default=1, help="Determinism seed.")
    parser.add_argument("--preset", default="balanced", help="ScoreWeights preset.")
    parser.add_argument("--uv-size", type=int, default=900, help="UV panel size in pixels.")
    parser.add_argument("--view-size", type=int, default=512, help="3D view size in pixels.")
    parser.add_argument("--samples", type=int, default=16, help="EEVEE render samples.")
    parser.add_argument(
        "--meshes", default="", help="Comma-separated corpus subset (default all)."
    )
    parser.add_argument(
        "--no-views", action="store_true", help="Skip the 3D renders; UV panels only."
    )
    parser.add_argument(
        "--gen-timeout",
        type=float,
        default=300.0,
        help="Seconds before one generation is cancelled through the engine's own "
             "should_cancel probe (0 disables).",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Render the report.  Returns a process exit code."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    weights = ScoreWeights.preset(args.preset)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    wanted = [n.strip() for n in args.meshes.split(",") if n.strip()] or list(corpus.NAMES)
    unknown = [n for n in wanted if n not in corpus.NAMES]
    if unknown:
        print("ERROR: unknown mesh(es): %s" % ", ".join(unknown))
        return 1

    print("RENDERING %d corpus meshes to %s" % (len(wanted), out_dir))
    print("Blender %s | engine %s | seed %d | count %d"
          % (bpy.app.version_string, _eevee_engine(), args.seed, args.count))

    entries = []
    failures = []
    for name in wanted:
        started = time.time()
        try:
            mesh = corpus.build(name)
        except Exception as exc:  # noqa: BLE001 -- one bad mesh must not stop the report
            print("%-17s BUILD FAILED: %s" % (name, exc))
            failures.append(name)
            continue

        variant_ids = ["baseline", "bestofn"]
        if mesh.uvs is not None:
            # A mesh that ships its own layout gets that layout rendered too.
            # For folded_ribbon this is the deliberate defect itself, and being
            # able to look at it is the difference between believing the
            # injectivity oracle and taking it on trust.
            variant_ids.insert(0, "corpus")
        variants = []
        for variant_id in variant_ids:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            _setup_lighting()
            obj = corpus.to_object(mesh)
            if variant_id == "corpus":
                label = "corpus layout (as shipped)"
            elif variant_id == "baseline":
                label = variant_baseline(obj)
            else:
                label, _report = variant_bestofn(
                    obj, count=args.count, seed=args.seed, weights=weights,
                    timeout=args.gen_timeout,
                )
            metrics, value = measure(obj, weights)

            prefix = os.path.join(out_dir, "%s_%s" % (name, variant_id))
            sample = bridge.extract_sample(obj)
            drawn = render_uv_layout(sample, prefix + "_uv.png", size=args.uv_size)
            images = [(prefix + "_uv.png", "UV layout -- %d islands, %d red"
                       % (drawn["islands"], drawn["flipped"] + drawn["overlapping"]))]
            if not args.no_views:
                for path in render_views(
                    obj, prefix, size=args.view_size, samples=args.samples
                ):
                    images.append((path, os.path.basename(path).rsplit("_", 1)[-1][:-4]))

            variants.append(
                {
                    "label": label,
                    "score": value,
                    "islands": int(metrics.island_count),
                    "seam_ratio": float(metrics.seam_ratio),
                    "stretch_l2": float(metrics.stretch_l2),
                    "overlap_area": float(metrics.overlap_area),
                    "flipped": int(metrics.flipped_tris),
                    "images": images,
                }
            )
            print("  %-9s %-24s score %6.2f  islands %4d  overlap %.4g  red %d"
                  % (name, variant_id, value, metrics.island_count, metrics.overlap_area,
                     drawn["flipped"] + drawn["overlapping"]))

        inv = mesh.invariants
        entries.append(
            {
                "mesh": name,
                "why": mesh.why_hard,
                "tris": inv.n_tris,
                "shells": inv.shells,
                "chi": ",".join(str(c) for c in inv.chi),
                "variants": variants,
            }
        )
        print("%-17s done in %.1fs" % (name, time.time() - started))

    index = os.path.join(out_dir, "index.html")
    write_index(
        index,
        entries,
        {
            "blender": bpy.app.version_string,
            "seed": args.seed,
            "count": args.count,
            "preset": args.preset,
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print("-" * 78)
    print("wrote %s" % index)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
