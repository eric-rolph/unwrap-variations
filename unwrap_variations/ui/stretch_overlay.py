"""The stretch / angle / checker viewport overlay and the checker material.

Two independent ways to *see* what a UV layout is doing to your texture, because
they answer different questions:

* The **GPU overlay** shades every triangle by its own distortion.  It is drawn
  with a ``SpaceView3D`` ``POST_VIEW`` handler over the solid shading, needs no
  material, no texture slot and no scene changes at all, and it answers "where
  exactly is this unwrap bad?".
* The **checker material** swaps a generated checker texture onto the object so
  the eye can read texel density and orientation directly off the surface.  It
  answers "are my texels the same size everywhere, and are they square?".  It is
  the only genuinely destructive thing this add-on's UI does, so it saves and
  restores the original material assignment *and* the viewport shading mode.

Robustness notes, in order of how badly they bite:

* A draw handler that survives ``unregister()`` and then touches freed data will
  hard-crash Blender.  :func:`unregister` always removes it, and the callback
  itself only ever reaches datablocks through :data:`bpy.data` name lookups, so
  a deleted object degrades to "draw nothing" instead of a dangling pointer.
* The GPU batch is rebuilt only when a cheap identity key changes (object,
  mode, element counts, active candidate, explicit invalidation).  Rebuilding a
  batch every redraw would make a dense mesh unusable.
* ``gpu.shader.from_builtin`` names were renamed between Blender generations.
  The legacy ``'3D_SMOOTH_COLOR'`` / ``'2D_IMAGE'`` spellings **do not exist**
  on 4.5.4 or 5.1.2 -- both accept only the unprefixed names.  Resolution walks
  a preference list so the module keeps working either way.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import bpy
from bpy.props import EnumProperty
from bpy.types import Context, Object, Operator, UILayout

try:  # numpy ships inside Blender, but never assume.
    import numpy as np

    HAVE_NUMPY = True
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]
    HAVE_NUMPY = False

__all__ = [
    "OVERLAY_MODES",
    "OP_OVERLAY_CONTROL",
    "OP_CHECKER_MATERIAL",
    "CHECKER_MATERIAL_NAME",
    "is_enabled",
    "get_mode",
    "set_mode",
    "enable_overlay",
    "disable_overlay",
    "toggle_overlay",
    "invalidate",
    "sync_from_props",
    "draw_legend",
    "checker_material_is_active",
    "apply_checker_material",
    "restore_checker_material",
    "toggle_checker_material",
    "resolve_shader",
    "UVVAR_OT_overlay_control",
    "UVVAR_OT_checker_material",
    "classes",
    "register",
    "unregister",
]

#: Overlay shading modes, in the order they appear in the UI.
OVERLAY_MODES: tuple[str, ...] = ("STRETCH", "ANGLE", "CHECKER")

#: Operator ids owned by this module.  Deliberately *not* the ones in
#: ARCHITECTURE section 10 -- those belong to ``operators/`` -- so both can be
#: registered side by side without an id clash.
OP_OVERLAY_CONTROL = "uvvar.overlay_control"
OP_CHECKER_MATERIAL = "uvvar.checker_material"

#: Name of the generated checker material and of its texture image.
CHECKER_MATERIAL_NAME = "UVVar_Checker"

#: Object custom property holding the JSON material/shading backup.
BACKUP_KEY = "uvvar_material_backup"

#: Scratch UV layer written by the preview operator (ARCHITECTURE section 7).
PREVIEW_LAYER = "UVVar_Preview"

DEFAULT_OPACITY = 0.55

#: Builtin shader names to try, best first.  Verified on 4.5.4 and 5.1.2: only
#: the unprefixed spellings exist; the ``3D_``/``2D_`` names raise ValueError.
#: The legacy names stay in the list purely so the module also works on 3.x.
_SHADER_PREFERENCE: dict[str, tuple[str, ...]] = {
    "smooth": ("SMOOTH_COLOR", "3D_SMOOTH_COLOR"),
    "flat": ("FLAT_COLOR", "3D_FLAT_COLOR"),
    "uniform": ("UNIFORM_COLOR", "3D_UNIFORM_COLOR"),
    "image": ("IMAGE", "3D_IMAGE", "2D_IMAGE"),
}

#: Colour ramps as ``(stop, (r, g, b))``.  Blue = compressed (too few texels),
#: green = correct, red = stretched (too many texels) -- the same convention as
#: Blender's own UV stretch overlay, so the reading transfers.
_RAMP_STRETCH: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.0, (0.13, 0.36, 0.92)),
    (0.5, (0.16, 0.78, 0.36)),
    (1.0, (0.95, 0.27, 0.15)),
)
_RAMP_ANGLE: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.0, (0.16, 0.78, 0.36)),
    (0.55, (0.97, 0.80, 0.18)),
    (1.0, (0.95, 0.27, 0.15)),
)

#: Legend rows per mode: ``(icon, caption)``.  ``COLORSET_nn_VEC`` icons are
#: real colour swatches, which beats trying to describe a ramp in words.
_LEGEND: dict[str, tuple[tuple[str, str], ...]] = {
    "STRETCH": (
        ("COLORSET_04_VEC", "Compressed - texels too small"),
        ("COLORSET_03_VEC", "Even - matches 3D area"),
        ("COLORSET_01_VEC", "Stretched - texels too large"),
    ),
    "ANGLE": (
        ("COLORSET_03_VEC", "Angles preserved"),
        ("COLORSET_09_VEC", "Sheared"),
        ("COLORSET_01_VEC", "Badly sheared - 20 deg or more"),
    ),
    "CHECKER": (
        ("MESH_GRID", "Squares equal size = even texel density"),
        ("MESH_GRID", "Squares stretched = distortion"),
        ("COLORSET_02_VEC", "Orange cell marks UV origin"),
    ),
}


class _OverlayState:
    """Mutable module state for the viewport overlay.

    Attributes:
        handle: The value returned by ``draw_handler_add``, or ``None``.
        obj_name: Name of the object being overlaid, ``""`` to follow the
            active object.
        mode: One of :data:`OVERLAY_MODES`.
        opacity: Overlay alpha in ``[0, 1]``.
        batch: The cached :class:`gpu.types.GPUBatch`.
        shader: The cached shader used with ``batch``.
        texture: The cached checker :class:`gpu.types.GPUTexture`.
        key: Identity of the data the cached batch was built from.
        version: Bumped by :func:`invalidate` to force a rebuild.
        error: Last build error, surfaced in the panel.
    """

    def __init__(self) -> None:
        self.handle: Any = None
        self.obj_name: str = ""
        self.mode: str = "STRETCH"
        self.opacity: float = DEFAULT_OPACITY
        self.batch: Any = None
        self.shader: Any = None
        self.texture: Any = None
        self.key: tuple[Any, ...] | None = None
        self.version: int = 0
        self.error: str = ""

    def drop_batch(self) -> None:
        """Release the cached GPU batch so the next draw rebuilds it."""
        self.batch = None
        self.key = None


_state = _OverlayState()
_shader_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Shader resolution
# ---------------------------------------------------------------------------


def resolve_shader(kind: str) -> Any:
    """Return a builtin GPU shader for ``kind``, trying every known spelling.

    Blender renamed its builtin shaders when the 2D/3D split was removed: the
    ``'3D_SMOOTH_COLOR'`` spelling that older add-ons use raises ``ValueError``
    on 4.5 and 5.1, where the name is simply ``'SMOOTH_COLOR'``.  Rather than
    branch on ``bpy.app.version`` -- which is exactly the check that rots -- this
    walks a preference list and caches whatever the running build accepts.

    Args:
        kind: One of ``"smooth"``, ``"flat"``, ``"uniform"`` or ``"image"``.

    Returns:
        A ``gpu.types.GPUShader``, or ``None`` if no spelling worked (which is
        also the case in background mode, where the GPU module refuses to draw).
    """
    if kind in _shader_cache:
        return _shader_cache[kind]
    shader = None
    try:
        import gpu

        for name in _SHADER_PREFERENCE.get(kind, ()):
            try:
                shader = gpu.shader.from_builtin(name)
                break
            except Exception:  # noqa: BLE001 - wrong spelling for this build.
                continue
    except Exception:  # noqa: BLE001 - no gpu module at all.
        shader = None
    _shader_cache[kind] = shader
    return shader


# ---------------------------------------------------------------------------
# Public overlay API.  ``operators/`` may drive the overlay through these.
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return whether the viewport draw handler is currently installed."""
    return _state.handle is not None


def get_mode() -> str:
    """Return the active overlay mode."""
    return _state.mode


def set_mode(mode: str) -> str:
    """Set the overlay mode and force a batch rebuild.

    Args:
        mode: One of :data:`OVERLAY_MODES`; anything else falls back to
            ``"STRETCH"``.

    Returns:
        The mode that was actually set.
    """
    clean = str(mode or "").upper()
    _state.mode = clean if clean in OVERLAY_MODES else "STRETCH"
    invalidate()
    return _state.mode


def invalidate() -> None:
    """Mark the cached batch stale so the next redraw rebuilds it.

    Call this after previewing a different candidate or after any change to the
    mesh or its UVs.  It is cheap -- it only bumps a counter.
    """
    _state.version += 1
    _state.drop_batch()


def enable_overlay(context: Context | None = None, obj_name: str = "") -> tuple[bool, str]:
    """Install the viewport draw handler.

    Args:
        context: Optional context used to pick the target object.
        obj_name: Pin the overlay to this object; empty follows the active one.

    Returns:
        ``(ok, message)``.
    """
    if is_enabled():
        _state.obj_name = obj_name or _state.obj_name
        invalidate()
        return True, "Overlay already on"
    if not HAVE_NUMPY:
        return False, "numpy unavailable; overlay needs it"
    try:
        import gpu  # noqa: F401 - presence check only.
        from gpu_extras.batch import batch_for_shader  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, "GPU module unavailable: %s" % exc
    if resolve_shader("smooth") is None:
        return False, "No usable builtin shader (background mode?)"

    if not obj_name and context is not None:
        obj = getattr(context, "active_object", None)
        obj_name = getattr(obj, "name", "") if obj is not None else ""
    _state.obj_name = obj_name

    try:
        _state.handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), "WINDOW", "POST_VIEW"
        )
    except Exception as exc:  # noqa: BLE001
        _state.handle = None
        return False, "Could not add draw handler: %s" % exc

    _install_handlers()
    invalidate()
    _tag_redraw()
    return True, "Overlay on (%s)" % _state.mode.title()


def disable_overlay() -> tuple[bool, str]:
    """Remove the viewport draw handler and release GPU resources.

    Safe to call when the overlay is already off, and safe to call during
    ``unregister()`` -- which is exactly when a forgotten handler turns into a
    crash the next time Blender redraws.

    Returns:
        ``(ok, message)``.
    """
    handle, _state.handle = _state.handle, None
    if handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except Exception:  # noqa: BLE001 - already gone; nothing to do.
            pass
    _state.drop_batch()
    _state.texture = None
    _remove_handlers()
    _tag_redraw()
    return True, "Overlay off"


def toggle_overlay(context: Context | None = None) -> tuple[bool, str]:
    """Turn the overlay on if it is off, and off if it is on.

    Args:
        context: Optional context used to pick the target object.

    Returns:
        ``(ok, message)``.
    """
    return disable_overlay() if is_enabled() else enable_overlay(context)


def sync_from_props(context: Context | None) -> tuple[bool, str]:
    """Bring the overlay in line with ``Scene.uvvar``.

    Provided so ``operators/`` can implement ``uvvar.toggle_overlay`` with a
    single call and without importing any drawing code.  An ``overlay_mode`` of
    ``NONE`` or ``OFF`` turns the overlay off; anything else turns it on with
    that mode.

    Args:
        context: Current context; ``None`` falls back to :data:`bpy.context`.

    Returns:
        ``(ok, message)``.
    """
    ctx = context if context is not None else bpy.context
    props = None
    try:
        props = getattr(getattr(ctx, "scene", None), "uvvar", None)
    except Exception:  # noqa: BLE001
        props = None

    mode = str(getattr(props, "overlay_mode", _state.mode) or _state.mode).upper()
    try:
        _state.opacity = float(getattr(props, "overlay_opacity", _state.opacity))
    except (TypeError, ValueError):
        pass

    if mode in ("NONE", "OFF", "DISABLED"):
        return disable_overlay()
    set_mode(mode)
    return enable_overlay(ctx)


# ---------------------------------------------------------------------------
# Depsgraph / load handlers keep the batch honest without polling
# ---------------------------------------------------------------------------


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene: Any, depsgraph: Any = None) -> None:
    """Invalidate the batch when the overlaid geometry actually changes."""
    if _state.handle is None:
        return
    try:
        updates = getattr(depsgraph, "updates", None)
        if updates is None:
            invalidate()
            return
        for update in updates:
            if getattr(update, "is_updated_geometry", False):
                invalidate()
                return
    except Exception:  # noqa: BLE001
        pass


@bpy.app.handlers.persistent
def _on_load_post(*args: Any) -> None:
    """Drop the overlay on file load; the old batch refers to a dead scene."""
    try:
        disable_overlay()
    except Exception:  # noqa: BLE001
        pass


def _install_handlers() -> None:
    """Add the depsgraph and load handlers if they are not already there."""
    try:
        if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
        if _on_load_post not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_on_load_post)
    except Exception:  # noqa: BLE001
        pass


def _remove_handlers() -> None:
    """Remove the depsgraph and load handlers, ignoring absence."""
    for collection, func in (
        (getattr(bpy.app.handlers, "depsgraph_update_post", None), _on_depsgraph_update),
        (getattr(bpy.app.handlers, "load_post", None), _on_load_post),
    ):
        try:
            if collection is not None and func in collection:
                collection.remove(func)
        except Exception:  # noqa: BLE001
            pass


def _tag_redraw() -> None:
    """Ask every 3D viewport to redraw."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def _target_object() -> Object | None:
    """Return the mesh object to overlay, or ``None``.

    Resolution goes through :data:`bpy.data` by name so a deleted object yields
    ``None`` rather than a freed ``StructRNA`` -- touching one of those from a
    draw handler is a hard crash, not an exception.
    """
    try:
        obj = bpy.data.objects.get(_state.obj_name) if _state.obj_name else None
        if obj is None:
            obj = getattr(bpy.context, "active_object", None)
        if obj is None or getattr(obj, "type", "") != "MESH":
            return None
        if obj.data is None:
            return None
        return obj
    except Exception:  # noqa: BLE001
        return None


def _read_geometry(obj: Object) -> tuple[Any, Any, Any] | None:
    """Read triangles and UVs for ``obj``, handling Edit Mode.

    In Object Mode the original datablock is preferred: it is the base cage the
    user is actually unwrapping, so the distortion reported is the distortion of
    the UVs being edited rather than of some subdivided or mirrored result.

    In **Edit Mode** that datablock is useless -- the live geometry lives in the
    BMesh and ``mesh.loop_triangles`` comes back empty -- and the evaluated
    object's ``.data`` is no better, because the edit cage is not stored there.
    ``Object.to_mesh()`` on the *evaluated* object does produce a real
    :class:`bpy.types.Mesh` with the ``foreach_get`` fast paths intact, so that
    is the fallback.  Since UV work happens almost entirely in Edit Mode, this
    is the path that matters most.

    Args:
        obj: The mesh object to read.

    Returns:
        ``(tri_positions, tri_loops, uvs)`` or ``None``.
    """
    # In Edit Mode the original datablock still reports its pre-edit triangle
    # count, so reading it can succeed while returning stale geometry.  Skip it
    # outright rather than relying on the UV read happening to fail.
    if getattr(obj, "mode", "") != "EDIT":
        arrays = _read_mesh_arrays(obj.data)
        if arrays is not None:
            return arrays

    arrays = None
    evaluated = None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        temporary = evaluated.to_mesh()
        if temporary is not None:
            # Every array below is copied out by foreach_get or by fancy
            # indexing, so freeing the temporary mesh immediately is safe.
            arrays = _read_mesh_arrays(temporary)
    except Exception:  # noqa: BLE001
        arrays = None
    finally:
        try:
            if evaluated is not None:
                evaluated.to_mesh_clear()
        except Exception:  # noqa: BLE001
            pass
    return arrays


def _uv_layer(mesh: Any) -> Any:
    """Return the UV layer to visualise: the preview layer, else the active."""
    try:
        layers = mesh.uv_layers
        if len(layers) == 0:
            return None
        return layers.get(PREVIEW_LAYER) or layers.active or layers[0]
    except Exception:  # noqa: BLE001
        return None


def _build_key(obj: Object) -> tuple[Any, ...]:
    """Return a cheap identity for the data the batch is built from.

    Every element is an O(1) read, so this can safely run once per redraw; the
    expensive numpy work only happens when the tuple actually changes.
    """
    mesh = obj.data
    layer = _uv_layer(mesh)
    active_index = 0
    try:
        active_index = int(getattr(bpy.context.scene.uvvar, "active_index", 0))
    except Exception:  # noqa: BLE001
        pass
    return (
        obj.name,
        # Entering or leaving Edit Mode swaps which mesh the batch is read from.
        getattr(obj, "mode", ""),
        _state.mode,
        round(_state.opacity, 3),
        len(mesh.vertices),
        len(mesh.polygons),
        len(mesh.loops),
        getattr(layer, "name", ""),
        active_index,
        _state.version,
    )


def _read_mesh_arrays(mesh: Any) -> tuple[Any, Any, Any] | None:
    """Return ``(positions, tri_loops, uvs)`` via the verified fast paths.

    Returns:
        ``(V*3 float32, T*3 int32, L*2 float32)`` reshaped, or ``None`` if the
        mesh has no triangles or no UVs.
    """
    try:
        if hasattr(mesh, "calc_loop_triangles"):
            try:
                mesh.calc_loop_triangles()
            except Exception:  # noqa: BLE001 - 4.1+ keeps these up to date.
                pass
        n_verts = len(mesh.vertices)
        n_tris = len(mesh.loop_triangles)
        n_loops = len(mesh.loops)
        if n_verts == 0 or n_tris == 0 or n_loops == 0:
            return None

        co = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)

        tri_loops = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)

        tri_verts = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_verts)

        layer = _uv_layer(mesh)
        if layer is None:
            return None
        uvs = np.empty(n_loops * 2, dtype=np.float32)
        if hasattr(layer, "uv"):
            layer.uv.foreach_get("vector", uvs)
        else:
            layer.data.foreach_get("uv", uvs)

        return (
            co.reshape(-1, 3)[tri_verts.reshape(-1, 3)],
            tri_loops.reshape(-1, 3),
            uvs.reshape(-1, 2),
        )
    except Exception:  # noqa: BLE001
        return None


def _ramp(values: Any, stops: Sequence[tuple[float, tuple[float, float, float]]]) -> Any:
    """Map ``values`` in ``[0, 1]`` onto an RGB ramp.

    Args:
        values: ``(N,)`` float array.
        stops: Ramp control points, ascending by position.

    Returns:
        ``(N, 3)`` float32 RGB.
    """
    t = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    out = np.empty((t.shape[0], 3), dtype=np.float32)
    for channel in range(3):
        ys = np.array([s[1][channel] for s in stops], dtype=np.float32)
        out[:, channel] = np.interp(t, xs, ys)
    return out


def _stretch_factors(tri_pos: Any, tri_uv: Any) -> Any:
    """Return ``0..1`` area-distortion per triangle, ``0.5`` meaning perfect.

    The per-triangle ratio of normalised UV area to normalised 3D area is taken
    in log2 space and mapped so that a factor of four in either direction hits
    the ends of the ramp.  Normalising both totals first makes the reading
    independent of the object's scale and of the layout's overall UV scale.
    """
    e1 = tri_pos[:, 1] - tri_pos[:, 0]
    e2 = tri_pos[:, 2] - tri_pos[:, 0]
    area3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    f1 = tri_uv[:, 1] - tri_uv[:, 0]
    f2 = tri_uv[:, 2] - tri_uv[:, 0]
    area_uv = np.abs(0.5 * (f1[:, 0] * f2[:, 1] - f1[:, 1] * f2[:, 0]))

    total3 = float(area3.sum())
    total_uv = float(area_uv.sum())
    if total3 <= 0.0 or total_uv <= 0.0:
        return np.full(area3.shape[0], 0.5, dtype=np.float32)

    eps = 1e-12
    ratio = (area_uv / total_uv + eps) / (area3 / total3 + eps)
    log_ratio = np.log2(np.clip(ratio, 1e-9, 1e9))
    return np.clip(0.5 + log_ratio / 4.0, 0.0, 1.0).astype(np.float32)


def _angle_factors(tri_pos: Any, tri_uv: Any) -> Any:
    """Return ``0..1`` angle distortion per triangle, ``0`` meaning perfect.

    The mean absolute difference between each triangle's three 3D corner angles
    and the corresponding UV corner angles, in degrees, scaled so that 20
    degrees of average shear saturates the ramp.
    """

    def corner_angles(points: Any) -> Any:
        """Return ``(T, 3)`` corner angles in radians."""
        angles = np.empty((points.shape[0], 3), dtype=np.float64)
        for i in range(3):
            a = points[:, (i + 1) % 3] - points[:, i]
            b = points[:, (i + 2) % 3] - points[:, i]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            denom = np.where((na * nb) > 1e-20, na * nb, 1.0)
            cosine = np.clip(np.einsum("ij,ij->i", a, b) / denom, -1.0, 1.0)
            angles[:, i] = np.arccos(cosine)
        return angles

    a3 = corner_angles(tri_pos.astype(np.float64))
    auv = corner_angles(tri_uv.astype(np.float64))
    deviation = np.degrees(np.abs(a3 - auv)).mean(axis=1)
    return np.clip(np.nan_to_num(deviation, nan=0.0) / 20.0, 0.0, 1.0).astype(np.float32)


def _offset_positions(tri_pos: Any) -> Any:
    """Nudge triangles along their normals so they do not z-fight the surface.

    The offset is proportional to the object's own bounding box, which keeps it
    invisible on a 1 m cube and still effective on a 500 m terrain.
    """
    flat = tri_pos.reshape(-1, 3)
    span = float(np.max(flat, axis=0).max() - np.min(flat, axis=0).min()) if flat.size else 1.0
    eps = max(abs(span), 1e-6) * 8e-4

    e1 = tri_pos[:, 1] - tri_pos[:, 0]
    e2 = tri_pos[:, 2] - tri_pos[:, 0]
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1)
    lengths = np.where(lengths > 1e-20, lengths, 1.0)
    normals = (normals / lengths[:, None]) * eps
    return (tri_pos + normals[:, None, :]).astype(np.float32)


def _checker_texture(size: int = 512, cells: int = 32) -> Any:
    """Build (once) the generated checker texture used by ``CHECKER`` mode.

    The cell count matters more than it looks: a typical unwrap packs islands
    that occupy a tenth of UV space or less, so a coarse 8x8 checker puts a
    whole island inside a single cell and the overlay reads as flat blocks
    rather than as a grid.  32 cells puts several squares across even a small
    island, which is what makes distortion legible.

    Args:
        size: Texture edge length in pixels.
        cells: Checker squares along each edge of UV space.

    Returns:
        A ``gpu.types.GPUTexture``, or ``None`` if it could not be created.
    """
    if _state.texture is not None:
        return _state.texture
    try:
        import gpu

        axis = (np.arange(size) * cells) // size
        parity = (axis[:, None] + axis[None, :]) & 1
        image = np.empty((size, size, 4), dtype=np.float32)
        dark = (0.17, 0.18, 0.21)
        light = (0.82, 0.83, 0.87)
        for channel in range(3):
            image[:, :, channel] = np.where(parity == 1, light[channel], dark[channel])
        image[:, :, 3] = 1.0
        # One tinted cell so UV rotation and mirroring are readable at a glance.
        step = max(1, size // cells)
        image[0:step, 0:step, 0] = 0.95
        image[0:step, 0:step, 1] = 0.45
        image[0:step, 0:step, 2] = 0.10

        flat = np.ascontiguousarray(image.reshape(-1))
        try:
            buffer = gpu.types.Buffer("FLOAT", flat.size, flat)
        except Exception:  # noqa: BLE001 - older builds want a plain sequence.
            buffer = gpu.types.Buffer("FLOAT", flat.size, flat.tolist())
        _state.texture = gpu.types.GPUTexture(
            (size, size), format="RGBA32F", data=buffer
        )
    except Exception:  # noqa: BLE001
        _state.texture = None
    return _state.texture


def _rebuild(obj: Object) -> bool:
    """Rebuild the cached GPU batch for ``obj``.

    Returns:
        ``True`` when a drawable batch is now cached.
    """
    _state.drop_batch()
    _state.error = ""
    if not HAVE_NUMPY:
        _state.error = "numpy unavailable"
        return False
    try:
        from gpu_extras.batch import batch_for_shader
    except Exception as exc:  # noqa: BLE001
        _state.error = "gpu_extras unavailable: %s" % exc
        return False

    arrays = _read_geometry(obj)
    if arrays is None:
        _state.error = "mesh has no triangles or no UV layer"
        return False
    tri_pos, tri_loops, uvs = arrays
    tri_uv = uvs[tri_loops]
    positions = _offset_positions(tri_pos).reshape(-1, 3)

    mode = _state.mode
    alpha = float(min(max(_state.opacity, 0.05), 1.0))

    if mode == "CHECKER":
        texture = _checker_texture()
        shader = resolve_shader("image") if texture is not None else None
        if shader is not None and texture is not None:
            try:
                batch = batch_for_shader(
                    shader,
                    "TRIS",
                    {
                        "pos": positions,
                        "texCoord": tri_uv.reshape(-1, 2).astype(np.float32),
                    },
                )
                _state.shader = shader
                _state.batch = batch
                return True
            except Exception as exc:  # noqa: BLE001 - attribute name mismatch.
                _state.error = "image shader batch failed: %s" % exc
        # Fallback: per-corner checker parity through the smooth-colour shader.
        cells = 32.0
        parity = (
            np.floor(tri_uv[:, :, 0] * cells) + np.floor(tri_uv[:, :, 1] * cells)
        ).astype(np.int64) & 1
        shade = np.where(parity == 1, 0.82, 0.18).astype(np.float32).reshape(-1)
        colors = np.empty((shade.shape[0], 4), dtype=np.float32)
        colors[:, 0] = shade
        colors[:, 1] = shade
        colors[:, 2] = shade
        colors[:, 3] = alpha
    else:
        if mode == "ANGLE":
            factors = _angle_factors(tri_pos, tri_uv)
            rgb = _ramp(factors, _RAMP_ANGLE)
        else:
            factors = _stretch_factors(tri_pos, tri_uv)
            rgb = _ramp(factors, _RAMP_STRETCH)
        per_corner = np.repeat(rgb, 3, axis=0)
        colors = np.empty((per_corner.shape[0], 4), dtype=np.float32)
        colors[:, :3] = per_corner
        colors[:, 3] = alpha

    shader = resolve_shader("smooth")
    if shader is None:
        _state.error = "no usable builtin shader"
        return False
    try:
        _state.batch = batch_for_shader(
            shader, "TRIS", {"pos": positions, "color": colors}
        )
        _state.shader = shader
        return True
    except Exception as exc:  # noqa: BLE001
        _state.error = "batch build failed: %s" % exc
        _state.drop_batch()
        return False


def _draw_callback() -> None:
    """``POST_VIEW`` draw handler.  Must never raise and never touch dead data."""
    try:
        if _state.handle is None:
            return
        obj = _target_object()
        if obj is None:
            return

        key = _build_key(obj)
        if _state.batch is None or key != _state.key:
            if not _rebuild(obj):
                _state.key = key  # Do not retry every redraw on a broken mesh.
                return
            _state.key = key
        if _state.batch is None or _state.shader is None:
            return

        import gpu

        prev_depth_mask = gpu.state.depth_mask_get()
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        gpu.state.face_culling_set("NONE")
        try:
            with gpu.matrix.push_pop():
                gpu.matrix.multiply_matrix(obj.matrix_world)
                _state.shader.bind()
                if _state.mode == "CHECKER" and _state.texture is not None:
                    try:
                        _state.shader.uniform_sampler("image", _state.texture)
                    except Exception:  # noqa: BLE001 - colour fallback batch.
                        pass
                _state.batch.draw(_state.shader)
        finally:
            gpu.state.depth_mask_set(prev_depth_mask)
            gpu.state.depth_test_set("NONE")
            gpu.state.blend_set("NONE")
    except Exception:  # noqa: BLE001 - a draw handler must never propagate.
        _state.drop_batch()


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def draw_legend(layout: UILayout, mode: str = "") -> None:
    """Draw the colour-ramp key for ``mode`` into a panel.

    Args:
        layout: Layout to draw into.
        mode: Overlay mode; empty uses the current one.
    """
    try:
        clean = (mode or _state.mode or "STRETCH").upper()
        rows = _LEGEND.get(clean, _LEGEND["STRETCH"])
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.9
        col.label(text="%s legend" % clean.title())
        for icon, caption in rows:
            try:
                col.label(text=caption, icon=icon)
            except Exception:  # noqa: BLE001 - unknown icon on this build.
                col.label(text=caption)
        if _state.error:
            warn = col.row()
            warn.alert = True
            warn.label(text=_state.error[:56], icon="ERROR")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Procedural checker material
# ---------------------------------------------------------------------------


def ensure_checker_material() -> Any:
    """Create or reuse the generated checker material.

    A ``ShaderNodeTexChecker`` driven by the mesh's own UVs, wired through an
    emission-flavoured Principled setup so the checker reads clearly in Material
    Preview shading without depending on scene lighting.

    Returns:
        The material, or ``None`` if it could not be built.
    """
    try:
        mat = bpy.data.materials.get(CHECKER_MATERIAL_NAME)
        if mat is not None:
            return mat
        mat = bpy.data.materials.new(CHECKER_MATERIAL_NAME)
        mat.use_nodes = True
        tree = mat.node_tree
        nodes, links = tree.nodes, tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (420, 0)
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (140, 0)
        checker = nodes.new("ShaderNodeTexChecker")
        checker.location = (-140, 0)
        coords = nodes.new("ShaderNodeUVMap")
        coords.location = (-380, 0)

        try:
            # Matches the GPU overlay's 32 cells so the two readings agree.
            checker.inputs["Scale"].default_value = 32.0
            checker.inputs["Color1"].default_value = (0.16, 0.17, 0.20, 1.0)
            checker.inputs["Color2"].default_value = (0.82, 0.83, 0.87, 1.0)
        except Exception:  # noqa: BLE001 - socket names differ across versions.
            pass
        for socket, value in (("Roughness", 0.6), ("Metallic", 0.0)):
            try:
                principled.inputs[socket].default_value = value
            except Exception:  # noqa: BLE001
                pass

        links.new(coords.outputs["UV"], checker.inputs["Vector"])
        links.new(checker.outputs["Color"], principled.inputs["Base Color"])
        links.new(principled.outputs[0], output.inputs["Surface"])
        return mat
    except Exception:  # noqa: BLE001
        return None


def checker_material_is_active(obj: Object | None) -> bool:
    """Return whether ``obj`` currently has the checker material forced on."""
    try:
        return obj is not None and BACKUP_KEY in obj
    except Exception:  # noqa: BLE001
        return False


def _save_shading(activate: bool = True) -> list[dict[str, Any]]:
    """Record every 3D viewport's shading mode, optionally switching to MATERIAL.

    Args:
        activate: When ``True``, switch each viewport to Material Preview so the
            procedural checker is actually visible.

    Returns:
        A JSON-safe list of records for :func:`_restore_shading`.
    """
    saved: list[dict[str, Any]] = []
    try:
        for wi, window in enumerate(bpy.context.window_manager.windows):
            for ai, area in enumerate(window.screen.areas):
                if area.type != "VIEW_3D":
                    continue
                for si, space in enumerate(area.spaces):
                    if getattr(space, "type", "") != "VIEW_3D":
                        continue
                    shading = space.shading
                    saved.append(
                        {
                            "w": wi,
                            "a": ai,
                            "s": si,
                            "type": shading.type,
                            "color_type": shading.color_type,
                        }
                    )
                    if activate:
                        try:
                            shading.type = "MATERIAL"
                        except Exception:  # noqa: BLE001
                            pass
    except Exception:  # noqa: BLE001
        pass
    return saved


def _restore_shading(saved: Iterable[dict[str, Any]]) -> None:
    """Put every recorded viewport back the way :func:`_save_shading` found it."""
    try:
        windows = list(bpy.context.window_manager.windows)
        for record in saved or ():
            try:
                area = windows[int(record["w"])].screen.areas[int(record["a"])]
                space = area.spaces[int(record["s"])]
                if getattr(space, "type", "") != "VIEW_3D":
                    continue
                space.shading.type = record.get("type", "SOLID")
                space.shading.color_type = record.get("color_type", "MATERIAL")
            except Exception:  # noqa: BLE001 - layout changed since we saved.
                continue
    except Exception:  # noqa: BLE001
        pass


def apply_checker_material(
    context: Context | None, obj: Object | None
) -> tuple[bool, str]:
    """Force the generated checker material onto ``obj``, saving what was there.

    The original assignment is preserved by switching each slot's link to
    ``OBJECT`` and overriding only the *object* level material, which leaves the
    mesh datablock -- and therefore every other object sharing it -- untouched.
    The full backup is written to an object custom property as JSON so it also
    survives a script reload or a save/load cycle.

    Args:
        context: Current context, used only for viewport shading.
        obj: Target mesh object.

    Returns:
        ``(ok, message)``.
    """
    if obj is None or getattr(obj, "type", "") != "MESH":
        return False, "Active object is not a mesh"
    if checker_material_is_active(obj):
        return False, "Checker material is already active"

    mat = ensure_checker_material()
    if mat is None:
        return False, "Could not build the checker material"

    backup: dict[str, Any] = {"slots": [], "added_data_slot": False, "shading": []}
    try:
        if len(obj.material_slots) == 0:
            obj.data.materials.append(mat)
            backup["added_data_slot"] = True
        else:
            for slot in obj.material_slots:
                original_link = slot.link
                # Reading the object-level material requires the OBJECT link;
                # flipping the flag back and forth changes no data.
                try:
                    slot.link = "OBJECT"
                    object_material = slot.material.name if slot.material else ""
                except Exception:  # noqa: BLE001
                    object_material = ""
                backup["slots"].append(
                    {"link": original_link, "object_material": object_material}
                )
                slot.link = "OBJECT"
                slot.material = mat
        backup["shading"] = _save_shading(activate=True)
        obj[BACKUP_KEY] = json.dumps(backup)
    except Exception as exc:  # noqa: BLE001
        # Best effort rollback so a half-applied state never survives.
        try:
            _restore_from_backup(obj, backup)
        except Exception:  # noqa: BLE001
            pass
        return False, "Could not apply checker material: %s" % exc
    _tag_redraw()
    return True, "Checker material on '%s'" % obj.name


def _restore_from_backup(obj: Object, backup: dict[str, Any]) -> None:
    """Reverse :func:`apply_checker_material` from a decoded backup dict."""
    if backup.get("added_data_slot"):
        try:
            obj.data.materials.pop(index=len(obj.data.materials) - 1)
        except Exception:  # noqa: BLE001
            pass
    else:
        # strict=False on purpose: the user may have added or removed material
        # slots while the checker was applied, and restoring what still lines up
        # beats refusing to restore anything at all.
        for slot, record in zip(
            obj.material_slots, backup.get("slots", ()), strict=False
        ):
            try:
                slot.link = "OBJECT"
                name = record.get("object_material") or ""
                slot.material = bpy.data.materials.get(name) if name else None
                slot.link = record.get("link", "DATA")
            except Exception:  # noqa: BLE001
                continue
    _restore_shading(backup.get("shading", ()))


def restore_checker_material(
    context: Context | None, obj: Object | None
) -> tuple[bool, str]:
    """Undo :func:`apply_checker_material` and drop the backup.

    Args:
        context: Current context (unused, kept for symmetry).
        obj: Target mesh object.

    Returns:
        ``(ok, message)``.
    """
    if obj is None:
        return False, "No object"
    if not checker_material_is_active(obj):
        return False, "No checker material backup on '%s'" % obj.name
    try:
        backup = json.loads(str(obj[BACKUP_KEY]))
    except Exception:  # noqa: BLE001
        backup = {"slots": [], "added_data_slot": False, "shading": []}
    try:
        _restore_from_backup(obj, backup)
    except Exception as exc:  # noqa: BLE001
        return False, "Restore failed: %s" % exc
    finally:
        try:
            del obj[BACKUP_KEY]
        except Exception:  # noqa: BLE001
            pass
    # Clean up the generated material once nothing references it any more.
    try:
        mat = bpy.data.materials.get(CHECKER_MATERIAL_NAME)
        if mat is not None and mat.users == 0:
            bpy.data.materials.remove(mat)
    except Exception:  # noqa: BLE001
        pass
    _tag_redraw()
    return True, "Materials restored on '%s'" % obj.name


def toggle_checker_material(
    context: Context | None, obj: Object | None = None
) -> tuple[bool, str]:
    """Apply the checker material, or restore the originals if it is already on.

    Args:
        context: Current context.
        obj: Target object; defaults to the context's active object.

    Returns:
        ``(ok, message)``.
    """
    if obj is None:
        obj = getattr(context or bpy.context, "active_object", None)
    if checker_material_is_active(obj):
        return restore_checker_material(context, obj)
    return apply_checker_material(context, obj)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class UVVAR_OT_overlay_control(Operator):
    """Enable, disable or rebuild the distortion overlay."""

    bl_idname = OP_OVERLAY_CONTROL
    bl_label = "UV Distortion Overlay"
    bl_description = "Toggle the stretch, angle or checker viewport overlay"
    bl_options = {"REGISTER", "INTERNAL"}

    action: EnumProperty(
        name="Action",
        description="What to do with the overlay",
        items=(
            ("TOGGLE", "Toggle", "Turn the overlay on or off"),
            ("ENABLE", "Enable", "Turn the overlay on"),
            ("DISABLE", "Disable", "Turn the overlay off"),
            ("REFRESH", "Refresh", "Rebuild the overlay from current data"),
            ("SET_MODE", "Set Mode", "Switch the overlay shading mode"),
        ),
        default="TOGGLE",
    )
    mode: EnumProperty(
        name="Mode",
        description="Overlay shading mode",
        items=tuple(
            (name, name.title(), "Shade by %s" % name.lower()) for name in OVERLAY_MODES
        ),
        default="STRETCH",
    )

    def execute(self, context: Context) -> set[str]:
        """Run the requested overlay action."""
        try:
            if self.action == "ENABLE":
                ok, message = enable_overlay(context)
            elif self.action == "DISABLE":
                ok, message = disable_overlay()
            elif self.action == "REFRESH":
                invalidate()
                _tag_redraw()
                ok, message = True, "Overlay rebuilt"
            elif self.action == "SET_MODE":
                set_mode(self.mode)
                ok, message = True, "Overlay mode: %s" % self.mode.title()
                if not is_enabled():
                    ok, message = enable_overlay(context)
            else:
                ok, message = toggle_overlay(context)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Overlay failed: %s" % exc)
            return {"CANCELLED"}
        if not ok:
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


class UVVAR_OT_checker_material(Operator):
    """Swap in a generated checker material, or put the originals back."""

    bl_idname = OP_CHECKER_MATERIAL
    bl_label = "Checker Material"
    bl_description = (
        "Assign a generated checker material so texel density is visible on the "
        "model. The original materials and viewport shading are restored on toggle"
    )
    bl_options = {"REGISTER", "UNDO"}

    action: EnumProperty(
        name="Action",
        items=(
            ("TOGGLE", "Toggle", "Apply or restore"),
            ("APPLY", "Apply", "Assign the checker material"),
            ("RESTORE", "Restore", "Put the original materials back"),
        ),
        default="TOGGLE",
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Only available with an active mesh object."""
        obj = getattr(context, "active_object", None)
        return obj is not None and getattr(obj, "type", "") == "MESH"

    def execute(self, context: Context) -> set[str]:
        """Apply or restore the checker material."""
        obj = getattr(context, "active_object", None)
        try:
            if self.action == "APPLY":
                ok, message = apply_checker_material(context, obj)
            elif self.action == "RESTORE":
                ok, message = restore_checker_material(context, obj)
            else:
                ok, message = toggle_checker_material(context, obj)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Checker material failed: %s" % exc)
            return {"CANCELLED"}
        if not ok:
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        self.report({"INFO"}, message)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: Every registrable class in this module.
classes: tuple[type, ...] = (
    UVVAR_OT_overlay_control,
    UVVAR_OT_checker_material,
)


def register() -> None:
    """Register this module's operators."""
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception:  # noqa: BLE001
            pass


def unregister() -> None:
    """Remove the draw handler first, then unregister the operators.

    Order matters: a draw handler left installed after the module's code is gone
    will be called by Blender on the next redraw and take the process with it.
    """
    try:
        disable_overlay()
    except Exception:  # noqa: BLE001
        pass
    _remove_handlers()
    _shader_cache.clear()
    _state.texture = None
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            pass
