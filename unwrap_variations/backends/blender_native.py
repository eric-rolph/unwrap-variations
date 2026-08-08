"""The built-in backend: drives Blender's own UV operators.

This is the only module in the add-on that calls ``bpy.ops.uv.*``.  Everything
awkward about doing that from a script is contained here:

* **Degrees stay in the core.** :attr:`UnwrapParams.angle_limit_deg` is in
  degrees; ``bpy.ops.uv.smart_project`` wants **radians**.  The single
  ``math.radians`` call in :func:`_run_method` is the one and only place that
  conversion happens.
* **Operators need EDIT mode and a full selection.** They act on selected,
  unhidden geometry, so the backend forces the object active, reveals and
  selects everything, and puts all of it back afterwards.
* **Packing needs selected UVs.** ``bpy.ops.uv.pack_islands`` packs *selected*
  islands.  On Blender 4.5 ``tool_settings.use_uv_select_sync`` defaults to
  ``False`` and a fresh unwrap leaves every UV **deselected**, so the operator
  finds nothing to pack and returns ``CANCELLED``.  Blender 5.x flipped that
  default to ``True``, where UV selection follows the (fully selected) mesh and
  the same call returns ``FINISHED``.  It was never a screen/area problem:
  overriding an ``IMAGE_EDITOR`` area is impossible headless anyway
  (``temp_override(area=...)`` raises "Area not found in screen" because the
  background window's screen holds no areas).  :func:`_pack_islands` fixes it
  properly by selecting every UV first, with a sync-selection retry, and treats
  a still-cancelled pack as a note on an otherwise valid candidate.
* **Nothing leaks.** :func:`_capture_state` snapshots mode, active object,
  visibility, seams, hide and selection flags, UV-selection attributes, tool
  settings and the UV layer stack; :func:`_restore_state` puts every one of them
  back from a ``finally``, step by step, swallowing its own errors.

The unwrap itself happens in a throwaway UV layer (``UVVar__work``) that is
deleted on the way out, so the user's own UV maps are never written to.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, ClassVar, Sequence

import numpy as np

from ..core.params import (
    MARGIN_METHODS,
    METHODS,
    PACK_SHAPE_METHODS,
    ROTATE_METHODS,
    UnwrapParams,
)
from .base import UnwrapBackend, UnwrapRequest, UnwrapResult, _Notes, register_backend

try:  # pragma: no cover - exercised implicitly when the sibling module exists
    from ..backends.seams import SEAM_STRATEGIES, compute_seams
except BaseException:  # pragma: no cover - keep importable if seams.py is absent
    SEAM_STRATEGIES = (
        "NONE",
        "EXISTING",
        "SHARP",
        "ANGLE",
        "CURVATURE",
        "DUAL_GRAPH",
        "SMART",
    )

    def compute_seams(  # type: ignore[misc]
        sample: Any,
        strategy: str,
        *,
        angle_deg: float = 30.0,
        curvature_threshold: float = 0.35,
        existing: np.ndarray | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Degenerate stand-in used only when ``backends.seams`` cannot be imported.

        Keeps this module importable during parallel development.  ``NONE``
        clears every seam; anything else preserves the seams already on the
        mesh, which is the least surprising fallback.
        """
        n_edges = int(getattr(sample, "n_edges", 0))
        if strategy == "NONE" or existing is None:
            return np.zeros(n_edges, dtype=bool)
        return np.asarray(existing, dtype=bool).reshape(-1)


__all__ = ["BlenderNativeBackend", "WORK_UV_LAYER"]

#: Name of the scratch UV layer the operators write into.  Deleted on exit; a
#: leftover from a crashed run is removed at the start of the next one.
WORK_UV_LAYER = "UVVar__work"

#: Blender's hard cap on UV layers per mesh (see ARCHITECTURE section 0).
MAX_UV_LAYERS = 8

#: Methods routed through ``bpy.ops.uv.unwrap``; these are the seam-driven ones.
SOLVER_METHODS = ("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")

#: Methods that project rather than solve, and therefore ignore seams entirely.
PROJECTION_METHODS = ("SMART_PROJECT", "CUBE_PROJECT")

#: ``rotate_method`` handed to ``bpy.ops.uv.pack_islands``.
#:
#: Left at Blender's own default so that a candidate matches what the user would
#: get by pressing Pack Islands by hand.  Tempting though it is to pick a
#: "faster" constant here, the cost of each rotate method is strongly
#: input-dependent and *not* monotonic: measured on Blender 4.5.4,
#: ``AXIS_ALIGNED_X`` totals 1.13 s across a ten-case suite where ``ANY`` totals
#: 26.12 s -- yet on a cube smart-projected at 30 degrees with margin 0.02 it is
#: the **worst** option (7.43 s versus 0.85 s for ``ANY``).  No constant is safe,
#: so the real fix is :data:`CONCAVE_MIN_TRIS` below.  Overridable per recipe
#: with ``params.extra["pack_rotate_method"]``; ``extra`` participates in
#: :meth:`UnwrapParams.signature`, so overriding still de-duplicates correctly.
DEFAULT_PACK_ROTATE_METHOD = "ANY"

#: Triangle count below which packing always uses ``shape_method='AABB'``.
#:
#: Blender's island packer runs an exhaustive placement search whose cost *rises*
#: as the island count *falls*, so it is pathologically slow on exactly the
#: meshes that should be instant.  Subdividing a single cube, measured on 4.5.4
#: with ``pack_islands`` alone:
#:
#: ======  =======  ===========
#: faces   islands  ``CONCAVE``
#: ======  =======  ===========
#: 6             6       8.46 s
#: 24            6       0.34 s
#: 54            6       0.23 s
#: 96            6       0.15 s
#: ======  =======  ===========
#:
#: No constant choice of ``rotate_method``/``shape_method`` is safe: the knobs
#: interact chaotically and non-monotonically (on one cube recipe ``CONCAVE``
#: costs 7.18 s and ``CONVEX`` 0.044 s; on another, substituting ``CONVEX``
#: made a ten-candidate run *nine times slower*).  So instead of guessing, six
#: policies were benchmarked end-to-end -- ten candidates from
#: :func:`~unwrap_variations.core.params.sample_grid`, on Blender 4.5.4:
#:
#: =============================  ========  ========  =========  =============
#: policy                             cube      cone    Suzanne  mean coverage
#: =============================  ========  ========  =========  =============
#: ``ANY`` + shape as-is            10.64 s   28.65 s     6.77 s         0.6855
#: ``AXIS_ALIGNED_X`` + as-is        9.36 s    3.66 s     2.25 s         0.6741
#: ``AXIS_ALIGNED_Y`` + as-is        3.07 s   12.04 s     2.68 s         0.6728
#: ``ANY`` + small -> ``CONVEX``    98.99 s   25.32 s     7.02 s         0.6857
#: ``ANY`` + small -> ``AABB``       0.01 s    0.02 s     7.19 s         0.6906
#: ``AXIS_ALIGNED_X`` + ``AABB``     0.01 s    0.02 s     2.19 s         0.6862
#: =============================  ========  ========  =========  =============
#:
#: Substituting ``AABB`` below this threshold is ~1000x faster on a cube and
#: ~1400x on a cone, and -- because the exhaustive search was not actually
#: finding better layouts on so few islands -- it also scores the **highest**
#: mean packing coverage of any policy tested.  Above the threshold the
#: requested shape method is passed through untouched, where it is both
#: affordable and genuinely better (+0.08 coverage on a torus).
#:
#: Consequence worth knowing: on a mesh below this size ``pack_shape_method``
#: has no effect, which is why a candidate says so in its message.
#: ``params.extra["pack_shape_guard"] = False`` disables the substitution.
SMALL_MESH_TRIS = 256

#: Shape method substituted for small meshes.  See :data:`SMALL_MESH_TRIS`.
SMALL_MESH_SHAPE_METHOD = "AABB"

#: Legal ``rotate_method`` identifiers for ``uv.pack_islands`` (a superset of
#: :data:`unwrap_variations.core.params.ROTATE_METHODS`).
PACK_ROTATE_METHODS = ("ANY", "CARDINAL", "AXIS_ALIGNED", "AXIS_ALIGNED_X",
                       "AXIS_ALIGNED_Y")

#: Names of the boolean mesh attributes that carry UV selection and pinning.
#: Blender 4.5 stores them per UV map as ``.vs.<name>`` / ``.es.<name>`` /
#: ``.pn.<name>``; Blender 5.x replaced those with single global layers.  Both
#: spellings are snapshotted so selecting all UVs is undone either way.
_UV_SELECT_ATTR_PREFIXES = (".vs.", ".es.", ".pn.")
_UV_SELECT_ATTR_NAMES = (
    ".uv_select_vert",
    ".uv_select_edge",
    ".uv_select_face",
    ".uv_pin",
)


# ---------------------------------------------------------------------------
# Small bpy helpers
# ---------------------------------------------------------------------------


def _import_bpy() -> ModuleType | None:
    """Return the :mod:`bpy` module, or ``None`` when running outside Blender."""
    try:
        import bpy  # noqa: PLC0415 - deliberately lazy so the module imports anywhere
    except BaseException:  # noqa: BLE001
        return None
    return bpy


def _enum(value: Any, allowed: Sequence[str], default: str) -> str:
    """Return ``value`` if it is one of ``allowed``, else ``default``.

    Passing an unknown enum identifier to an operator raises ``TypeError``, so
    every enum crossing into ``bpy.ops`` goes through here first.
    """
    text = str(value)
    return text if text in tuple(allowed) else default


def _call_op(op: Callable[..., Any], **kwargs: Any) -> tuple[bool, str]:
    """Invoke a Blender operator and normalise its outcome.

    Args:
        op: The ``bpy.ops`` callable.
        **kwargs: Operator properties.

    Returns:
        ``(finished, detail)`` where ``finished`` is ``True`` only for a
        ``FINISHED`` return, and ``detail`` describes what happened -- the
        returned status set, or the exception text if the call raised (a failed
        ``poll()`` raises ``RuntimeError`` rather than returning ``CANCELLED``).
    """
    try:
        status = op(**kwargs)
    except BaseException as exc:  # noqa: BLE001 - operators raise on poll failure
        return False, "%s: %s" % (type(exc).__name__, exc)
    try:
        names = ",".join(sorted(status))
    except BaseException:  # noqa: BLE001
        names = repr(status)
    return ("FINISHED" in status), names


def _quiet(action: Callable[[], Any]) -> bool:
    """Run ``action``, swallowing anything it raises.

    Used throughout :func:`_restore_state`, where a failure to put one property
    back must not prevent the remaining properties from being put back.
    """
    try:
        action()
    except BaseException:  # noqa: BLE001 - restoration is strictly best-effort
        return False
    return True


def _leave_edit_mode(bpy: ModuleType) -> None:
    """Force whatever object is active back into OBJECT mode.

    Mesh data (attributes, UV layer creation, ``foreach_set``) is only reliable
    from OBJECT mode, because in EDIT mode the authoritative copy lives in a
    BMesh that has not been flushed yet.
    """
    if bpy.context.mode != "OBJECT":
        _quiet(lambda: bpy.ops.object.mode_set(mode="OBJECT"))


def _read_bool_array(collection: Any, prop: str, count: int) -> np.ndarray | None:
    """Return ``count`` values of boolean property ``prop``, or ``None`` if unreadable.

    Returning ``None`` rather than raising lets the snapshot survive a Blender
    build that spells one of these properties differently; the corresponding
    restore step is then simply skipped.
    """
    buf = np.zeros(count, dtype=bool)
    if not count:
        return buf
    try:
        collection.foreach_get(prop, buf)
    except BaseException:  # noqa: BLE001
        return None
    return buf


def _layer_index(layers: Any, name: str) -> int:
    """Return the index of UV layer ``name``, or ``-1`` when it is absent."""
    try:
        found = int(layers.find(name))
        if found >= 0:
            return found
    except BaseException:  # noqa: BLE001 - fall back to a linear scan
        pass
    for index, layer in enumerate(layers):
        if layer.name == name:
            return index
    return -1


def _read_uv_array(layer: Any, n_loops: int) -> np.ndarray:
    """Return a UV layer's contents as an ``(L, 2)`` float32 array.

    Uses the verified fast path ``layer.uv.foreach_get("vector", ...)`` and
    falls back to the legacy ``layer.data.foreach_get("uv", ...)`` alias.
    """
    buf = np.zeros(n_loops * 2, dtype=np.float32)
    if not n_loops:
        return buf.reshape(0, 2)
    try:
        layer.uv.foreach_get("vector", buf)
    except BaseException:  # noqa: BLE001 - older/newer RNA spelling
        layer.data.foreach_get("uv", buf)
    return buf.reshape(n_loops, 2)


def _write_uv_array(layer: Any, uvs: np.ndarray) -> None:
    """Blit an ``(L, 2)`` array back into a UV layer."""
    flat = np.ascontiguousarray(uvs, dtype=np.float32).reshape(-1)
    try:
        layer.uv.foreach_set("vector", flat)
    except BaseException:  # noqa: BLE001 - older/newer RNA spelling
        layer.data.foreach_set("uv", flat)


# ---------------------------------------------------------------------------
# Scene state snapshot
# ---------------------------------------------------------------------------


@dataclass
class _SceneState:
    """Everything :meth:`BlenderNativeBackend.unwrap` is allowed to disturb.

    Captured before any modification and replayed from a ``finally`` block, so
    that a candidate generation run is invisible to the user apart from the
    numbers it returns.
    """

    obj_name: str
    obj_mode: str = "OBJECT"
    obj_selected: bool = True
    obj_hide_viewport: bool = False
    obj_hide_view_layer: bool | None = None
    active_name: str | None = None
    active_mode: str | None = None

    uv_active_name: str | None = None
    uv_active_index: int = 0
    uv_render_name: str | None = None
    uv_backup: dict[str, np.ndarray] = field(default_factory=dict)
    work_layer: str | None = None

    seams: np.ndarray | None = None
    select_vert: np.ndarray | None = None
    select_edge: np.ndarray | None = None
    select_poly: np.ndarray | None = None
    hide_vert: np.ndarray | None = None
    hide_edge: np.ndarray | None = None
    hide_poly: np.ndarray | None = None
    uv_attrs: list[tuple[str, np.ndarray]] = field(default_factory=list)

    uv_select_sync: bool | None = None


def _snapshot_uv_select_attrs(mesh: Any) -> list[tuple[str, np.ndarray]]:
    """Return ``(name, values)`` for every UV-selection / pin boolean attribute.

    Selecting all UVs (which packing requires) writes these, so they have to be
    restored like any other user state.  The set of names differs between 4.5
    and 5.x, hence the prefix *and* exact-name matching.
    """
    out: list[tuple[str, np.ndarray]] = []
    try:
        attributes = list(mesh.attributes)
    except BaseException:  # noqa: BLE001
        return out
    for attr in attributes:
        name = getattr(attr, "name", "")
        interesting = name in _UV_SELECT_ATTR_NAMES or any(
            name.startswith(p) for p in _UV_SELECT_ATTR_PREFIXES
        )
        if not interesting or getattr(attr, "data_type", "") != "BOOLEAN":
            continue
        try:
            count = len(attr.data)
            buf = np.zeros(count, dtype=bool)
            if count:
                attr.data.foreach_get("value", buf)
            out.append((name, buf))
        except BaseException:  # noqa: BLE001 - skip anything unreadable
            continue
    return out


def _capture_state(bpy: ModuleType, obj: Any) -> _SceneState:
    """Snapshot every piece of scene state this backend may modify.

    Modes are recorded *first*, because the very next thing this does is drop
    out of EDIT mode -- mesh arrays read while an object is being edited are
    stale, since the authoritative copy lives in an unflushed BMesh.
    """
    state = _SceneState(obj_name=obj.name)
    state.obj_mode = obj.mode
    state.obj_selected = bool(_safe(lambda: obj.select_get(), True))
    state.obj_hide_viewport = bool(_safe(lambda: obj.hide_viewport, False))
    state.obj_hide_view_layer = _safe(lambda: bool(obj.hide_get()), None)

    view_layer = _safe(lambda: bpy.context.view_layer, None)
    active = _safe(lambda: view_layer.objects.active, None) if view_layer else None
    if active is not None:
        state.active_name = active.name
        state.active_mode = active.mode

    # Safe to flush now that the modes are on record.
    _leave_edit_mode(bpy)

    mesh = obj.data
    layers = mesh.uv_layers
    state.uv_active_index = int(getattr(layers, "active_index", 0) or 0)
    active_layer = _safe(lambda: layers.active, None)
    state.uv_active_name = active_layer.name if active_layer is not None else None
    for layer in layers:
        if bool(getattr(layer, "active_render", False)):
            state.uv_render_name = layer.name
            break

    state.seams = _read_bool_array(mesh.edges, "use_seam", len(mesh.edges))
    state.select_vert = _read_bool_array(mesh.vertices, "select", len(mesh.vertices))
    state.select_edge = _read_bool_array(mesh.edges, "select", len(mesh.edges))
    state.select_poly = _read_bool_array(mesh.polygons, "select", len(mesh.polygons))
    state.hide_vert = _read_bool_array(mesh.vertices, "hide", len(mesh.vertices))
    state.hide_edge = _read_bool_array(mesh.edges, "hide", len(mesh.edges))
    state.hide_poly = _read_bool_array(mesh.polygons, "hide", len(mesh.polygons))
    state.uv_attrs = _snapshot_uv_select_attrs(mesh)

    tool_settings = _safe(lambda: bpy.context.scene.tool_settings, None)
    if tool_settings is not None:
        state.uv_select_sync = _safe(lambda: bool(tool_settings.use_uv_select_sync), None)
    return state


def _safe(getter: Callable[[], Any], default: Any) -> Any:
    """Return ``getter()``, or ``default`` if it raises."""
    try:
        return getter()
    except BaseException:  # noqa: BLE001
        return default


def _restore_state(bpy: ModuleType, state: _SceneState) -> None:
    """Put every snapshotted property back, best-effort and in dependency order.

    Mesh data has to be written from OBJECT mode, and the object's mode has to
    be restored last (re-entering EDIT mode rebuilds the BMesh from whatever the
    mesh holds at that moment), which fixes the ordering below.
    """
    obj = bpy.data.objects.get(state.obj_name)
    if obj is None or getattr(obj, "type", None) != "MESH":
        return
    _leave_edit_mode(bpy)
    mesh = obj.data

    # 1. Drop the scratch UV layer.
    if state.work_layer:
        layer = _safe(lambda: mesh.uv_layers.get(state.work_layer), None)
        if layer is not None:
            _quiet(lambda: mesh.uv_layers.remove(layer))

    # 2. Put back the contents of any real UV layer we had to borrow.
    for name, values in state.uv_backup.items():
        layer = _safe(lambda n=name: mesh.uv_layers.get(n), None)
        if layer is not None:
            _quiet(lambda ly=layer, v=values: _write_uv_array(ly, v))

    # 3. Active / render UV layer, restored by name because indices shift.
    def _restore_active_uv() -> None:
        layers = mesh.uv_layers
        if not len(layers):
            return
        index = -1
        if state.uv_active_name:
            index = _layer_index(layers, state.uv_active_name)
        if index < 0:
            index = min(max(state.uv_active_index, 0), len(layers) - 1)
        layers.active_index = index

    _quiet(_restore_active_uv)
    if state.uv_render_name:
        layer = _safe(lambda: mesh.uv_layers.get(state.uv_render_name), None)
        if layer is not None:
            _quiet(lambda: setattr(layer, "active_render", True))

    # 4. Seams, selection and visibility flags.
    for collection, prop, values in (
        (mesh.edges, "use_seam", state.seams),
        (mesh.vertices, "select", state.select_vert),
        (mesh.edges, "select", state.select_edge),
        (mesh.polygons, "select", state.select_poly),
        (mesh.vertices, "hide", state.hide_vert),
        (mesh.edges, "hide", state.hide_edge),
        (mesh.polygons, "hide", state.hide_poly),
    ):
        if values is None or len(collection) != values.shape[0]:
            continue
        _quiet(lambda c=collection, p=prop, v=values: c.foreach_set(p, v))

    # 5. UV selection / pin attributes.
    for name, values in state.uv_attrs:
        attr = _safe(lambda n=name: mesh.attributes.get(n), None)
        if attr is None or _safe(lambda a=attr: len(a.data), -1) != values.shape[0]:
            continue
        _quiet(lambda a=attr, v=values: a.data.foreach_set("value", v))

    _quiet(mesh.update)

    # 6. Tool settings.
    if state.uv_select_sync is not None:
        tool_settings = _safe(lambda: bpy.context.scene.tool_settings, None)
        if tool_settings is not None:
            _quiet(lambda: setattr(tool_settings, "use_uv_select_sync", state.uv_select_sync))

    # 7. Active object and modes, then per-object visibility flags.
    view_layer = _safe(lambda: bpy.context.view_layer, None)
    if view_layer is not None:
        previous = bpy.data.objects.get(state.active_name) if state.active_name else None
        if previous is not None and previous.name in view_layer.objects:
            _quiet(lambda: setattr(view_layer.objects, "active", previous))
            if state.active_mode and state.active_mode != "OBJECT":
                _quiet(lambda: bpy.ops.object.mode_set(mode=state.active_mode))
        elif state.obj_mode != "OBJECT" and obj.name in view_layer.objects:
            _quiet(lambda: setattr(view_layer.objects, "active", obj))
            _quiet(lambda: bpy.ops.object.mode_set(mode=state.obj_mode))

    _quiet(lambda: setattr(obj, "hide_viewport", state.obj_hide_viewport))
    if state.obj_hide_view_layer is not None:
        _quiet(lambda: obj.hide_set(state.obj_hide_view_layer))
    _quiet(lambda: obj.select_set(state.obj_selected))


# ---------------------------------------------------------------------------
# The work itself
# ---------------------------------------------------------------------------


def _seam_mask(req: UnwrapRequest, notes: _Notes) -> np.ndarray | None:
    """Return the ``(E,)`` seam mask to stamp on the mesh, or ``None`` to leave it.

    Projection methods ignore seams entirely, so they get ``None``.  A seam
    strategy that blows up is downgraded to "keep whatever the artist marked"
    with a note, because a missing cut is a worse unwrap, not a failed one.
    """
    params = req.params
    if params.method in PROJECTION_METHODS:
        return None
    strategy = _enum(params.seam_strategy, SEAM_STRATEGIES, "SHARP")
    try:
        mask = compute_seams(
            req.sample,
            strategy,
            angle_deg=float(params.seam_angle_deg),
            curvature_threshold=float(params.curvature_threshold),
            existing=req.sample.edge_seam,
            seed=int(params.seed),
        )
    except BaseException as exc:  # noqa: BLE001 - seams are advisory, never fatal
        notes.add("seam strategy %s failed (%s: %s); kept existing seams",
                  strategy, type(exc).__name__, exc)
        return None
    array = np.asarray(mask, dtype=bool).reshape(-1)
    if array.shape[0] != req.sample.n_edges:
        notes.add("seam mask length %d != %d edges; kept existing seams",
                  array.shape[0], req.sample.n_edges)
        return None
    return array


def _prepare_work_layer(mesh: Any, state: _SceneState, notes: _Notes) -> Any:
    """Create (or borrow) the UV layer the operators will write into.

    Returns the layer, made active.  Normally this is a throwaway
    :data:`WORK_UV_LAYER`.  If the mesh already sits at Blender's eight-layer
    cap, the active layer is borrowed instead and its contents are stashed in
    ``state.uv_backup`` for restoration.

    Raises:
        RuntimeError: If the mesh has no UV layer and none can be created.
    """
    layers = mesh.uv_layers
    stale = layers.get(WORK_UV_LAYER)
    if stale is not None:
        layers.remove(stale)

    layer = None
    if len(layers) < MAX_UV_LAYERS:
        layer = layers.new(name=WORK_UV_LAYER, do_init=False)
    if layer is not None:
        state.work_layer = layer.name
    else:
        layer = layers.active or (layers[0] if len(layers) else None)
        if layer is None:
            raise RuntimeError("mesh has no UV layer and the 8-layer cap blocks creating one")
        state.uv_backup[layer.name] = _read_uv_array(layer, len(mesh.loops))
        notes.add("UV layer cap reached; reused %r in place", layer.name)

    index = _layer_index(layers, layer.name)
    if index >= 0:
        layers.active_index = index
    return layer


def _run_method(bpy: ModuleType, params: UnwrapParams, notes: _Notes) -> bool:
    """Run the operator matching ``params.method``.  Returns whether it finished.

    This function owns the **only** degrees-to-radians conversion in the add-on:
    ``smart_project``'s ``angle_limit`` is in radians while
    :attr:`UnwrapParams.angle_limit_deg` is in degrees.
    """
    method = _enum(params.method, METHODS, "ANGLE_BASED")
    margin_method = _enum(params.margin_method, MARGIN_METHODS, "SCALED")

    if method == "SMART_PROJECT":
        ok, detail = _call_op(
            bpy.ops.uv.smart_project,
            angle_limit=math.radians(float(params.angle_limit_deg)),
            margin_method=margin_method,
            rotate_method=_enum(params.rotate_method, ROTATE_METHODS, "AXIS_ALIGNED_Y"),
            island_margin=float(params.island_margin),
            area_weight=float(params.area_weight),
            correct_aspect=bool(params.correct_aspect),
            scale_to_bounds=bool(params.scale_to_bounds),
        )
    elif method == "CUBE_PROJECT":
        extra = params.extra if isinstance(params.extra, dict) else {}
        ok, detail = _call_op(
            bpy.ops.uv.cube_project,
            cube_size=float(extra.get("cube_size", 1.0)),
            correct_aspect=bool(params.correct_aspect),
            clip_to_bounds=False,
            scale_to_bounds=bool(params.scale_to_bounds),
        )
    else:
        ok, detail = _call_op(
            bpy.ops.uv.unwrap,
            method=method,
            fill_holes=bool(params.fill_holes),
            correct_aspect=bool(params.correct_aspect),
            use_subsurf_data=False,
            margin_method=margin_method,
            margin=float(params.island_margin),
            no_flip=bool(params.no_flip),
            iterations=int(params.iterations),
        )
    if not ok:
        notes.add("%s returned %s", method, detail)
    return ok


def _select_all_uvs(bpy: ModuleType) -> bool:
    """Select every UV of the (already fully selected) mesh.

    Required before packing, averaging island scale or minimising stretch: all
    three operate on the UV *selection*, which a fresh unwrap leaves empty when
    ``use_uv_select_sync`` is off (Blender 4.5's default).
    """
    ok, _ = _call_op(bpy.ops.uv.select_all, action="SELECT")
    return ok


def _pack_shape_method(params: UnwrapParams, n_tris: int, notes: _Notes) -> str:
    """Return the ``shape_method`` to pack with, applying the small-mesh guard.

    See :data:`SMALL_MESH_TRIS` for the benchmark behind the substitution.  It is
    announced in the candidate's message, so a user who wonders why two recipes
    differing only in ``pack_shape_method`` came out identical on a small mesh
    can see exactly why.
    """
    requested = _enum(params.pack_shape_method, PACK_SHAPE_METHODS, "CONCAVE")
    extra = params.extra if isinstance(params.extra, dict) else {}
    if not extra.get("pack_shape_guard", True):
        return requested
    if requested != SMALL_MESH_SHAPE_METHOD and 0 < n_tris < SMALL_MESH_TRIS:
        notes.add("packed as %s: below %d tris Blender's exhaustive packer costs "
                  "seconds without packing any tighter", SMALL_MESH_SHAPE_METHOD,
                  SMALL_MESH_TRIS)
        return SMALL_MESH_SHAPE_METHOD
    return requested


def _pack_islands(bpy: ModuleType, params: UnwrapParams, notes: _Notes,
                  n_tris: int = 0) -> bool:
    """Pack the UV islands, with the 4.5-compatible selection fallback.

    Strategy, in order:

    1. Select every UV, then pack.  This alone fixes Blender 4.5, where packing
       otherwise returns ``CANCELLED`` because nothing is selected.
    2. If that still cancels, force ``use_uv_select_sync`` on -- UV selection
       then mirrors the fully selected mesh -- and pack again.  The flag is
       restored by :func:`_restore_state`.
    3. If packing genuinely cannot run, record a note and carry on: an unpacked
       layout is a worse candidate, not an invalid one.

    ``rotate_method`` comes from :data:`DEFAULT_PACK_ROTATE_METHOD` (see there
    for the measurements behind that choice), overridable per recipe through
    ``params.extra["pack_rotate_method"]``.  It is deliberately *not* taken from
    :attr:`UnwrapParams.rotate_method`, which the parameter space scopes to
    Smart Project only: feeding that in here would let two recipes which hash to
    the same signature produce different UVs, breaking grid de-duplication.
    """
    extra = params.extra if isinstance(params.extra, dict) else {}
    rotate_method = _enum(extra.get("pack_rotate_method", DEFAULT_PACK_ROTATE_METHOD),
                          PACK_ROTATE_METHODS, DEFAULT_PACK_ROTATE_METHOD)
    kwargs: dict[str, Any] = {
        "udim_source": "CLOSEST_UDIM",
        "rotate": bool(params.pack_rotate),
        "rotate_method": rotate_method,
        "scale": True,
        "merge_overlap": False,
        "margin_method": _enum(params.margin_method, MARGIN_METHODS, "SCALED"),
        "margin": float(params.island_margin),
        "pin": False,
        "pin_method": "LOCKED",
        "shape_method": _pack_shape_method(params, n_tris, notes),
    }

    _select_all_uvs(bpy)
    ok, detail = _call_op(bpy.ops.uv.pack_islands, **kwargs)
    if ok:
        return True

    tool_settings = _safe(lambda: bpy.context.scene.tool_settings, None)
    can_force_sync = (
        tool_settings is not None
        and not _safe(lambda: tool_settings.use_uv_select_sync, True)
        and _quiet(lambda: setattr(tool_settings, "use_uv_select_sync", True))
    )
    if can_force_sync:
        ok, detail = _call_op(bpy.ops.uv.pack_islands, **kwargs)
        if ok:
            notes.add("packed via UV sync-selection fallback")
            return True

    notes.add("pack_islands did not run (%s); layout left unpacked", detail)
    return False


def _post_process(bpy: ModuleType, params: UnwrapParams, notes: _Notes,
                  n_tris: int = 0) -> None:
    """Run the optional relax / normalise / pack passes after the main unwrap.

    ``minimize_stretch`` is applied only to the seam-driven solvers, matching
    :meth:`UnwrapParams.signature`, which excludes ``minimize_stretch_iters``
    from a projection recipe's identity.
    """
    _select_all_uvs(bpy)

    iterations = int(params.minimize_stretch_iters)
    if iterations > 0 and params.method in SOLVER_METHODS:
        ok, detail = _call_op(
            bpy.ops.uv.minimize_stretch,
            fill_holes=bool(params.fill_holes),
            blend=0.0,
            iterations=iterations,
        )
        if not ok:
            notes.add("minimize_stretch returned %s", detail)

    if bool(params.average_islands_scale):
        _select_all_uvs(bpy)
        ok, detail = _call_op(bpy.ops.uv.average_islands_scale, scale_uv=False, shear=False)
        if not ok:
            notes.add("average_islands_scale returned %s", detail)

    _pack_islands(bpy, params, notes, n_tris)


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


@register_backend
class BlenderNativeBackend(UnwrapBackend):
    """Unwraps with Blender's own operators -- the default, always-present backend."""

    id: ClassVar[str] = "blender_native"
    label: ClassVar[str] = "Blender Native"
    requires_bpy: ClassVar[bool] = True

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return whether :mod:`bpy` and the UV operators are reachable."""
        bpy = _import_bpy()
        if bpy is None:
            return False, "not running inside Blender (bpy is not importable)"
        if not hasattr(bpy.ops, "uv") or not hasattr(bpy.ops.uv, "unwrap"):
            return False, "bpy.ops.uv.unwrap is missing from this Blender build"
        return True, "Blender %s" % bpy.app.version_string

    def supported_methods(self) -> tuple[str, ...]:
        """Return every method in the parameter space; all of them are native."""
        return tuple(METHODS)

    def unwrap(self, req: UnwrapRequest) -> UnwrapResult:
        """Generate one UV layout with Blender's operators.

        Args:
            req: Object name, recipe and mesh snapshot.

        Returns:
            An :class:`UnwrapResult` whose ``uvs`` is ``(L, 2)`` float32 on
            success.  Never raises; never leaves the scene modified.
        """
        start = time.perf_counter()
        notes = _Notes()

        bpy = _import_bpy()
        if bpy is None:
            return UnwrapResult.failure(
                "bpy is not available; the native backend only runs inside Blender",
                backend=self.id,
                seconds=time.perf_counter() - start,
            )

        obj = bpy.data.objects.get(req.obj_name)
        if obj is None:
            return UnwrapResult.failure(
                "object %r not found" % req.obj_name, backend=self.id,
                seconds=time.perf_counter() - start)
        if obj.type != "MESH":
            return UnwrapResult.failure(
                "object %r is a %s, not a mesh" % (req.obj_name, obj.type),
                backend=self.id, seconds=time.perf_counter() - start)

        mesh = obj.data
        if not len(mesh.polygons) or not len(mesh.loops):
            return UnwrapResult.failure(
                "mesh %r has no faces to unwrap" % mesh.name, backend=self.id,
                seconds=time.perf_counter() - start)

        # _capture_state records the modes, then drops to OBJECT mode itself.
        try:
            state = _capture_state(bpy, obj)
        except BaseException as exc:  # noqa: BLE001
            return UnwrapResult.failure(
                "could not snapshot scene state (%s: %s)" % (type(exc).__name__, exc),
                backend=self.id, seconds=time.perf_counter() - start)

        uvs: np.ndarray | None = None
        error = ""
        try:
            uvs = self._execute(bpy, obj, req, state, notes)
        except BaseException as exc:  # noqa: BLE001 - containment, per the contract
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            _quiet(lambda: _restore_state(bpy, state))

        seconds = time.perf_counter() - start
        if error or uvs is None:
            return UnwrapResult.failure(
                error or "the unwrap produced no UVs", backend=self.id, seconds=seconds)
        if uvs.shape[0] != req.sample.n_loops:
            notes.add("mesh has %d loops but the sample expected %d",
                      uvs.shape[0], req.sample.n_loops)
        return UnwrapResult.success(
            uvs, backend=self.id, seconds=seconds, message=notes.text())

    # -- internals -----------------------------------------------------------

    def _execute(
        self,
        bpy: ModuleType,
        obj: Any,
        req: UnwrapRequest,
        state: _SceneState,
        notes: _Notes,
    ) -> np.ndarray:
        """Do the actual unwrap.  May raise; the caller contains it."""
        mesh = obj.data
        params = req.params

        # -- make the object the operators' target ---------------------------
        view_layer = bpy.context.view_layer
        if obj.name not in view_layer.objects:
            raise RuntimeError("object %r is not in the active view layer" % obj.name)
        _quiet(lambda: setattr(obj, "hide_viewport", False))
        _quiet(lambda: obj.hide_set(False))
        _quiet(lambda: obj.select_set(True))
        view_layer.objects.active = obj
        if obj.mode != "OBJECT":
            _call_op(bpy.ops.object.mode_set, mode="OBJECT")

        # -- reveal everything so the unwrap covers the whole mesh ------------
        # Hidden geometry is skipped by the operators, which would silently
        # leave part of the mesh carrying stale UVs.  Hide flags are in the
        # snapshot, so revealing here is free.
        n_verts, n_edges, n_polys = len(mesh.vertices), len(mesh.edges), len(mesh.polygons)
        if state.hide_poly is not None and state.hide_poly.any():
            notes.add("%d hidden faces were temporarily revealed",
                      int(state.hide_poly.sum()))
        for collection, count in ((mesh.vertices, n_verts), (mesh.edges, n_edges),
                                  (mesh.polygons, n_polys)):
            _quiet(lambda c=collection, n=count: c.foreach_set("hide", np.zeros(n, dtype=bool)))

        # -- seams ------------------------------------------------------------
        mask = _seam_mask(req, notes)
        if mask is not None:
            if mask.shape[0] == n_edges:
                mesh.edges.foreach_set("use_seam", mask)
            else:
                notes.add("seam mask (%d) does not match %d mesh edges; skipped",
                          mask.shape[0], n_edges)
        mesh.update()

        # -- scratch UV layer -------------------------------------------------
        layer = _prepare_work_layer(mesh, state, notes)
        work_name = layer.name

        # -- operators (EDIT mode, full selection) ----------------------------
        ok, detail = _call_op(bpy.ops.object.mode_set, mode="EDIT")
        if not ok:
            raise RuntimeError("could not enter EDIT mode (%s)" % detail)
        try:
            ok, detail = _call_op(bpy.ops.mesh.select_all, action="SELECT")
            if not ok:
                notes.add("mesh.select_all returned %s", detail)
            _run_method(bpy, params, notes)
            _post_process(bpy, params, notes, req.sample.n_tris)
        finally:
            _call_op(bpy.ops.object.mode_set, mode="OBJECT")

        # -- read back --------------------------------------------------------
        layer = mesh.uv_layers.get(work_name)
        if layer is None:
            raise RuntimeError("working UV layer %r vanished during the unwrap" % work_name)
        uvs = _read_uv_array(layer, len(mesh.loops))
        if not np.isfinite(uvs).all():
            bad = int((~np.isfinite(uvs)).any(axis=1).sum())
            notes.add("%d loops came back non-finite and were zeroed", bad)
            uvs = np.nan_to_num(uvs, nan=0.0, posinf=0.0, neginf=0.0)
        return uvs
