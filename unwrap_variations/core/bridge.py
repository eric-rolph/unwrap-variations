"""The only place :mod:`bpy` meets the pure-Python core.

Everything crossing between Blender datablocks and the numpy core goes through
this module, which exists so that the rest of the add-on can be reasoned about
-- and unit-tested -- without Blender running.

Three properties matter here more than anywhere else in the codebase:

**Speed.**  This is the interactive path.  Every read uses the ``foreach_get``
fast paths verified in ``docs/ARCHITECTURE.md`` section 0 and never iterates
mesh elements in Python.  Measured inside Blender on a 250k-face grid
(1,000,000 loops):

===========================================  ==========  ==========
operation                                    4.5.4       5.1.2
===========================================  ==========  ==========
``uv_layer.uv.foreach_set("vector", ...)``   0.4 ms      0.4 ms
``uv_layer.uv.foreach_get("vector", ...)``   0.8 ms      1.3 ms
``uv_layer.data.foreach_set("uv", ...)``     22.4 ms     20.9 ms
``uv_layer.data.foreach_get("uv", ...)``     22.2 ms     16.9 ms
===========================================  ==========  ==========

The legacy ``.data``/``"uv"`` alias is 25x to 50x slower for identical results,
which is why :func:`apply_uvs` insists on the ``.uv``/``"vector"`` collection
and only falls back when a build does not expose it.

**Containment.**  No function here raises into Blender.  A malformed object, a
missing UV layer, a mesh with no loops, a full UV-layer table -- all degrade to
a documented sentinel plus a message, because an exception escaping an operator
or a draw callback is a user-visible crash.

**Reversibility.**  :func:`snapshot_state` / :func:`restore_state` bracket every
destructive generation run.  Restoration is exact: UV layer names *and* order,
their data, per-layer pin and UV-selection flags, seams, element selection,
active and active-render layer, and the object's interaction mode.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

from .meshdata import UVMeshSample

try:  # pragma: no cover - bpy exists only inside Blender.
    import bpy
except Exception:  # noqa: BLE001
    bpy = None  # type: ignore[assignment]

__all__ = [
    "extract_sample",
    "apply_uvs",
    "apply_uvs_ex",
    "ensure_preview_layer",
    "ensure_preview_layer_ex",
    "remove_preview_layer",
    "activate_preview_layer",
    "snapshot_state",
    "restore_state",
    "free_uv_layer_slots",
    "uv_layer_names",
    "mesh_state_digest",
    "UV_LAYER_CAP",
    "PREVIEW_LAYER",
    "HAVE_BPY",
]

_LOG = logging.getLogger("unwrap_variations.bridge")

#: True when running inside Blender.
HAVE_BPY = bpy is not None

#: Hard cap on UV layers per mesh, verified on 4.5.4 and 5.1.2: the ninth
#: ``uv_layers.new()`` returns ``None`` instead of raising.  This single number
#: is the reason candidates are numpy arrays rather than UV layers.
UV_LAYER_CAP = 8

#: The one scratch UV layer the add-on owns.
PREVIEW_LAYER = "UVVar_Preview"

#: Per-UV-layer boolean sidecar accessors, used **only** as a write fallback.
#:
#: Never read through these.  On 4.5.4, merely evaluating ``layer.pin`` (or
#: ``vertex_selection`` / ``edge_selection``) *allocates* the backing
#: ``.pn.<uv>`` / ``.vs.<uv>`` / ``.es.<uv>`` attribute as a side effect.  That
#: is fatal twice over: a supposedly read-only snapshot silently mutates the
#: mesh, and because the new attribute lands in the same corner CustomData
#: block that ``mesh.uv_layers`` indexes into, the allocation happens *during*
#: iteration and every subsequent layer pointer resolves to the wrong slot --
#: which is how ``active_render`` came back False for every layer.
#:
#: Reading goes through :func:`_read_aux_attrs` instead, which walks
#: ``mesh.attributes``, allocates nothing, and is version-agnostic (5.1.2 has no
#: ``vertex_selection`` / ``edge_selection`` at all -- UV selection moved to the
#: mesh-level ``.uv_select_vert`` / ``.uv_select_edge`` / ``.uv_select_face``
#: attributes -- and reports ``len(layer.pin) == 0`` until an operator allocates
#: it, where ``foreach_get`` then raises a length-mismatch ``TypeError``).
_UV_SUBLAYER_FOR_PREFIX = {".pn.": "pin", ".vs.": "vertex_selection", ".es.": "edge_selection"}

#: Per-element boolean state snapshotted alongside the UVs, as
#: ``(collection name, property name)``.
_ELEMENT_FLAGS = (
    ("vertices", "select"),
    ("edges", "select"),
    ("polygons", "select"),
    ("vertices", "hide"),
    ("edges", "hide"),
    ("polygons", "hide"),
)

#: Which generic attribute backs each entry of :data:`_ELEMENT_FLAGS`.
#:
#: Blender allocates these lazily and treats "absent" as "all false".  Writing
#: an all-false array therefore *creates* storage that was not there before --
#: on 5.1.2, restoring untouched hide flags conjured ``.hide_vert``,
#: ``.hide_edge`` and ``.hide_poly`` onto a mesh that had never had them, which
#: is a real difference in the saved file even though every value matched.
_FLAG_BACKING_ATTR = {
    "vertices.select": ".select_vert",
    "edges.select": ".select_edge",
    "polygons.select": ".select_poly",
    "vertices.hide": ".hide_vert",
    "edges.hide": ".hide_edge",
    "polygons.hide": ".hide_poly",
}

#: Mesh attributes that Blender's UV operators create on demand and that are
#: safe to delete again during restore.  Anything not matching one of these is
#: left strictly alone -- removing, say, ``.select_vert`` would destroy real
#: user data.
_REMOVABLE_ATTR_PREFIXES = (".vs.", ".es.", ".pn.", ".uv_select")
_REMOVABLE_ATTR_NAMES = (
    "uv_seam",
    ".hide_vert",
    ".hide_edge",
    ".hide_poly",
    ".select_vert",
    ".select_edge",
    ".select_poly",
)


# ---------------------------------------------------------------------------
# Small guarded helpers.  Every one of these answers "what is actually there?"
# without assuming a well-formed object, because operators get called on
# whatever the user has selected.
# ---------------------------------------------------------------------------


def _mesh_of(obj: Any) -> Any | None:
    """Return ``obj``'s mesh datablock, or ``None`` if it is not a mesh object."""
    if obj is None:
        return None
    if getattr(obj, "type", None) != "MESH":
        return None
    return getattr(obj, "data", None)


def _uv_seq(layer: Any) -> tuple[Any, str] | None:
    """Return ``(collection, property)`` for the fastest UV access on ``layer``.

    Prefers ``layer.uv`` with ``"vector"``: measured 25x-50x faster than the
    legacy ``layer.data`` / ``"uv"`` alias at one million loops, because the
    former is a flat float2 attribute while the latter walks an RNA wrapper per
    element.
    """
    seq = getattr(layer, "uv", None)
    if seq is not None:
        return seq, "vector"
    seq = getattr(layer, "data", None)
    if seq is not None:
        return seq, "uv"
    return None


def _write_sublayer_fallback(mesh: Any, attr_name: str, values: np.ndarray) -> bool:
    """Restore a ``.pn.``/``.vs.``/``.es.`` sidecar through its UV layer accessor.

    Used only when ``mesh.attributes.new()`` refuses the dot-prefixed internal
    name.  Touching the accessor allocates the attribute, which is exactly what
    is wanted here -- unlike on the read path, where it is a bug.

    Returns:
        Whether the values were written.
    """
    prefix = attr_name[:4]
    prop = _UV_SUBLAYER_FOR_PREFIX.get(prefix)
    if prop is None:
        return False
    try:
        layer = mesh.uv_layers.get(attr_name[4:])
        if layer is None:
            return False
        holder = getattr(layer, prop, None)
        if holder is None or len(holder) != values.size:
            return False
        holder.foreach_set("value", np.ascontiguousarray(values, dtype=bool))
        return True
    except Exception:  # noqa: BLE001
        return False


def _set_mode(obj: Any, mode: str) -> bool:
    """Switch ``obj`` into ``mode``, returning whether it succeeded.

    Blender's ``object.mode_set`` acts on the *active* object, so the object is
    made active first.  Mesh data read from Python is only authoritative in
    OBJECT mode -- in EDIT mode the truth lives in the BMesh and ``mesh.loops``
    is a stale copy -- which is why every reader here brackets itself with this.
    """
    if bpy is None or obj is None:
        return False
    if getattr(obj, "mode", "OBJECT") == mode:
        return True
    try:
        view_layer = bpy.context.view_layer
        if view_layer.objects.active is not obj:
            view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode=mode)
        return getattr(obj, "mode", "OBJECT") == mode
    except Exception:  # noqa: BLE001
        _LOG.debug("mode_set(%s) failed for %r", mode, getattr(obj, "name", obj), exc_info=True)
        return False


class _ObjectMode:
    """Context manager that forces OBJECT mode and restores what it found.

    Note on multi-object edit mode: leaving EDIT drops *every* co-edited object
    out of it, and re-entering only brings this one back.  That is an accepted
    limitation -- the alternative is a bmesh round trip that costs far more than
    the whole extraction.
    """

    def __init__(self, obj: Any) -> None:
        """Remember ``obj`` and prepare to restore its mode."""
        self.obj = obj
        self.prev_mode = str(getattr(obj, "mode", "OBJECT") or "OBJECT")
        self.prev_active: Any | None = None
        self.changed = False
        self.ok = True

    def __enter__(self) -> "_ObjectMode":
        """Enter OBJECT mode if needed; set :attr:`ok` False when impossible."""
        if bpy is None:
            self.ok = False
            return self
        if self.prev_mode == "OBJECT":
            return self
        try:
            self.prev_active = bpy.context.view_layer.objects.active
        except Exception:  # noqa: BLE001
            self.prev_active = None
        self.ok = _set_mode(self.obj, "OBJECT")
        self.changed = self.ok
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Restore the original mode and active object.  Never suppresses."""
        if self.changed:
            _set_mode(self.obj, self.prev_mode)
            if self.prev_active is not None and self.prev_active is not self.obj:
                try:
                    bpy.context.view_layer.objects.active = self.prev_active
                except Exception:  # noqa: BLE001
                    pass
        return False


def _empty_sample(name: str) -> UVMeshSample:
    """Return a structurally valid, entirely empty sample.

    Used for every failure path so callers get a real :class:`UVMeshSample` with
    zero elements rather than ``None``; ``validate()`` reports "mesh has no
    triangles" and the evaluator handles it without special-casing.
    """
    return UVMeshSample(
        verts=np.zeros((0, 3), dtype=np.float64),
        tri_verts=np.zeros((0, 3), dtype=np.int32),
        tri_loops=np.zeros((0, 3), dtype=np.int32),
        uvs=np.zeros((0, 2), dtype=np.float64),
        edge_verts=np.zeros((0, 2), dtype=np.int32),
        edge_seam=np.zeros(0, dtype=bool),
        loop_verts=np.zeros(0, dtype=np.int32),
        name=name,
    )


# ---------------------------------------------------------------------------
# Public API (docs/ARCHITECTURE.md section 8)
# ---------------------------------------------------------------------------


def free_uv_layer_slots(obj: Any) -> int:
    """Return how many more UV layers ``obj``'s mesh can hold.

    Blender caps a mesh at :data:`UV_LAYER_CAP` UV layers and reports going over
    by returning ``None`` from ``uv_layers.new()`` -- no exception, no message.
    Callers check this *before* trying, so the add-on can explain the situation
    instead of mysteriously doing nothing.

    Args:
        obj: Any object; non-meshes yield 0.

    Returns:
        ``UV_LAYER_CAP - len(mesh.uv_layers)``, never negative.
    """
    mesh = _mesh_of(obj)
    if mesh is None:
        return 0
    try:
        return max(0, UV_LAYER_CAP - len(mesh.uv_layers))
    except Exception:  # noqa: BLE001
        return 0


def uv_layer_names(obj: Any) -> list[str]:
    """Return the UV layer names of ``obj``'s mesh, in Blender's own order."""
    mesh = _mesh_of(obj)
    return [] if mesh is None else _uv_names(mesh)


def extract_sample(obj: Any, *, uv_layer_name: str | None = None) -> UVMeshSample:
    """Read ``obj`` into a bpy-free :class:`UVMeshSample`.

    The whole extraction is bulk ``foreach_get`` calls -- there is no Python
    loop over any mesh element, because at a million loops that difference is
    seconds versus milliseconds.  ``calc_loop_triangles()`` runs first, since
    ``mesh.loop_triangles`` is otherwise empty or stale.

    A mesh with **zero** UV layers is fully supported and yields zeroed UVs;
    that is the normal state of a freshly imported mesh, which is exactly when
    someone reaches for an unwrapping add-on.

    Works from any mode: the object is briefly forced into OBJECT mode so the
    mesh arrays are authoritative, then put back into whatever mode it was in.

    Args:
        obj: The mesh object to read.
        uv_layer_name: Which UV layer to sample.  ``None`` uses the active one.
            An unknown name falls back to the active layer rather than failing,
            so a renamed layer degrades instead of aborting a generation run.

    Returns:
        A populated sample, or an empty one if ``obj`` is not a usable mesh.
        Never raises.
    """
    name = str(getattr(obj, "name", "") or "")
    mesh = _mesh_of(obj)
    if mesh is None:
        _LOG.debug("extract_sample: %r is not a mesh object", name)
        return _empty_sample(name)

    with _ObjectMode(obj) as guard:
        if not guard.ok:
            _LOG.warning("extract_sample: could not reach OBJECT mode for %r", name)
            return _empty_sample(name)
        try:
            return _extract_locked(mesh, name, uv_layer_name)
        except Exception:  # noqa: BLE001 - a broken mesh must not kill the run.
            _LOG.warning("extract_sample failed for %r", name, exc_info=True)
            return _empty_sample(name)


def _extract_locked(mesh: Any, name: str, uv_layer_name: str | None) -> UVMeshSample:
    """Do the actual bulk read.  Assumes OBJECT mode is already in force."""
    n_verts = len(mesh.vertices)
    n_loops = len(mesh.loops)
    n_edges = len(mesh.edges)

    if n_loops == 0 or n_verts == 0:
        return _empty_sample(name)

    # Triangulation must be requested explicitly; loop_triangles is a cache.
    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)

    # Positions arrive as float32 from Blender; widening afterwards is a single
    # vectorised pass and far cheaper than making RNA convert element by element.
    vbuf = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", vbuf)
    verts = vbuf.reshape(n_verts, 3).astype(np.float64)

    if n_tris:
        tri_loops = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)
        tri_verts = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_verts)
        tri_loops = tri_loops.reshape(n_tris, 3)
        tri_verts = tri_verts.reshape(n_tris, 3)
    else:
        tri_loops = np.zeros((0, 3), dtype=np.int32)
        tri_verts = np.zeros((0, 3), dtype=np.int32)

    loop_verts = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)

    if n_edges:
        edge_verts = np.empty(n_edges * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edge_verts)
        edge_verts = edge_verts.reshape(n_edges, 2)
        edge_seam = np.empty(n_edges, dtype=bool)
        # Works whether or not the underlying ``uv_seam`` attribute has been
        # allocated; Blender reports False for the unallocated case.
        mesh.edges.foreach_get("use_seam", edge_seam)
    else:
        edge_verts = np.zeros((0, 2), dtype=np.int32)
        edge_seam = np.zeros(0, dtype=bool)

    layer = _pick_uv_layer(mesh, uv_layer_name)
    if layer is None:
        uvs = np.zeros((n_loops, 2), dtype=np.float64)
    else:
        access = _uv_seq(layer)
        if access is None:
            uvs = np.zeros((n_loops, 2), dtype=np.float64)
        else:
            seq, prop = access
            ubuf = np.empty(n_loops * 2, dtype=np.float32)
            seq.foreach_get(prop, ubuf)
            uvs = ubuf.reshape(n_loops, 2).astype(np.float64)

    return UVMeshSample(
        verts=verts,
        tri_verts=tri_verts,
        tri_loops=tri_loops,
        uvs=uvs,
        edge_verts=edge_verts,
        edge_seam=edge_seam,
        loop_verts=loop_verts,
        name=name,
    )


def _pick_uv_layer(mesh: Any, uv_layer_name: str | None) -> Any | None:
    """Return the UV layer to sample, or ``None`` when the mesh has none."""
    try:
        layers = mesh.uv_layers
        if not len(layers):
            return None
        if uv_layer_name:
            found = layers.get(uv_layer_name)
            if found is not None:
                return found
            _LOG.debug("uv layer %r not found; using active", uv_layer_name)
        return layers.active or layers[0]
    except Exception:  # noqa: BLE001
        return None


def apply_uvs(obj: Any, uvs: np.ndarray, *, layer_name: str) -> None:
    """Blit ``uvs`` into the named UV layer of ``obj``.

    This is the interactive path -- the whole "instant preview" promise is that
    switching candidate costs one ``foreach_set``.  Measured at **0.4 ms for
    1,000,000 loops** on both 4.5.4 and 5.1.2 (see the module docstring); the
    array is normalised to contiguous float32 first so Blender can memcpy rather
    than convert per element.

    Failures are logged, never raised.  Use :func:`apply_uvs_ex` when the caller
    needs to report the reason.

    Args:
        obj: The mesh object to write into.
        uvs: ``(L, 2)`` or flat ``(2L,)`` array of UVs.
        layer_name: Target UV layer.  Created if absent and a slot is free.
    """
    ok, message = apply_uvs_ex(obj, uvs, layer_name=layer_name)
    if not ok:
        _LOG.warning("apply_uvs: %s", message)


def apply_uvs_ex(obj: Any, uvs: np.ndarray, *, layer_name: str) -> tuple[bool, str]:
    """Same as :func:`apply_uvs` but reports success and a reason.

    The contract fixes :func:`apply_uvs` at ``-> None``, which leaves an
    operator no way to tell the user why nothing happened -- most importantly
    when the UV layer table is full.  This companion supplies that channel
    without changing the contracted signature.

    Returns:
        ``(True, "")`` on success, otherwise ``(False, reason)``.
    """
    mesh = _mesh_of(obj)
    if mesh is None:
        return False, "not a mesh object"
    if uvs is None:
        return False, "no UV data to apply"

    try:
        n_loops = len(mesh.loops)
        flat = np.ascontiguousarray(uvs, dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        return False, "UV array could not be prepared: %s" % exc

    if flat.size != n_loops * 2:
        return False, ("UV array has %d values but the mesh needs %d (topology changed?)"
                       % (flat.size, n_loops * 2))

    with _ObjectMode(obj) as guard:
        if not guard.ok:
            return False, "could not switch to OBJECT mode to write UVs"
        try:
            layer = mesh.uv_layers.get(layer_name)
            if layer is None:
                if free_uv_layer_slots(obj) <= 0:
                    return False, _cap_message(layer_name)
                layer = mesh.uv_layers.new(name=layer_name, do_init=False)
                if layer is None:
                    return False, _cap_message(layer_name)
            access = _uv_seq(layer)
            if access is None:
                return False, "UV layer %r exposes no writable data" % layer_name
            seq, prop = access
            seq.foreach_set(prop, flat)
            # Cheap dirty flag; a full mesh.update() would rebuild derived data
            # we have not touched and is measurably slower on dense meshes.
            mesh.update_tag()
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("apply_uvs failed", exc_info=True)
            return False, "could not write UVs: %s" % exc
    return True, ""


def _cap_message(layer_name: str) -> str:
    """Return the standard explanation for the UV-layer table being full."""
    return (
        "cannot create UV layer %r: this mesh already has the maximum of %d UV layers. "
        "Delete one in Object Data Properties > UV Maps and try again."
        % (layer_name, UV_LAYER_CAP)
    )


def ensure_preview_layer(obj: Any) -> str:
    """Return the name of the scratch preview UV layer, creating it if needed.

    Exactly one scratch layer exists no matter how many candidates are in the
    store -- with a cap of :data:`UV_LAYER_CAP` layers per mesh, one per
    candidate is not an option even for a small grid.

    Degradation, not a crash, when the table is full: the return value is the
    empty string and :func:`ensure_preview_layer_ex` carries the explanation.
    Callers must treat ``""`` as "no preview possible".

    Returns:
        The layer's actual name, or ``""`` when it could not be created.  The
        name Blender handed back is returned rather than the constant, because
        ``uv_layers.new()`` silently appends ``.001`` on a clash instead of
        failing -- a caller that assumed the constant would write into nothing.
    """
    name, message = ensure_preview_layer_ex(obj)
    if not name and message:
        _LOG.warning("ensure_preview_layer: %s", message)
    return name


def ensure_preview_layer_ex(obj: Any) -> tuple[str, str]:
    """Same as :func:`ensure_preview_layer` but also returns a reason on failure.

    Returns:
        ``(name, "")`` on success, or ``("", reason)`` when the layer could not
        be created -- in practice because the mesh already holds
        :data:`UV_LAYER_CAP` UV layers.
    """
    mesh = _mesh_of(obj)
    if mesh is None:
        return "", "not a mesh object"
    try:
        existing = mesh.uv_layers.get(PREVIEW_LAYER)
        if existing is not None:
            return str(existing.name), ""
        if free_uv_layer_slots(obj) <= 0:
            return "", _cap_message(PREVIEW_LAYER)
    except Exception as exc:  # noqa: BLE001
        return "", "could not inspect UV layers: %s" % exc

    with _ObjectMode(obj) as guard:
        if not guard.ok:
            return "", "could not switch to OBJECT mode to create the preview layer"
        try:
            # do_init=False skips copying the active layer's UVs into the new
            # one; every value is overwritten by the first preview anyway.
            layer = mesh.uv_layers.new(name=PREVIEW_LAYER, do_init=False)
        except Exception as exc:  # noqa: BLE001
            return "", "could not create the preview UV layer: %s" % exc
        if layer is None:
            return "", _cap_message(PREVIEW_LAYER)
        # Blender appends ".001" on a name clash rather than failing, so trust
        # the object it handed back instead of the name that was asked for.
        return str(layer.name), ""


def remove_preview_layer(obj: Any) -> None:
    """Delete the scratch preview UV layer, if it exists.

    Also frees the layer's boolean sidecar attributes (``.pn.<name>`` and, on
    4.5, ``.vs.``/``.es.<name>``) when Blender leaves them behind, so a Clear
    really does return the mesh to its original attribute set.
    """
    mesh = _mesh_of(obj)
    if mesh is None:
        return
    with _ObjectMode(obj) as guard:
        if not guard.ok:
            return
        _remove_uv_layer(mesh, PREVIEW_LAYER)
        _drop_attributes(
            mesh,
            [
                n
                for n in _attribute_names(mesh)
                if n.endswith("." + PREVIEW_LAYER) and n.startswith(_REMOVABLE_ATTR_PREFIXES)
            ],
        )
        try:
            mesh.update_tag()
        except Exception:  # noqa: BLE001
            pass


def activate_preview_layer(obj: Any, *, render: bool = True) -> bool:
    """Make the preview layer the one the viewport and renderer display.

    Separate from :func:`ensure_preview_layer` because creating the scratch
    layer and *taking over what the user sees* are different decisions: an
    operator that only wants a place to write should not hijack the display.

    Args:
        obj: The mesh object.
        render: Also set ``active_render``, which is what shading nodes without
            an explicit UV Map node actually sample.

    Returns:
        Whether the preview layer exists and is now active.
    """
    mesh = _mesh_of(obj)
    if mesh is None:
        return False
    try:
        layer = mesh.uv_layers.get(PREVIEW_LAYER)
        if layer is None:
            return False
        mesh.uv_layers.active = layer
        if render:
            layer.active_render = True
        mesh.update_tag()
        return True
    except Exception:  # noqa: BLE001
        _LOG.debug("could not activate preview layer", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


def snapshot_state(obj: Any) -> dict:
    """Capture every piece of mesh state a generation run can disturb.

    Non-destructive-by-default is the product's core claim, and it is only true
    if this is complete.  Captured: each UV layer's name, order, data and
    boolean sidecars; the active and active-render UV layer; edge seams;
    vertex/edge/face selection and hide flags; the mesh's attribute name list
    (so operator-created sidecars can be removed again); and the object's mode.

    The returned dict holds live numpy arrays and is a *runtime* structure --
    it is never persisted, so it can be as fat and as fast as it likes.

    Args:
        obj: The mesh object to snapshot.

    Returns:
        An opaque dict for :func:`restore_state`.  Empty (falsy) when ``obj``
        is not a usable mesh, which :func:`restore_state` then ignores.
    """
    mesh = _mesh_of(obj)
    if mesh is None:
        return {}

    snap: dict[str, Any] = {
        "version": 1,
        "obj_name": str(getattr(obj, "name", "") or ""),
        "mode": str(getattr(obj, "mode", "OBJECT") or "OBJECT"),
    }

    with _ObjectMode(obj) as guard:
        if not guard.ok:
            _LOG.warning("snapshot_state: could not reach OBJECT mode")
            return {}
        try:
            n_loops = len(mesh.loops)
            n_edges = len(mesh.edges)
            snap["loop_count"] = n_loops
            snap["edge_count"] = n_edges
            snap["vert_count"] = len(mesh.vertices)
            snap["poly_count"] = len(mesh.polygons)

            layers = []
            active_index = -1
            active_render_index = -1
            # Resolve each layer by name rather than holding the pointers from a
            # single iteration; see _remove_uv_layer for why cached
            # MeshUVLoopLayer pointers are unsafe on 4.5.
            for i, layer_name in enumerate(_uv_names(mesh)):
                layer = mesh.uv_layers.get(layer_name)
                if layer is None:
                    continue
                entry: dict[str, Any] = {"name": layer_name}
                access = _uv_seq(layer)
                if access is not None and n_loops:
                    seq, prop = access
                    buf = np.empty(n_loops * 2, dtype=np.float32)
                    seq.foreach_get(prop, buf)
                    entry["uv"] = buf
                else:
                    entry["uv"] = np.zeros(n_loops * 2, dtype=np.float32)
                if getattr(layer, "active_render", False):
                    active_render_index = i
                layers.append(entry)
            try:
                active_index = int(mesh.uv_layers.active_index)
            except Exception:  # noqa: BLE001
                active_index = -1
            snap["uv_layers"] = layers
            snap["uv_active_index"] = active_index
            snap["uv_active_render_index"] = active_render_index

            if n_edges:
                seams = np.empty(n_edges, dtype=bool)
                mesh.edges.foreach_get("use_seam", seams)
                snap["edge_seam"] = seams
            else:
                snap["edge_seam"] = np.zeros(0, dtype=bool)

            flags: dict[str, np.ndarray] = {}
            for coll_name, prop in _ELEMENT_FLAGS:
                coll = getattr(mesh, coll_name, None)
                if coll is None:
                    continue
                try:
                    count = len(coll)
                    buf = np.empty(count, dtype=bool)
                    coll.foreach_get(prop, buf)
                    flags["%s.%s" % (coll_name, prop)] = buf
                except Exception:  # noqa: BLE001
                    _LOG.debug("could not snapshot %s.%s", coll_name, prop, exc_info=True)
            snap["flags"] = flags
            snap["attr_names"] = _attribute_names(mesh)
            snap["aux_attrs"] = _read_aux_attrs(mesh)
        except Exception:  # noqa: BLE001
            _LOG.warning("snapshot_state failed", exc_info=True)
            return {}
    return snap


def restore_state(obj: Any, snap: dict) -> None:
    """Put ``obj`` back exactly as :func:`snapshot_state` found it.

    Called from a ``finally`` around every generation run, so it must cope with
    a half-finished, arbitrarily mangled mesh and must not raise on the way out
    -- an exception here would mask whatever real error triggered the unwind.

    Restoration is exact rather than best-effort.  When the UV layer *names and
    order* still match the snapshot the data is written back in place (a few
    memcpys); when they do not -- an operator added, removed or renamed a layer
    -- every UV layer is dropped and rebuilt in the snapshot's order, because
    Blender offers no way to reorder them and order is user-visible in exporters
    and in the UV map list.

    Operator-created sidecar attributes (pin, UV selection, ``uv_seam``) that
    were not present at snapshot time are removed, so the mesh's attribute list
    matches too.

    Args:
        obj: The mesh object to restore.
        snap: A dict from :func:`snapshot_state`.  Empty or foreign dicts are
            ignored silently.
    """
    if not snap or not isinstance(snap, dict):
        return
    mesh = _mesh_of(obj)
    if mesh is None:
        return

    target_mode = str(snap.get("mode", "OBJECT") or "OBJECT")
    if not _set_mode(obj, "OBJECT"):
        _LOG.warning("restore_state: could not reach OBJECT mode; mesh left as is")
        return

    try:
        n_loops = len(mesh.loops)
        if n_loops != int(snap.get("loop_count", n_loops)):
            # The caller changed topology on purpose (or a backend went rogue).
            # Restoring UVs by index would scramble the mesh, so only the parts
            # that are still meaningful are put back.
            _LOG.warning(
                "restore_state: loop count changed %s -> %d; restoring seams only",
                snap.get("loop_count"),
                n_loops,
            )
            _restore_seams(mesh, snap)
            _restore_flags(mesh, snap)
            return
        _restore_uv_layers(mesh, snap, n_loops)
        _restore_seams(mesh, snap)
        _restore_flags(mesh, snap)
        # After the layer table is final, because rebuilding UV layers discards
        # their sidecars; before dropping extras, because that is the step that
        # makes the attribute list match again.
        _write_aux_attrs(mesh, snap.get("aux_attrs") or {})
        _drop_new_attributes(mesh, snap)
        try:
            mesh.update_tag()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        _LOG.warning("restore_state failed", exc_info=True)
    finally:
        _set_mode(obj, target_mode)


def _uv_names(mesh: Any) -> list[str]:
    """Return the current UV layer names, or an empty list on failure."""
    try:
        return [str(layer.name) for layer in mesh.uv_layers]
    except Exception:  # noqa: BLE001
        return []


def _remove_uv_layer(mesh: Any, name: str) -> None:
    """Remove one UV layer by name, resolving the pointer immediately before use.

    Never cache a ``MeshUVLoopLayer`` across a removal.  On 4.5.4 those RNA
    pointers are positional into the raw corner CustomData block -- which also
    holds the ``.vs.`` / ``.es.`` / ``.pn.`` boolean sidecars -- so once one
    layer goes, a previously collected pointer silently re-resolves to a
    different slot.  Removing from a ``list(mesh.uv_layers)`` therefore deletes
    the wrong layers and eventually raises ``UV map '.pn.X' not found``.
    Re-resolving by name (or by index 0) each time is the only safe pattern.
    """
    try:
        layer = mesh.uv_layers.get(name)
        if layer is not None:
            mesh.uv_layers.remove(layer)
    except Exception:  # noqa: BLE001
        _LOG.debug("could not remove UV layer %s", name, exc_info=True)


def _remove_all_uv_layers(mesh: Any) -> None:
    """Drop every UV layer, always re-resolving index 0 (see :func:`_remove_uv_layer`)."""
    guard = 0
    try:
        while len(mesh.uv_layers) and guard <= UV_LAYER_CAP + 2:
            guard += 1
            mesh.uv_layers.remove(mesh.uv_layers[0])
    except Exception:  # noqa: BLE001
        _LOG.warning("could not clear the UV layer table", exc_info=True)


def _restore_uv_layers(mesh: Any, snap: dict, n_loops: int) -> None:
    """Reconcile the UV layer table with the snapshot, then refill it.

    Repaired incrementally first -- extra layers removed, missing ones re-added
    -- because that leaves untouched layers and their sidecar attributes exactly
    as they were.  Only if the resulting *order* still differs is the table
    rebuilt wholesale, since Blender has no reorder API and layer order is
    user-visible in exporters and in the UV Maps list.
    """
    entries: list[dict[str, Any]] = list(snap.get("uv_layers") or [])
    want = [str(e["name"]) for e in entries]

    if _uv_names(mesh) != want:
        for name in [n for n in _uv_names(mesh) if n not in want]:
            _remove_uv_layer(mesh, name)
        for name in want:
            try:
                if mesh.uv_layers.get(name) is None and len(mesh.uv_layers) < UV_LAYER_CAP:
                    mesh.uv_layers.new(name=name, do_init=False)
            except Exception:  # noqa: BLE001
                _LOG.debug("could not re-add UV layer %s", name, exc_info=True)
        if _uv_names(mesh) != want:
            _remove_all_uv_layers(mesh)
            for name in want[:UV_LAYER_CAP]:
                try:
                    mesh.uv_layers.new(name=name, do_init=False)
                except Exception:  # noqa: BLE001
                    _LOG.warning("could not recreate UV layer %s", name, exc_info=True)

    for entry in entries:
        layer = mesh.uv_layers.get(str(entry["name"]))
        if layer is None:
            continue
        access = _uv_seq(layer)
        data = entry.get("uv")
        if access is not None and data is not None and data.size == n_loops * 2:
            seq, prop = access
            try:
                seq.foreach_set(prop, data)
            except Exception:  # noqa: BLE001
                _LOG.debug("could not restore UVs for %s", entry["name"], exc_info=True)

    try:
        idx = int(snap.get("uv_active_index", -1))
        if 0 <= idx < len(mesh.uv_layers):
            mesh.uv_layers.active_index = idx
    except Exception:  # noqa: BLE001
        _LOG.debug("could not restore active UV index", exc_info=True)

    try:
        ridx = int(snap.get("uv_active_render_index", -1))
        if 0 <= ridx < len(mesh.uv_layers):
            # Assigning True clears the flag on every other layer, which is the
            # behaviour we want; there is no way to have two render layers.
            mesh.uv_layers[ridx].active_render = True
    except Exception:  # noqa: BLE001
        _LOG.debug("could not restore active_render", exc_info=True)


def _restore_seams(mesh: Any, snap: dict) -> None:
    """Write the snapshotted seam flags back onto the edges.

    Skipped entirely when the mesh had no ``uv_seam`` attribute and no seam was
    set, for the same allocate-on-write reason as :func:`_restore_flags`.
    """
    seams = snap.get("edge_seam")
    if seams is None:
        return
    if "uv_seam" not in set(snap.get("attr_names") or []) and not seams.any():
        return
    try:
        if len(mesh.edges) != seams.size:
            return
        mesh.edges.foreach_set("use_seam", np.ascontiguousarray(seams, dtype=bool))
    except Exception:  # noqa: BLE001
        _LOG.debug("could not restore seams", exc_info=True)


def _restore_flags(mesh: Any, snap: dict) -> None:
    """Write back the per-element selection and hide flags.

    An all-false array whose backing attribute did not exist at snapshot time is
    skipped rather than written: the write would allocate the attribute, leaving
    the mesh carrying storage it never had (see :data:`_FLAG_BACKING_ATTR`).
    Skipping is not merely tidier, it is what makes restoration exact.
    """
    flags: dict[str, np.ndarray] = snap.get("flags") or {}
    before = set(snap.get("attr_names") or [])
    for key, values in flags.items():
        coll_name, _, prop = key.partition(".")
        coll = getattr(mesh, coll_name, None)
        if coll is None:
            continue
        backing = _FLAG_BACKING_ATTR.get(key)
        if backing and backing not in before and not values.any():
            continue
        try:
            if len(coll) != values.size:
                continue
            coll.foreach_set(prop, np.ascontiguousarray(values, dtype=bool))
        except Exception:  # noqa: BLE001
            _LOG.debug("could not restore %s", key, exc_info=True)


def _attribute_names(mesh: Any) -> list[str]:
    """Return the mesh's generic attribute names, or an empty list on old builds."""
    try:
        return [str(a.name) for a in mesh.attributes]
    except Exception:  # noqa: BLE001
        return []


def _is_uv_aux_attr(name: str) -> bool:
    """Return whether ``name`` is a UV-related boolean sidecar attribute."""
    return name.startswith(_REMOVABLE_ATTR_PREFIXES) or name in _REMOVABLE_ATTR_NAMES


def _read_aux_attrs(mesh: Any) -> dict[str, tuple[str, str, np.ndarray]]:
    """Read the values of every UV-related sidecar attribute.

    Necessary because the two supported Blender versions keep UV selection in
    different places: 4.5 hangs ``.vs.<uv>`` / ``.es.<uv>`` off each UV layer,
    while 5.1 promoted them to mesh-level ``.uv_select_vert`` /
    ``.uv_select_edge`` / ``.uv_select_face``.  Going through the generic
    attribute API covers both without version branching, and catches anything a
    future release adds under the same naming convention.

    Returns:
        ``{name: (domain, data_type, values)}`` for boolean sidecars only.
    """
    out: dict[str, tuple[str, str, np.ndarray]] = {}
    try:
        attrs = list(mesh.attributes)
    except Exception:  # noqa: BLE001
        return out
    for attr in attrs:
        try:
            name = str(attr.name)
            if not _is_uv_aux_attr(name) or attr.data_type != "BOOLEAN":
                continue
            count = len(attr.data)
            buf = np.empty(count, dtype=bool)
            attr.data.foreach_get("value", buf)
            out[name] = (str(attr.domain), str(attr.data_type), buf)
        except Exception:  # noqa: BLE001
            _LOG.debug("could not read aux attribute", exc_info=True)
    return out


def _write_aux_attrs(mesh: Any, state: dict[str, tuple[str, str, np.ndarray]]) -> None:
    """Write UV sidecar attribute values back, recreating missing ones."""
    for name, (domain, data_type, values) in (state or {}).items():
        try:
            attr = mesh.attributes.get(name)
            if attr is None:
                # Internal names begin with a dot and some builds refuse to
                # create them through the public API; the UV layer accessor then
                # allocates the same attribute by another route.
                try:
                    attr = mesh.attributes.new(name=name, type=data_type, domain=domain)
                except Exception:  # noqa: BLE001
                    attr = None
                if attr is None or str(getattr(attr, "name", "")) != name:
                    if _write_sublayer_fallback(mesh, name, values):
                        continue
                    attr = mesh.attributes.get(name)
            if attr is None or len(attr.data) != values.size:
                continue
            attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=bool))
        except Exception:  # noqa: BLE001
            _LOG.debug("could not restore aux attribute %s", name, exc_info=True)


def _drop_attributes(mesh: Any, names: Iterable[str]) -> None:
    """Remove the named attributes, ignoring any that refuse to go."""
    for name in list(names):
        try:
            attr = mesh.attributes.get(name)
            if attr is not None:
                mesh.attributes.remove(attr)
        except Exception:  # noqa: BLE001
            _LOG.debug("could not remove attribute %s", name, exc_info=True)


def _drop_new_attributes(mesh: Any, snap: dict) -> None:
    """Remove UV sidecar attributes that appeared since the snapshot.

    Blender's UV operators allocate ``.pn.<uv>`` (pin), ``.vs.``/``.es.<uv>``
    (4.5 UV selection), ``.uv_select_*`` (5.1 UV selection) and ``uv_seam`` the
    first time they are needed and never free them.  Left behind, the mesh's
    attribute list differs from before the run even though every value matches,
    which is a real difference in the saved file.  Only these known-safe names
    are touched -- deleting anything else could destroy user data.
    """
    before = set(snap.get("attr_names") or [])
    if not before:
        return
    now = _attribute_names(mesh)
    doomed = [
        name
        for name in now
        if name not in before
        and (name.startswith(_REMOVABLE_ATTR_PREFIXES) or name in _REMOVABLE_ATTR_NAMES)
    ]
    _drop_attributes(mesh, doomed)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def mesh_state_digest(obj: Any) -> str:
    """Return a hash over everything :func:`snapshot_state` claims to restore.

    Exists so the round-trip guarantee is *checkable* rather than asserted: take
    a digest, run something destructive, restore, take another digest, compare.
    The add-on's own tests use it, and it is cheap enough for an operator to use
    as a paranoid post-condition.

    Unallocated boolean sidecars hash identically to allocated all-false ones,
    because Blender treats the two as the same state and only differs in whether
    it has bothered to allocate memory.

    Returns:
        A hex sha256 digest, or ``""`` when ``obj`` is not a usable mesh.
    """
    import hashlib

    mesh = _mesh_of(obj)
    if mesh is None:
        return ""
    digest = hashlib.sha256()

    def feed(tag: str, arr: np.ndarray | None) -> None:
        digest.update(tag.encode("ascii"))
        if arr is None:
            digest.update(b"<none>")
            return
        contiguous = np.ascontiguousarray(arr)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())

    with _ObjectMode(obj) as guard:
        if not guard.ok:
            return ""
        try:
            n_loops = len(mesh.loops)
            n_edges = len(mesh.edges)
            digest.update(("mode=%s" % getattr(obj, "mode", "?")).encode("ascii"))
            digest.update(("counts=%d,%d,%d,%d" % (
                len(mesh.vertices), n_edges, len(mesh.polygons), n_loops)).encode("ascii"))

            # Read-only, and it must stay that way: the per-layer pin and
            # UV-selection accessors allocate on access (see
            # _UV_SUBLAYER_FOR_PREFIX), so the sidecars are hashed from
            # mesh.attributes at the end instead.
            for i, layer_name in enumerate(_uv_names(mesh)):
                layer = mesh.uv_layers.get(layer_name)
                if layer is None:
                    continue
                digest.update(("layer%d=%s" % (i, layer_name)).encode("utf-8", "replace"))
                digest.update(b"render1" if getattr(layer, "active_render", False) else b"render0")
                access = _uv_seq(layer)
                if access is not None and n_loops:
                    seq, prop = access
                    buf = np.empty(n_loops * 2, dtype=np.float32)
                    seq.foreach_get(prop, buf)
                    feed("uv", buf)
            try:
                digest.update(("active=%d" % int(mesh.uv_layers.active_index)).encode("ascii"))
            except Exception:  # noqa: BLE001
                digest.update(b"active=?")

            if n_edges:
                seams = np.empty(n_edges, dtype=bool)
                mesh.edges.foreach_get("use_seam", seams)
                feed("seam", seams)
            for coll_name, prop in _ELEMENT_FLAGS:
                coll = getattr(mesh, coll_name, None)
                if coll is None:
                    continue
                buf = np.empty(len(coll), dtype=bool)
                coll.foreach_get(prop, buf)
                feed("%s.%s" % (coll_name, prop), buf)

            vbuf = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", vbuf)
            feed("co", vbuf)
            lbuf = np.empty(n_loops, dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", lbuf)
            feed("loopvert", lbuf)
            digest.update(("attrs=%s" % ",".join(sorted(_attribute_names(mesh)))).encode("utf-8"))
            for name, (domain, data_type, values) in sorted(_read_aux_attrs(mesh).items()):
                feed("aux:%s:%s:%s" % (name, domain, data_type), values)
        except Exception:  # noqa: BLE001
            _LOG.debug("mesh_state_digest failed", exc_info=True)
            return ""
    return digest.hexdigest()
