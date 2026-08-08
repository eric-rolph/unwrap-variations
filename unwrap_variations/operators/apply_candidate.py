"""``uvvar.apply_candidate`` -- commit a candidate into a real UV map.

Everything up to this point is non-destructive: candidates live as numpy arrays
and are shown through the single scratch layer ``UVVar_Preview``.  Apply is the
one operation that writes into the user's actual data, so it is the one that
asks first.

Two decisions the user makes here, and the trap behind the second one:

* **Overwrite or create.**  Overwriting replaces an existing UV map in place --
  right when iterating on a model that is already textured.  Creating adds a
  new map, which keeps the original as a fallback.
* **The eight-layer cap.**  Verified on 4.5.4 and 5.1.2: ``uv_layers.new()``
  returns ``None`` past eight maps and Blender logs "Cannot add more than 8 UV
  maps" to the console, where nobody sees it.  Silently doing nothing is the
  worst possible outcome for an operator whose entire job is to write data, so
  the cap is checked up front and reported as a proper error -- with the one
  genuinely useful remedy (reclaiming the preview layer's slot) offered rather
  than merely described.
"""

from __future__ import annotations

from typing import Any

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Context, Event, Operator

from .. import properties as uvvar_props
from ..utils import compat
from ..utils import logging as uvvar_logging
from .select_candidate import PREVIEW_LAYER, _resolve_store_index

__all__ = [
    "UVVAR_OT_apply_candidate",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)

#: Identifier used when a mesh has no overwritable UV map.  An enum may never
#: be empty, so a placeholder is offered and rejected in ``execute``.
_NO_LAYER = "__UVVAR_NONE__"

#: Backing store for the dynamic layer enum.  Blender does not copy the strings
#: an ``items`` callback returns; keeping them alive here is the standard fix
#: for the "enum shows garbage" bug.
_LAYER_ITEMS: list[tuple[str, str, str]] = [(_NO_LAYER, "No UV Map", "This mesh has no UV map yet")]


def _layer_items(_self: Any, context: Context | None) -> list[tuple[str, str, str]]:
    """Return the overwritable UV maps of the active mesh, preview excluded."""
    items: list[tuple[str, str, str]] = []
    obj = compat.active_mesh_object(context)
    if obj is not None:
        try:
            for layer in obj.data.uv_layers:
                if layer.name == PREVIEW_LAYER:
                    continue
                items.append((layer.name, layer.name, "Overwrite the UV map %r" % layer.name))
        except Exception:  # noqa: BLE001 - a broken mesh must not break the dialog.
            items = []
    if not items:
        items = [(_NO_LAYER, "No UV Map", "This mesh has no UV map to overwrite")]
    _LAYER_ITEMS[:] = items
    return _LAYER_ITEMS


class UVVAR_OT_apply_candidate(Operator):
    """Write the selected candidate into a real UV map."""

    bl_idname = "uvvar.apply_candidate"
    bl_label = "Apply Candidate"
    bl_description = "Commit the selected variation into a real UV map on the mesh"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(
        name="Index",
        description="Row to apply; -1 uses the scene's active candidate",
        default=-1,
        min=-1,
    )

    store_index: IntProperty(
        name="Store Index",
        description="Address the candidate store directly, bypassing list order; -1 to ignore",
        default=-1,
        min=-1,
    )

    mode: EnumProperty(
        name="Target",
        description="Where to write the UVs",
        items=(
            ("NEW", "New UV Map", "Create a new UV map and leave the existing ones untouched"),
            ("OVERWRITE", "Overwrite", "Replace the contents of an existing UV map"),
        ),
        default="NEW",
    )

    new_name: StringProperty(
        name="Name",
        description="Name for the new UV map",
        default="UVMap_Variation",
        maxlen=59,
    )

    target_layer: EnumProperty(
        name="UV Map",
        description="Which existing UV map to overwrite",
        items=_layer_items,
    )

    make_active: BoolProperty(
        name="Make Active",
        description="Select the written UV map for editing and for rendering",
        default=True,
    )

    remove_preview: BoolProperty(
        name="Remove Preview Layer",
        description=(
            "Delete the scratch UVVar_Preview layer afterwards. Also frees its "
            "slot, which matters on a mesh that is near the 8 UV map limit"
        ),
        default=True,
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled when the active mesh object has a candidate to commit."""
        obj = compat.active_mesh_object(context)
        if obj is None:
            return False
        store = uvvar_props.get_store(obj)
        if store is not None and len(store):
            return True
        props = uvvar_props.get_props(context)
        return bool(props is not None and len(props.candidates))

    def invoke(self, context: Context, _event: Event) -> set[str]:
        """Ask the user where to write before touching anything."""
        obj = compat.active_mesh_object(context)
        if obj is not None:
            try:
                candidate = self._resolve_candidate(context, obj)
            except Exception:  # noqa: BLE001
                candidate = None
            if candidate is not None:
                suffix = ""
                try:
                    suffix = candidate.params.describe().split(" ")[0]
                except Exception:  # noqa: BLE001
                    suffix = ""
                base = "UVMap_%s" % (suffix or "Variation")
                self.new_name = compat.unique_uv_layer_name(obj.data, base)
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context: Context) -> None:
        """Draw the confirmation dialog."""
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "mode", expand=True)
        if self.mode == "NEW":
            layout.prop(self, "new_name")
            obj = compat.active_mesh_object(context)
            free = compat.free_uv_layer_slots(obj)
            if free <= 0:
                box = layout.box()
                box.alert = True
                box.label(text="No free UV map slot (Blender allows 8)", icon="ERROR")
                if compat.uv_layer_by_name(getattr(obj, "data", None), PREVIEW_LAYER) is not None:
                    box.label(text="Enable Remove Preview Layer to free one", icon="INFO")
                else:
                    box.label(text="Delete a UV map, or choose Overwrite", icon="INFO")
            elif free == 1:
                layout.label(text="1 UV map slot left", icon="INFO")
        else:
            layout.prop(self, "target_layer")

        col = layout.column(align=True)
        col.prop(self, "make_active")
        col.prop(self, "remove_preview")

    def execute(self, context: Context) -> set[str]:
        """Write the candidate into the chosen UV map."""
        try:
            return self._execute(context)
        except Exception as exc:  # noqa: BLE001 - nothing escapes into Blender.
            _LOG.exception("apply_candidate failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}

    # -- implementation ------------------------------------------------------

    def _execute(self, context: Context) -> set[str]:
        """Do the work; every failure returns a reported ``CANCELLED``."""
        obj = compat.active_mesh_object(context)
        if obj is None:
            self.report({"ERROR"}, "No active mesh object")
            return {"CANCELLED"}

        candidate = self._resolve_candidate(context, obj)
        if candidate is None:
            self.report({"ERROR"}, "No candidate selected")
            return {"CANCELLED"}
        if candidate.uvs is None:
            self.report(
                {"ERROR"},
                uvvar_logging.ascii_safe(
                    "Candidate %d has no UVs (%s)"
                    % (int(candidate.index), candidate.message or "the backend failed")
                ),
            )
            return {"CANCELLED"}

        mesh = obj.data
        loops = len(mesh.loops)
        if int(candidate.uvs.shape[0]) != loops:
            self.report(
                {"ERROR"},
                "Candidate was built for %d loops but the mesh now has %d; run Generate again"
                % (int(candidate.uvs.shape[0]), loops),
            )
            return {"CANCELLED"}

        # Reclaiming the preview slot has to happen *before* the cap check,
        # because it is the difference between "no room" and "room".
        freed_preview = False
        if self.mode == "NEW" and self.remove_preview and compat.free_uv_layer_slots(obj) <= 0:
            freed_preview = self._remove_preview_layer(obj)

        layer, error = self._target_uv_layer(obj)
        if layer is None:
            if freed_preview:
                error += " (the preview layer was removed to try to make room)"
            self.report({"ERROR"}, uvvar_logging.ascii_safe(error))
            return {"CANCELLED"}

        layer_name = layer.name
        try:
            from ..core import bridge

            bridge.apply_uvs(obj, candidate.uvs, layer_name=layer_name)
        except Exception:  # noqa: BLE001 - the bridge is optional; the write is not.
            _LOG.debug("core.bridge.apply_uvs unavailable; writing directly", exc_info=True)
            try:
                compat.write_uv_layer(layer, candidate.uvs)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, uvvar_logging.ascii_safe(
                    "Could not write UV map %r: %s" % (layer_name, exc)))
                return {"CANCELLED"}

        if self.remove_preview and not freed_preview:
            self._remove_preview_layer(obj)

        # Re-fetch by name: removing a layer invalidates existing references.
        written = compat.uv_layer_by_name(mesh, layer_name)
        if written is not None and self.make_active:
            try:
                mesh.uv_layers.active = written
                written.active_render = True
            except Exception:  # noqa: BLE001 - a failed activation is not a failed apply.
                _LOG.debug("could not activate UV map %r", layer_name, exc_info=True)

        try:
            mesh.update()
        except Exception:  # noqa: BLE001
            pass
        compat.tag_redraw(context)

        label = ""
        try:
            label = candidate.params.describe()
        except Exception:  # noqa: BLE001
            pass
        message = "Applied candidate %d (%s) to UV map %r" % (
            int(candidate.index), label, layer_name)
        props = uvvar_props.get_props(context)
        if props is not None:
            try:
                props.status = message[:220]
            except Exception:  # noqa: BLE001
                pass
        self.report({"INFO"}, uvvar_logging.ascii_safe(message))
        _LOG.info("%s", message)
        return {"FINISHED"}

    def _resolve_candidate(self, context: Context, obj: Any) -> Any:
        """Return the candidate to apply, or ``None``."""
        store = uvvar_props.store_for(obj, create=False)
        if store is None or not len(store):
            return None
        if int(self.store_index) >= 0:
            return store.get(int(self.store_index))
        props = uvvar_props.get_props(context)
        row = int(self.index)
        if row < 0:
            row = int(getattr(props, "active_index", 0) or 0)
        return store.get(_resolve_store_index(props, row))

    def _target_uv_layer(self, obj: Any) -> tuple[Any, str]:
        """Return the UV layer to write into, or ``(None, reason)``."""
        mesh = obj.data
        if self.mode == "OVERWRITE":
            name = str(self.target_layer or "")
            if not name or name == _NO_LAYER:
                return None, (
                    "mesh %r has no UV map to overwrite; choose New UV Map instead" % mesh.name
                )
            layer = compat.uv_layer_by_name(mesh, name)
            if layer is None:
                return None, "UV map %r no longer exists on %r" % (name, mesh.name)
            if layer.name == PREVIEW_LAYER:
                return None, (
                    "%r is the scratch preview layer; overwriting it would not commit anything"
                    % PREVIEW_LAYER
                )
            return layer, ""

        if compat.free_uv_layer_slots(obj) <= 0:
            return None, (
                "mesh %r already has %d UV maps, which is Blender's hard limit; "
                "delete one or choose Overwrite" % (mesh.name, compat.MAX_UV_LAYERS)
            )
        name = compat.unique_uv_layer_name(mesh, self.new_name or "UVMap_Variation")
        try:
            layer = mesh.uv_layers.new(name=name, do_init=False)
        except TypeError:
            layer = mesh.uv_layers.new(name=name)
        except Exception as exc:  # noqa: BLE001
            return None, "could not create UV map %r: %s" % (name, exc)
        if layer is None:
            return None, (
                "Blender refused to create UV map %r; a mesh may hold at most %d"
                % (name, compat.MAX_UV_LAYERS)
            )
        return layer, ""

    @staticmethod
    def _remove_preview_layer(obj: Any) -> bool:
        """Delete the scratch preview layer; return whether one was removed."""
        try:
            from ..core import bridge

            if compat.uv_layer_by_name(obj.data, PREVIEW_LAYER) is None:
                return False
            bridge.remove_preview_layer(obj)
            return compat.uv_layer_by_name(obj.data, PREVIEW_LAYER) is None
        except Exception:  # noqa: BLE001 - fall through to the direct path.
            pass
        layer = compat.uv_layer_by_name(obj.data, PREVIEW_LAYER)
        if layer is None:
            return False
        try:
            obj.data.uv_layers.remove(layer)
            return True
        except Exception:  # noqa: BLE001
            _LOG.debug("could not remove the preview layer", exc_info=True)
            return False


#: Classes contributed by this module.
classes: tuple[type, ...] = (UVVAR_OT_apply_candidate,)


def register() -> None:
    """Register this module's operators."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    """Unregister this module's operators, tolerating partial registration."""
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            _LOG.debug("could not unregister %s", cls.__name__, exc_info=True)
