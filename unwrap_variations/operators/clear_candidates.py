"""``uvvar.clear_candidates`` -- discard the exploration and tidy up.

Clearing has to reach four places, and missing any one of them leaves the
add-on in a state the user experiences as a bug:

1. the **runtime cache** of numpy arrays (otherwise the candidates come
   straight back on the next panel refresh);
2. the **persisted blob** on the mesh (otherwise they come back after a file
   reload, and keep occupying space in the ``.blend``);
3. the **UI mirror list** (otherwise the panel shows rows that no longer have
   data behind them);
4. the scratch **``UVVar_Preview`` UV layer** (otherwise the mesh keeps a
   mystery UV map, and -- worse -- keeps occupying one of the eight slots).

Because this throws away real work, it confirms first.
"""

from __future__ import annotations

from typing import Any

import bpy
from bpy.props import BoolProperty
from bpy.types import Context, Event, Operator

from .. import properties as uvvar_props
from ..utils import compat
from ..utils import logging as uvvar_logging
from .select_candidate import PREVIEW_LAYER

__all__ = [
    "UVVAR_OT_clear_candidates",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)


class UVVAR_OT_clear_candidates(Operator):
    """Discard every generated variation for the active mesh."""

    bl_idname = "uvvar.clear_candidates"
    bl_label = "Clear Variations"
    bl_description = (
        "Discard the generated variations, remove the preview UV layer and "
        "drop the copy stored inside this .blend"
    )
    bl_options = {"REGISTER", "UNDO"}

    remove_preview: BoolProperty(
        name="Remove Preview Layer",
        description="Delete the scratch UVVar_Preview UV map and free its slot",
        default=True,
    )

    clear_persisted: BoolProperty(
        name="Clear Saved Copy",
        description="Also remove the candidate data stored inside the .blend file",
        default=True,
    )

    all_objects: BoolProperty(
        name="All Selected Objects",
        description="Clear every selected mesh instead of just the active one",
        default=False,
    )

    confirm: BoolProperty(
        name="Confirm",
        description="Ask before discarding",
        default=True,
        options={"SKIP_SAVE", "HIDDEN"},
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled when there is anything at all to clear."""
        obj = compat.active_mesh_object(context)
        if obj is None:
            return False
        if uvvar_props.get_store(obj) is not None:
            return True
        if compat.uv_layer_by_name(obj.data, PREVIEW_LAYER) is not None:
            return True
        props = uvvar_props.get_props(context)
        if props is not None and len(props.candidates):
            return True
        try:
            store_cls = uvvar_props.CandidateStore
            return store_cls is not None and obj.data.get(store_cls.ID_KEY) is not None
        except Exception:  # noqa: BLE001
            return False

    def invoke(self, context: Context, event: Event) -> set[str]:
        """Confirm before discarding, unless the caller opted out."""
        if not self.confirm:
            return self.execute(context)
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context: Context) -> set[str]:
        """Clear the candidates for one or many objects."""
        try:
            objects = self._targets(context)
            if not objects:
                self.report({"ERROR"}, "No mesh object to clear")
                return {"CANCELLED"}

            cleared = 0
            layers_removed = 0
            for obj in objects:
                did_clear, did_layer = self._clear_one(obj)
                cleared += 1 if did_clear else 0
                layers_removed += 1 if did_layer else 0

            props = uvvar_props.get_props(context)
            if props is not None:
                try:
                    props.candidates.clear()
                    props.active_index = 0
                    props.progress = 0.0
                    props.status = "Cleared"
                except Exception:  # noqa: BLE001
                    _LOG.debug("could not reset scene properties", exc_info=True)

            active = compat.active_mesh_object(context)
            if active is not None and active not in objects:
                uvvar_props.sync_ui_list(getattr(context, "scene", None), active)

            compat.tag_redraw(context)
            message = "Cleared variations on %d object(s)" % len(objects)
            if layers_removed:
                message += ", removed %d preview layer(s)" % layers_removed
            self.report({"INFO"}, uvvar_logging.ascii_safe(message))
            _LOG.info("%s", message)
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001 - nothing escapes into Blender.
            _LOG.exception("clear_candidates failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}

    # -- implementation ------------------------------------------------------

    def _targets(self, context: Context) -> list[Any]:
        """Return the objects to clear."""
        active = compat.active_mesh_object(context)
        if not self.all_objects:
            return [active] if active is not None else []
        selected = [
            obj
            for obj in (getattr(context, "selected_objects", None) or [])
            if getattr(obj, "type", "") == "MESH" and getattr(obj, "data", None) is not None
        ]
        if active is not None and active not in selected:
            selected.append(active)
        return selected

    def _clear_one(self, obj: Any) -> tuple[bool, bool]:
        """Clear one object; return ``(had_candidates, removed_preview_layer)``."""
        store = uvvar_props.get_store(obj)
        had = bool(store is not None and len(store))
        if store is not None:
            try:
                store.clear()
            except Exception:  # noqa: BLE001
                _LOG.debug("store.clear failed", exc_info=True)
        uvvar_props.drop_store(obj)

        if self.clear_persisted:
            had = self._clear_persisted(obj) or had

        removed = False
        if self.remove_preview:
            removed = self._remove_preview_layer(obj)
        return had, removed

    @staticmethod
    def _clear_persisted(obj: Any) -> bool:
        """Delete the persisted blob from the mesh; return whether one existed."""
        mesh = getattr(obj, "data", None)
        if mesh is None:
            return False
        key = None
        try:
            from ..core.candidate import CandidateStore

            key = CandidateStore.ID_KEY
        except Exception:  # noqa: BLE001
            key = "uvvar_store_v1"
        try:
            if mesh.get(key) is not None:
                del mesh[key]
                return True
        except Exception:  # noqa: BLE001
            _LOG.debug("could not delete %r from %r", key, getattr(mesh, "name", "?"),
                       exc_info=True)
        return False

    @staticmethod
    def _remove_preview_layer(obj: Any) -> bool:
        """Delete the scratch preview layer; return whether one was removed."""
        mesh = getattr(obj, "data", None)
        if mesh is None or compat.uv_layer_by_name(mesh, PREVIEW_LAYER) is None:
            return False
        try:
            from ..core import bridge

            bridge.remove_preview_layer(obj)
            if compat.uv_layer_by_name(mesh, PREVIEW_LAYER) is None:
                return True
        except Exception:  # noqa: BLE001 - fall through to the direct path.
            _LOG.debug("core.bridge.remove_preview_layer unavailable", exc_info=True)
        layer = compat.uv_layer_by_name(mesh, PREVIEW_LAYER)
        if layer is None:
            return True
        try:
            mesh.uv_layers.remove(layer)
            return True
        except Exception:  # noqa: BLE001
            _LOG.debug("could not remove the preview layer", exc_info=True)
            return False


#: Classes contributed by this module.
classes: tuple[type, ...] = (UVVAR_OT_clear_candidates,)


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
