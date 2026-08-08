"""``uvvar.select_candidate`` and ``uvvar.cycle_candidate`` -- instant preview.

Previewing a candidate is deliberately the cheapest thing the add-on does: one
``foreach_set`` of a pre-shaped float32 array into the single scratch UV layer
``UVVar_Preview``.  No unwrap re-runs, no layer creation, no evaluation.  That
is what makes holding down the cycle shortcut feel like scrubbing a timeline
rather than waiting on a solver, and it is the entire reason candidates are
stored as numpy arrays instead of as UV layers (Blender caps a mesh at 8, and a
10-candidate grid would not fit anyway).

**Why these two operators are not ``UNDO``.**  Everything else that mutates the
mesh carries ``bl_options = {"REGISTER", "UNDO"}``.  These two do not, on
purpose: the preview layer is scratch state that ``uvvar.clear_candidates``
removes wholesale, and an undo push copies the *entire* mesh.  On a
million-loop mesh that turns a 0.4 ms blit into a multi-hundred-millisecond
copy, on every arrow-key press, and fills the undo stack with steps the user
never asked for.  The user-visible result of a preview is undone by previewing
something else or by clearing; nothing is lost.
"""

from __future__ import annotations

from typing import Any, Callable

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Context, Operator

from .. import properties as uvvar_props
from ..utils import compat
from ..utils import logging as uvvar_logging

__all__ = [
    "UVVAR_OT_select_candidate",
    "UVVAR_OT_cycle_candidate",
    "preview_candidate",
    "PREVIEW_LAYER",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)

#: Name of the scratch UV layer candidates are blitted into.  Taken from
#: ``CandidateStore.PREVIEW_LAYER`` when that module is importable so the two
#: can never drift, with the verified literal as the fallback.
PREVIEW_LAYER = compat.PREVIEW_LAYER_NAME
try:  # pragma: no cover - import shim.
    from ..core.candidate import CandidateStore as _CandidateStore

    PREVIEW_LAYER = getattr(_CandidateStore, "PREVIEW_LAYER", PREVIEW_LAYER)
except Exception:  # noqa: BLE001
    _CandidateStore = None  # type: ignore[assignment]


def _ensure_preview_layer(obj: Any) -> tuple[Any, str]:
    """Return the preview UV layer for ``obj``, creating it if needed.

    Prefers ``core.bridge.ensure_preview_layer`` so that layer bookkeeping
    lives in the one module allowed to touch both worlds; falls back to a
    direct, equivalent implementation when the bridge is not importable, which
    keeps preview working while that module is still landing.

    Args:
        obj: The mesh object to preview on.

    Returns:
        ``(layer, note)``.  ``layer`` is ``None`` when no slot was free, in
        which case ``note`` explains why.
    """
    mesh = obj.data
    existing = compat.uv_layer_by_name(mesh, PREVIEW_LAYER)
    if existing is not None:
        return existing, ""

    try:
        from ..core import bridge

        name = bridge.ensure_preview_layer(obj)
        layer = compat.uv_layer_by_name(mesh, name or PREVIEW_LAYER)
        if layer is not None:
            return layer, ""
    except Exception:  # noqa: BLE001 - fall through to the direct path.
        _LOG.debug("core.bridge.ensure_preview_layer unavailable", exc_info=True)

    if compat.free_uv_layer_slots(obj) <= 0:
        return None, (
            "mesh %r already has %d UV maps, which is Blender's hard cap; "
            "delete one to make room for the preview layer"
            % (mesh.name, compat.MAX_UV_LAYERS)
        )
    try:
        layer = mesh.uv_layers.new(name=PREVIEW_LAYER, do_init=False)
    except TypeError:
        layer = mesh.uv_layers.new(name=PREVIEW_LAYER)
    except Exception as exc:  # noqa: BLE001
        return None, "could not create the preview UV layer: %s" % exc
    if layer is None:
        return None, "Blender refused to create another UV map (8 is the cap)"
    return layer, ""


def _resolve_store_index(props: Any, index: int) -> int:
    """Map a UI-list row to the store index it mirrors.

    The list is sorted by score, so row 3 is rarely candidate 3.  The mirror row
    carries the true store index; this indirection is where that is undone.

    Args:
        props: The scene property group, or ``None``.
        index: A row index in ``props.candidates``.

    Returns:
        The store index, or ``index`` unchanged when there is no mirror to
        consult (which is the right answer for a scripted call).
    """
    if props is None:
        return int(index)
    rows = props.candidates
    if not len(rows):
        return int(index)
    row_index = max(0, min(int(index), len(rows) - 1))
    try:
        return int(rows[row_index].index)
    except Exception:  # noqa: BLE001
        return int(index)


def preview_candidate(
    context: Context,
    *,
    index: int,
    report: Callable[[set[str], str], None] | None = None,
    obj: Any = None,
    store_index: int | None = None,
) -> tuple[bool, str]:
    """Blit one candidate's UVs into the preview layer.

    This is a plain function rather than only an operator because the scene's
    ``active_index`` update callback needs it: invoking ``bpy.ops`` from a
    property update is the classic route to "Writing to ID classes in this
    context is not allowed", while a direct ``foreach_set`` is safe anywhere the
    property itself can be edited.

    Args:
        context: The active context.
        index: Row index in the mirror list.
        report: Optional ``Operator.report``-style callback for user messages.
        obj: Object override; the active mesh object is used otherwise.
        store_index: Bypass the mirror and address the store directly.

    Returns:
        ``(ok, message)``.  Never raises.
    """
    def _say(level: str, text: str) -> None:
        """Forward a message to the operator report, if there is one."""
        if report is not None:
            try:
                report({level}, uvvar_logging.ascii_safe(text))
            except Exception:  # noqa: BLE001
                pass

    target = obj if obj is not None else compat.active_mesh_object(context)
    if target is None:
        _say("ERROR", "No active mesh object")
        return False, "no active mesh object"

    store = uvvar_props.store_for(target, create=False)
    if store is None or not len(store):
        message = "No candidates for %r; run Generate first" % getattr(target, "name", "?")
        _say("WARNING", message)
        return False, message

    props = uvvar_props.get_props(context)
    resolved = int(store_index) if store_index is not None else _resolve_store_index(props, index)
    candidate = store.get(resolved)
    if candidate is None:
        message = "Candidate %d does not exist" % resolved
        _say("WARNING", message)
        return False, message
    if candidate.uvs is None:
        message = "Candidate %d has no UVs (%s)" % (resolved, candidate.message or "backend failed")
        _say("WARNING", message)
        return False, message

    mesh = target.data
    loops = len(mesh.loops)
    if int(candidate.uvs.shape[0]) != loops:
        message = (
            "stored candidate has %d loops but the mesh now has %d; "
            "the mesh changed since Generate -- run it again"
            % (int(candidate.uvs.shape[0]), loops)
        )
        _say("ERROR", message)
        return False, message

    layer, note = _ensure_preview_layer(target)
    if layer is None:
        _say("ERROR", note)
        return False, note

    try:
        from ..core import bridge

        bridge.apply_uvs(target, candidate.uvs, layer_name=layer.name)
    except Exception:  # noqa: BLE001 - the bridge is optional; the blit is not.
        try:
            compat.write_uv_layer(layer, candidate.uvs)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("preview blit failed", exc_info=True)
            message = "could not write the preview UV layer: %s" % exc
            _say("ERROR", message)
            return False, message

    # Show the preview layer in the UV editor.  ``active_render`` is left alone
    # on purpose: that flag decides what actually renders, and a preview must
    # not quietly change a user's output. The CHECKER overlay covers the
    # "how does it look on the model" question without touching render state.
    try:
        mesh.uv_layers.active = layer
    except Exception:  # noqa: BLE001
        _LOG.debug("could not make the preview layer active", exc_info=True)

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
    message = "Candidate %d  %s  score %.1f" % (resolved, label, float(candidate.score or 0.0))
    if props is not None:
        try:
            props.status = message[:220]
        except Exception:  # noqa: BLE001
            pass
    return True, message


def _has_candidates(context: Context) -> bool:
    """Return whether the active object has anything to preview."""
    obj = compat.active_mesh_object(context)
    if obj is None:
        return False
    store = uvvar_props.get_store(obj)
    if store is not None and len(store):
        return True
    props = uvvar_props.get_props(context)
    return bool(props is not None and len(props.candidates))


class UVVAR_OT_select_candidate(Operator):
    """Show a candidate in the preview UV layer."""

    bl_idname = "uvvar.select_candidate"
    bl_label = "Preview Candidate"
    bl_description = "Blit the selected variation into the preview UV layer"
    # Not UNDO: see the module docstring. A preview is scratch state and an
    # undo push per keypress would cost far more than the blit itself.
    bl_options = {"REGISTER"}

    index: IntProperty(
        name="Index",
        description="Row to preview; -1 uses the scene's active candidate",
        default=-1,
        min=-1,
    )

    store_index: IntProperty(
        name="Store Index",
        description="Address the candidate store directly, bypassing list order; -1 to ignore",
        default=-1,
        min=-1,
    )

    set_active: BoolProperty(
        name="Set Active",
        description="Also move the list selection to this candidate",
        default=True,
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled when the active mesh object has candidates."""
        return _has_candidates(context)

    def execute(self, context: Context) -> set[str]:
        """Preview the requested candidate."""
        try:
            props = uvvar_props.get_props(context)
            row = int(self.index)
            if row < 0:
                row = int(getattr(props, "active_index", 0) or 0)

            if self.set_active and props is not None and len(props.candidates):
                clamped = max(0, min(row, len(props.candidates) - 1))
                if int(props.active_index) != clamped:
                    # Assigning fires the auto-preview callback, which performs
                    # exactly the blit below; suppress it to avoid doing the
                    # work twice.
                    auto = bool(props.auto_preview)
                    try:
                        props.auto_preview = False
                        props.active_index = clamped
                    finally:
                        props.auto_preview = auto
                row = clamped

            store_index = int(self.store_index)
            ok, message = preview_candidate(
                context,
                index=row,
                report=self.report,
                store_index=store_index if store_index >= 0 else None,
            )
            if not ok:
                return {"CANCELLED"}
            _LOG.debug("preview: %s", message)
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001 - nothing escapes into Blender.
            _LOG.exception("select_candidate failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}


class UVVAR_OT_cycle_candidate(Operator):
    """Step through candidates; bound to a keyboard shortcut."""

    bl_idname = "uvvar.cycle_candidate"
    bl_label = "Cycle Candidate"
    bl_description = "Move to the next or previous variation and preview it"
    # Not UNDO: same reasoning as uvvar.select_candidate.
    bl_options = {"REGISTER"}

    direction: EnumProperty(
        name="Direction",
        description="Which candidate to move to",
        items=(
            ("NEXT", "Next", "Move to the next candidate in the list"),
            ("PREV", "Previous", "Move to the previous candidate in the list"),
            ("FIRST", "First", "Jump to the first candidate"),
            ("LAST", "Last", "Jump to the last candidate"),
            ("BEST", "Best", "Jump to the highest scoring candidate"),
        ),
        default="NEXT",
    )

    wrap: BoolProperty(
        name="Wrap Around",
        description="Continue from the other end when the list runs out",
        default=True,
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled when the active mesh object has candidates."""
        return _has_candidates(context)

    def execute(self, context: Context) -> set[str]:
        """Move the selection and preview the result."""
        try:
            props = uvvar_props.get_props(context)
            if props is None:
                self.report({"ERROR"}, "Scene properties are not registered")
                return {"CANCELLED"}

            total = len(props.candidates)
            if total == 0:
                self.report({"WARNING"}, "No candidates to cycle through")
                return {"CANCELLED"}

            current = max(0, min(int(props.active_index), total - 1))
            target = self._target_index(current, total)
            if target is None:
                self.report({"INFO"}, "Already at the end of the list")
                return {"CANCELLED"}

            props.active_index = target  # Fires auto-preview when enabled.
            if not props.auto_preview:
                preview_candidate(context, index=target, report=self.report)

            row = props.candidates[target]
            self.report(
                {"INFO"},
                uvvar_logging.ascii_safe(
                    "%d/%d  %s  score %.1f" % (target + 1, total, row.description, row.score)
                ),
            )
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("cycle_candidate failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}

    def _target_index(self, current: int, total: int) -> int | None:
        """Return the row to move to, or ``None`` when the move is impossible."""
        if self.direction == "FIRST" or self.direction == "BEST":
            # The list is ranked best-first, so both land on row zero.
            return 0
        if self.direction == "LAST":
            return total - 1
        step = 1 if self.direction == "NEXT" else -1
        target = current + step
        if 0 <= target < total:
            return target
        if not self.wrap:
            return None
        return target % total


#: Classes contributed by this module.
classes: tuple[type, ...] = (
    UVVAR_OT_select_candidate,
    UVVAR_OT_cycle_candidate,
)


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
