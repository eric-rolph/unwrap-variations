"""A compact candidate switcher for the UV/Image Editor sidebar.

Flipping between candidates is far more informative when you are looking at the
UV layout itself, so the same grid is mirrored into the UV editor's ``N`` panel.
This module deliberately owns no logic of its own: it reuses the drawing helpers
from :mod:`.viewport_panel` so the two panels can never drift apart in how they
read the property mirror, which operators they call or how they fail.

Everything here obeys the same two rules as the viewport panel: ``draw()`` never
raises, and ``draw()`` reads only the ``Scene.uvvar`` mirror described in
ARCHITECTURE section 11.
"""

from __future__ import annotations

import bpy
from bpy.types import Context, Panel

from .viewport_panel import (
    CATEGORY,
    ICON_BEST,
    ICON_PLAIN,
    ICON_WARN,
    OP_APPLY,
    OP_CYCLE,
    OP_GENERATE,
    _active_index,
    _best_index,
    _f,
    _i,
    _s,
    candidate_items,
    draw_bar,
    draw_candidate_grid,
    draw_empty_state,
    draw_op,
    draw_prop,
    draw_warnings,
    get_props,
    region_width,
    set_op_props,
)

__all__ = [
    "UVVAR_PT_uv_editor",
    "UVVAR_PT_uv_editor_metrics",
    "classes",
    "register",
    "unregister",
]

#: Image editor modes in which the candidate switcher is useful.
_UV_MODES = frozenset({"UV", "VIEW"})


class _UVVarImagePanel:
    """Shared panel configuration for the UV/Image Editor sidebar."""

    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = CATEGORY

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Show only in the UV view of the image editor."""
        try:
            space = getattr(context, "space_data", None)
            if space is None or getattr(context, "scene", None) is None:
                return False
            return str(getattr(space, "mode", "UV")) in _UV_MODES
        except Exception:  # noqa: BLE001
            return False


class UVVAR_PT_uv_editor(_UVVarImagePanel, Panel):
    """Compact candidate switcher shown next to the UV layout."""

    bl_idname = "UVVAR_PT_uv_editor"
    bl_label = "Unwrap Variations"
    bl_order = 0

    def draw_header(self, context: Context) -> None:
        """Draw the small icon in the panel header."""
        try:
            self.layout.label(text="", icon="GROUP_UVS")
        except Exception:  # noqa: BLE001
            pass

    def draw(self, context: Context) -> None:
        """Draw the compact switcher: active slot, cycle, grid, apply."""
        layout = self.layout
        try:
            props = get_props(context)
            if props is None:
                row = layout.row()
                row.alert = True
                row.label(text="Scene properties not registered.", icon=ICON_WARN)
                return

            items = candidate_items(props, context)
            count = len(items)
            if count == 0:
                draw_empty_state(layout)
                op = draw_op(
                    layout, OP_GENERATE, text="Generate Grid", icon="SEQUENCE"
                )
                set_op_props(op, count=max(1, _i(props, "candidate_count", 8)))
                return

            active = _active_index(props, count)
            best = _best_index(items)
            item = items[active] if 0 <= active < count else None

            head = layout.box().column(align=True)
            head.label(
                text="Slot %02d of %d" % (active + 1, count),
                icon=ICON_BEST if active == best else ICON_PLAIN,
            )
            head.label(text=_s(item, "label", "?"))
            draw_bar(
                head, "Score", _f(item, "score") / 100.0, "%.1f" % _f(item, "score")
            )

            nav = layout.row(align=True)
            nav.scale_y = 1.2
            prev_op = draw_op(nav, OP_CYCLE, text="", icon="TRIA_LEFT")
            set_op_props(prev_op, direction="PREV", delta=-1, offset=-1)
            draw_prop(nav, props, "active_index", text="")
            next_op = draw_op(nav, OP_CYCLE, text="", icon="TRIA_RIGHT")
            set_op_props(next_op, direction="NEXT", delta=1, offset=1)

            layout.separator()
            draw_candidate_grid(
                layout, props, items, width=region_width(context)
            )

            layout.separator()
            apply_row = layout.row(align=True)
            apply_row.scale_y = 1.25
            op = draw_op(apply_row, OP_APPLY, text="Apply", icon="CHECKMARK")
            set_op_props(op, index=active)
        except Exception:  # noqa: BLE001 - last line of defence.
            try:
                row = layout.row()
                row.alert = True
                row.label(text="UI error - see console.", icon=ICON_WARN)
            except Exception:  # noqa: BLE001
                pass


class UVVAR_PT_uv_editor_metrics(_UVVarImagePanel, Panel):
    """The headline numbers for the candidate currently on screen."""

    bl_idname = "UVVAR_PT_uv_editor_metrics"
    bl_parent_id = "UVVAR_PT_uv_editor"
    bl_label = "Quality"
    bl_order = 1

    #: Metrics worth the space in a narrow sidebar, as
    #: ``(attribute, caption, formatter, quality)``.
    _ROWS = (
        ("stretch_l2", "Stretch L2", lambda v: "%.3f" % v,
         lambda v: (1.0 / v) if v >= 1.0 else 1.0),
        ("angle_distortion", "Angle", lambda v: "%.2f deg" % v,
         lambda v: 1.0 - v / 30.0),
        ("packing_efficiency", "Packing", lambda v: "%.1f%%" % (v * 100.0),
         lambda v: v),
        ("texel_density_cv", "Texel Even", lambda v: "%.3f" % v,
         lambda v: 1.0 - v),
    )

    def draw(self, context: Context) -> None:
        """Draw warnings plus a short metric summary."""
        layout = self.layout
        try:
            props = get_props(context)
            items = candidate_items(props, context)
            count = len(items)
            if count == 0:
                layout.label(text="Nothing generated yet.", icon="INFO")
                return
            item = items[_active_index(props, count)]

            draw_warnings(layout, item)

            col = layout.column(align=True)
            for attr, caption, fmt, quality in self._ROWS:
                value = _f(item, attr)
                draw_bar(col, caption, quality(value), fmt(value))

            info = layout.row(align=True)
            info.scale_y = 0.85
            info.label(text="Islands: %d" % _i(item, "island_count"))
            info.label(text="Seams: %.2f" % _f(item, "seam_length"))
        except Exception:  # noqa: BLE001
            try:
                row = layout.row()
                row.alert = True
                row.label(text="Metrics unavailable.", icon=ICON_WARN)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: Every registrable class in this module; the parent precedes its sub-panel.
classes: tuple[type, ...] = (
    UVVAR_PT_uv_editor,
    UVVAR_PT_uv_editor_metrics,
)


def register() -> None:
    """Register every panel in this module."""
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception:  # noqa: BLE001
            pass


def unregister() -> None:
    """Unregister every panel in this module, children first."""
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            pass
