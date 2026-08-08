"""The 3D Viewport sidebar: the candidate grid and everything around it.

This module is the front door of the add-on.  The pitch -- "Midjourney grid
selection, but for UV unwraps" -- lives or dies on this panel, so the candidate
grid is the centrepiece and everything else is arranged around it.

Two rules govern every line below and they are not negotiable:

1. **Draw code never raises.**  A panel that throws re-throws on every single
   redraw, floods the console and can leave the region in a half-drawn state.
   Every value is read with :func:`getattr` defaults, every operator is checked
   for existence before it is drawn, and every property is checked before
   :meth:`bpy.types.UILayout.prop` is called on it.
2. **Draw code is cheap.**  ``draw()`` runs on every redraw of every visible
   3D viewport.  It therefore reads *only* the ``Scene.uvvar`` property mirror
   described in ARCHITECTURE section 11 -- never numpy, never the candidate
   store, never the mesh.

The sibling modules (``core.engine``, ``operators.*``, ``properties``) are built
in parallel with this one, so nothing here imports them.  A missing operator
degrades to a greyed-out row that explains itself; a missing property degrades
to a dimmed placeholder.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import bpy
from bpy.types import Context, Panel, UILayout

__all__ = [
    "CATEGORY",
    "UVVAR_PT_main",
    "UVVAR_PT_generate",
    "UVVAR_PT_candidates",
    "UVVAR_PT_metrics",
    "UVVAR_PT_overlay",
    "UVVAR_PT_advanced",
    "classes",
    "register",
    "unregister",
    "get_props",
    "candidate_items",
    "draw_candidate_grid",
    "draw_metric_readout",
    "draw_warnings",
    "draw_empty_state",
    "draw_object_state",
    "region_width",
    "op_exists",
    "draw_op",
    "set_op_props",
    "draw_prop",
    "draw_bar",
    "has_progress",
]

#: Sidebar tab this add-on owns.  Shared by the UV/Image editor panels.
CATEGORY = "UnwrapVar"

# -- operator contract (ARCHITECTURE section 10) ----------------------------
OP_GENERATE = "uvvar.generate_grid"
OP_BRANCH = "uvvar.branch_variations"
OP_SELECT = "uvvar.select_candidate"
OP_APPLY = "uvvar.apply_candidate"
OP_CLEAR = "uvvar.clear_candidates"
OP_OVERLAY = "uvvar.toggle_overlay"
OP_CYCLE = "uvvar.cycle_candidate"

# -- icons ------------------------------------------------------------------
#: Every one of these was verified to exist in Blender 4.5.4 and 5.1.2.
ICON_BEST = "SOLO_ON"          # top-ranked candidate
ICON_PARETO = "KEYFRAME_HLT"   # non-dominated on the stretch/seam/pack front
ICON_PLAIN = "KEYFRAME"        # dominated candidate
ICON_FAILED = "CANCEL"         # backend refused this recipe
ICON_WARN = "ERROR"
ICON_BAR_FULL = "LAYER_ACTIVE"
ICON_BAR_EMPTY = "LAYER_USED"

#: Cells in the fallback bar used when ``UILayout.progress`` is unavailable.
BAR_CELLS = 10

_HAS_PROGRESS: bool | None = None


# ---------------------------------------------------------------------------
# Safety helpers.  Everything the panels touch goes through one of these.
# ---------------------------------------------------------------------------


def has_progress() -> bool:
    """Return whether ``UILayout.progress()`` exists in this Blender build.

    ``dir(bpy.types.UILayout)`` deliberately does *not* list RNA functions, so
    ``hasattr(bpy.types.UILayout, "progress")`` returns ``False`` even on builds
    that have it.  The reliable probe is the RNA function table.  Verified
    present on 4.5.4 LTS and 5.1.2; the icon-strip fallback exists for older
    builds and for any future build that drops it.

    Returns:
        ``True`` when ``layout.progress(factor=..., text=..., type='BAR')``
        can be called.
    """
    global _HAS_PROGRESS
    if _HAS_PROGRESS is None:
        try:
            _HAS_PROGRESS = "progress" in bpy.types.UILayout.bl_rna.functions
        except Exception:  # noqa: BLE001 - never let a probe break a draw.
            _HAS_PROGRESS = False
    return bool(_HAS_PROGRESS)


def op_exists(idname: str) -> bool:
    """Return whether an operator ``bl_idname`` is currently registered.

    Blender derives an operator's RNA type name from its ``bl_idname``
    deterministically (``uvvar.generate_grid`` -> ``UVVAR_OT_generate_grid``),
    so a single attribute probe on :mod:`bpy.types` answers the question without
    instantiating anything.  This lets the panel stay drawable while the
    operator modules are still being written.

    Args:
        idname: A dotted operator id such as ``"uvvar.generate_grid"``.

    Returns:
        ``True`` if the operator can be drawn.
    """
    try:
        category, _, name = idname.partition(".")
        return hasattr(bpy.types, "%s_OT_%s" % (category.upper(), name))
    except Exception:  # noqa: BLE001
        return False


def draw_op(
    layout: UILayout,
    idname: str,
    *,
    text: str = "",
    icon: str = "NONE",
    depress: bool = False,
    emboss: bool = True,
) -> Any:
    """Draw an operator button, or a self-explaining placeholder if it is gone.

    Args:
        layout: Layout to draw into.
        idname: Operator ``bl_idname``.
        text: Button label.
        icon: Button icon identifier.
        depress: Draw the button in its "pressed in" state.  This is how the
            active candidate is made visually obvious.
        emboss: Draw the button background.

    Returns:
        The operator properties object, or ``None`` when the operator is not
        registered (in which case a disabled placeholder was drawn instead).
    """
    if not op_exists(idname):
        row = layout.row(align=True)
        row.enabled = False
        row.label(text=text or idname, icon=ICON_WARN)
        return None
    try:
        return layout.operator(
            idname, text=text, icon=icon, depress=depress, emboss=emboss
        )
    except Exception:  # noqa: BLE001 - a bad icon name must not kill the panel.
        try:
            return layout.operator(idname, text=text)
        except Exception:  # noqa: BLE001
            return None


def set_op_props(op: Any, **values: Any) -> None:
    """Assign properties onto an operator button, ignoring unknown ones.

    The operator modules are written in parallel, so a property this panel wants
    to pass may not exist yet.  Silently skipping it keeps the button usable
    with its defaults instead of breaking the whole panel.

    Args:
        op: The object returned by :func:`draw_op`, possibly ``None``.
        **values: Property names and values to set.
    """
    if op is None:
        return
    for key, value in values.items():
        try:
            setattr(op, key, value)
        except Exception:  # noqa: BLE001
            continue


def draw_prop(
    layout: UILayout,
    data: Any,
    name: str,
    *,
    text: str | None = None,
    optional: bool = False,
    **kwargs: Any,
) -> bool:
    """Draw a property widget, or a dimmed placeholder if the property is gone.

    A missing *contract* property (ARCHITECTURE section 11) is worth surfacing,
    so it leaves a greyed-out row that names what is absent.  A missing
    *optional* property is not a fault, so it draws nothing at all rather than
    littering the panel with dead rows.

    Args:
        layout: Layout to draw into.
        data: The property group (may be ``None``).
        name: Property identifier.
        text: Optional label override.
        optional: Draw nothing when the property does not exist.
        **kwargs: Passed straight through to ``layout.prop``.

    Returns:
        ``True`` when a real widget was drawn.
    """
    if data is None or not hasattr(data, name):
        if optional:
            return False
        row = layout.row()
        row.enabled = False
        row.label(text=text if text is not None else name.replace("_", " ").title())
        return False
    try:
        if text is None:
            layout.prop(data, name, **kwargs)
        else:
            layout.prop(data, name, text=text, **kwargs)
        return True
    except Exception:  # noqa: BLE001
        return False


def get_props(context: Context) -> Any:
    """Return the ``Scene.uvvar`` property group, or ``None`` if unregistered.

    Args:
        context: The current Blender context.

    Returns:
        The :class:`UVVAR_SceneProps` instance or ``None``.
    """
    try:
        scene = getattr(context, "scene", None)
        return getattr(scene, "uvvar", None) if scene is not None else None
    except Exception:  # noqa: BLE001
        return None


def candidate_items(props: Any) -> Sequence[Any]:
    """Return the UI mirror collection of candidates.

    ARCHITECTURE section 11 names this a ``CollectionProperty`` of
    ``UVVAR_CandidateItem``.  A couple of plausible attribute names are tried so
    a naming difference in ``properties.py`` degrades to an empty grid with a
    helpful message rather than a traceback.

    Args:
        props: The scene property group, possibly ``None``.

    Returns:
        The collection, or an empty tuple.
    """
    if props is None:
        return ()
    for name in ("candidates", "candidate_items", "items", "grid"):
        coll = getattr(props, name, None)
        if coll is not None:
            try:
                len(coll)
            except Exception:  # noqa: BLE001
                continue
            return coll
    return ()


def _f(item: Any, name: str, default: float = 0.0) -> float:
    """Read a float from a mirror item without ever raising."""
    try:
        return float(getattr(item, name, default))
    except (TypeError, ValueError):
        return default


def _i(item: Any, name: str, default: int = 0) -> int:
    """Read an int from a mirror item without ever raising."""
    try:
        return int(getattr(item, name, default))
    except (TypeError, ValueError):
        return default


def _s(item: Any, name: str, default: str = "") -> str:
    """Read a string from a mirror item without ever raising."""
    try:
        value = getattr(item, name, default)
        return str(value) if value is not None else default
    except Exception:  # noqa: BLE001
        return default


def _b(item: Any, name: str, default: bool = False) -> bool:
    """Read a bool from a mirror item without ever raising."""
    try:
        return bool(getattr(item, name, default))
    except Exception:  # noqa: BLE001
        return default


def _clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``, mapping NaN to 0."""
    if value != value:  # NaN
        return 0.0
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def _active_index(props: Any, count: int) -> int:
    """Return the active candidate index clamped into ``[0, count)``."""
    if count <= 0:
        return -1
    idx = _i(props, "active_index", 0)
    return 0 if idx < 0 else (count - 1 if idx >= count else idx)


# ---------------------------------------------------------------------------
# Metric bars
# ---------------------------------------------------------------------------


def draw_bar(
    layout: UILayout,
    label: str,
    quality: float,
    value_text: str,
    *,
    warn: bool = False,
    split_factor: float = 0.44,
) -> None:
    """Draw one labelled quality bar.

    ``quality`` is always oriented so that **a full bar is good**, regardless of
    whether the underlying metric is minimised or maximised.  Mixing the two
    orientations in one column is the fastest way to make a metrics readout
    unreadable, so the conversion happens in :data:`METRIC_ROWS` instead.

    Args:
        layout: Layout to draw into.
        label: Left-hand caption.
        quality: ``0.0`` (bad) to ``1.0`` (ideal).
        value_text: The real number, shown inside or beside the bar.
        warn: Draw in Blender's alert colour.
        split_factor: Fraction of the width given to the caption.
    """
    try:
        row = layout.row(align=True)
        row.alert = bool(warn)
        split = row.split(factor=split_factor, align=True)
        split.label(text=label)
        bar = split.row(align=True)
        factor = _clamp01(quality)
        if has_progress():
            bar.progress(factor=factor, text=value_text, type="BAR")
            return
        # Fallback for builds without UILayout.progress(): an icon strip.
        cells = bar.row(align=True)
        cells.scale_x = 0.36
        filled = int(round(factor * BAR_CELLS))
        for i in range(BAR_CELLS):
            cells.label(text="", icon=ICON_BAR_FULL if i < filled else ICON_BAR_EMPTY)
        text_row = bar.row(align=True)
        text_row.alignment = "RIGHT"
        text_row.label(text=value_text)
    except Exception:  # noqa: BLE001 - a metric bar is never worth a traceback.
        try:
            layout.label(text="%s  %s" % (label, value_text))
        except Exception:  # noqa: BLE001
            pass


#: ``(attribute, caption, value formatter, quality function)``.
#:
#: The quality function maps the raw metric onto ``0..1`` where ``1`` is ideal,
#: so every bar in the readout points the same way.  The scalings are chosen to
#: put a *typical production-quality* unwrap around three quarters full:
#: ``stretch_l2`` of 1.0 is perfect and 2.0 is poor; mean angle distortion of
#: 30 degrees is poor; ``area_distortion`` is a mean absolute log2 ratio, so 2.0
#: means the average triangle is off by a factor of four.
METRIC_ROWS: tuple[tuple[str, str, Callable[[float], str], Callable[[float], float]], ...] = (
    (
        "stretch_l2",
        "Stretch L2",
        lambda v: "%.3f" % v,
        lambda v: _clamp01(1.0 / v) if v >= 1.0 else 1.0,
    ),
    (
        "angle_distortion",
        "Angle",
        lambda v: "%.2f deg" % v,
        lambda v: _clamp01(1.0 - v / 30.0),
    ),
    (
        "area_distortion",
        "Area",
        lambda v: "%.3f" % v,
        lambda v: _clamp01(1.0 - v / 2.0),
    ),
    (
        "packing_efficiency",
        "Packing",
        lambda v: "%.1f%%" % (v * 100.0),
        _clamp01,
    ),
    (
        "uv_coverage",
        "Coverage",
        lambda v: "%.1f%%" % (v * 100.0),
        _clamp01,
    ),
    (
        "texel_density_cv",
        "Texel Even",
        lambda v: "%.3f" % v,
        lambda v: _clamp01(1.0 - v),
    ),
    (
        "seam_ratio",
        "Few Seams",
        lambda v: "%.3f" % v,
        lambda v: _clamp01(1.0 - v / 4.0),
    ),
)


def draw_warnings(layout: UILayout, item: Any) -> bool:
    """Draw loud, unmissable warnings for the given candidate.

    Overlapping UVs and flipped faces are not "a number that is a bit high" --
    they are defects that silently corrupt baked normal and light maps.  They
    get their own alert-coloured box with an error icon rather than a quiet
    figure buried in a column.

    Args:
        layout: Layout to draw into.
        item: A candidate mirror item.

    Returns:
        ``True`` when at least one warning was drawn.
    """
    if item is None:
        return False
    drawn = False
    try:
        overlap = _f(item, "overlap_area")
        flipped = _i(item, "flipped_tris")
        degenerate = _i(item, "degenerate_tris")
        oob = _i(item, "out_of_bounds_tris")
        ok = _b(item, "ok", True)
        message = _s(item, "message")

        if not ok:
            box = layout.box()
            box.alert = True
            col = box.column(align=True)
            col.label(text="Candidate failed", icon=ICON_FAILED)
            if message:
                col.label(text=message[:64])
            drawn = True

        if flipped > 0 or overlap > 0.0:
            box = layout.box()
            box.alert = True
            col = box.column(align=True)
            if flipped > 0:
                col.label(text="%d flipped faces" % flipped, icon=ICON_WARN)
                col.label(text="Mirrored UVs break normal map bakes.")
            if overlap > 0.0:
                col.label(text="Overlapping UVs: %.4f" % overlap, icon=ICON_WARN)
                col.label(text="Islands share texel space; bakes will bleed.")
            drawn = True

        if degenerate > 0 or oob > 0:
            box = layout.box()
            col = box.column(align=True)
            if degenerate > 0:
                col.label(text="%d degenerate faces skipped" % degenerate, icon="INFO")
            if oob > 0:
                col.label(text="%d faces outside 0-1 space" % oob, icon="INFO")
            drawn = True
    except Exception:  # noqa: BLE001
        return drawn
    return drawn


def draw_metric_readout(layout: UILayout, item: Any) -> None:
    """Draw the full metric readout for one candidate.

    Args:
        layout: Layout to draw into.
        item: A candidate mirror item, or ``None``.
    """
    if item is None:
        layout.label(text="No candidate selected.", icon="INFO")
        return
    try:
        header = layout.row(align=True)
        header.label(
            text="Slot %02d  %s" % (_i(item, "index") + 1, _s(item, "label", "?")),
            icon=ICON_PLAIN,
        )
        score_row = layout.row(align=True)
        score_row.scale_y = 1.15
        draw_bar(
            score_row,
            "Score",
            _f(item, "score") / 100.0,
            "%.1f / 100" % _f(item, "score"),
            split_factor=0.44,
        )

        draw_warnings(layout, item)

        col = layout.column(align=True)
        for attr, caption, fmt, quality in METRIC_ROWS:
            value = _f(item, attr)
            draw_bar(col, caption, quality(value), fmt(value))

        layout.separator()
        grid = layout.grid_flow(
            row_major=True, columns=2, even_columns=True, align=True
        )
        grid.label(text="Islands: %d" % _i(item, "island_count"))
        grid.label(text="Tris: %d" % _i(item, "n_tris"))
        grid.label(text="Seam len: %.3f" % _f(item, "seam_length"))
        grid.label(text="Stretch inf: %.3f" % _f(item, "stretch_linf"))
        grid.label(text="Angle max: %.1f deg" % _f(item, "angle_distortion_max"))
        grid.label(text="Conformal: %.3f" % _f(item, "conformal_error"))

        seconds = _f(item, "seconds") or _f(item, "eval_seconds")
        if seconds > 0.0:
            layout.label(text="Computed in %.2f s" % seconds)
    except Exception:  # noqa: BLE001
        layout.label(text="Metrics unavailable.", icon=ICON_WARN)


# ---------------------------------------------------------------------------
# The candidate grid -- the centrepiece
# ---------------------------------------------------------------------------


def region_width(context: Context) -> float:
    """Return the drawing region's width in DPI-independent units.

    A sidebar can be anywhere from a 200 px sliver to half the screen, and a
    grid that is right at one width is unreadable at the other.  Dividing out
    ``ui_scale`` makes the number mean the same thing on a 4K display as on a
    laptop panel.

    Args:
        context: The context passed to ``draw()``.

    Returns:
        Effective width in UI units, or ``0.0`` when it cannot be determined.
    """
    try:
        region = getattr(context, "region", None)
        if region is None:
            return 0.0
        scale = float(getattr(context.preferences.system, "ui_scale", 1.0)) or 1.0
        return float(region.width) / scale
    except Exception:  # noqa: BLE001
        return 0.0


def _grid_columns(props: Any, count: int, width: float = 0.0) -> int:
    """Return how many columns the candidate grid should use.

    Args:
        props: Scene property group; an explicit ``grid_columns`` wins.
        count: Number of candidates.
        width: Effective region width from :func:`region_width`.

    Returns:
        A column count between 1 and 6.
    """
    override = _i(props, "grid_columns", 0)
    if override > 0:
        return min(override, 6)
    if width <= 0.0:
        columns = 2
    elif width < 220.0:
        columns = 1
    elif width < 460.0:
        columns = 2
    elif width < 700.0:
        columns = 3
    else:
        columns = 4
    return max(1, min(columns, count))


def _best_index(items: Sequence[Any]) -> int:
    """Return the index of the highest-scoring successful candidate."""
    best = -1
    best_score = float("-inf")
    for i, item in enumerate(items):
        if not _b(item, "ok", True):
            continue
        value = _f(item, "score", float("-inf"))
        if value > best_score:
            best_score = value
            best = i
    return best


def _slot_icon(item: Any, index: int, best: int) -> str:
    """Return the rank/status icon for one grid slot."""
    if not _b(item, "ok", True):
        return ICON_FAILED
    if index == best:
        return ICON_BEST
    if _b(item, "is_pareto") or _b(item, "pareto"):
        return ICON_PARETO
    return ICON_PLAIN


def draw_candidate_grid(
    layout: UILayout,
    props: Any,
    items: Sequence[Any],
    *,
    compact: bool = False,
    width: float = 0.0,
) -> None:
    """Draw the selectable candidate grid.

    Each slot is a two-line cell: a full-width operator button carrying the slot
    number, a rank icon and the short recipe label, then a score bar.  The
    active slot is drawn with ``depress=True`` so it reads as pressed-in the way
    a selected thumbnail does.

    Args:
        layout: Layout to draw into.
        props: The scene property group.
        items: The candidate mirror collection.
        compact: Drop the score bar, for the narrow UV editor sidebar.
        width: Effective region width, from :func:`region_width`; the column
            count adapts to it so the grid stays readable at any sidebar size.
    """
    count = len(items)
    if count == 0:
        draw_empty_state(layout)
        return

    active = _active_index(props, count)
    best = _best_index(items)
    columns = 1 if compact else _grid_columns(props, count, width)

    try:
        grid = layout.grid_flow(
            row_major=True,
            columns=columns,
            even_columns=True,
            even_rows=False,
            align=False,
        )
    except Exception:  # noqa: BLE001
        grid = layout.column(align=True)

    for i, item in enumerate(items):
        try:
            index = _i(item, "index", i)
            label = _s(item, "label", "candidate %d" % (index + 1))
            score = _f(item, "score")
            failed = not _b(item, "ok", True)
            bad = _f(item, "overlap_area") > 0.0 or _i(item, "flipped_tris") > 0

            cell = grid.column(align=True)
            button_row = cell.row(align=True)
            button_row.scale_y = 1.25
            op = draw_op(
                button_row,
                OP_SELECT,
                text="%02d  %s" % (index + 1, label),
                icon=_slot_icon(item, i, best),
                depress=(i == active),
            )
            set_op_props(op, index=index)

            if compact:
                continue

            info = cell.row(align=True)
            info.scale_y = 0.8
            info.alert = failed or bad
            if failed:
                info.label(text="failed", icon=ICON_FAILED)
            else:
                draw_bar(
                    info,
                    "",
                    score / 100.0,
                    ("! %.1f" % score) if bad else ("%.1f" % score),
                    warn=bad,
                    split_factor=0.06,
                )
        except Exception:  # noqa: BLE001 - one bad slot must not eat the grid.
            continue

    _draw_grid_legend(layout, count, best, active)


def _draw_grid_legend(
    layout: UILayout, count: int, best: int, active: int
) -> None:
    """Draw the one-line key that explains the grid's rank icons."""
    try:
        row = layout.row(align=True)
        row.scale_y = 0.85
        row.label(text="", icon=ICON_BEST)
        row.label(text="best")
        row.label(text="", icon=ICON_PARETO)
        row.label(text="pareto")
        row.label(text="%d slots" % count)
        if 0 <= active < count:
            sub = layout.row(align=True)
            sub.scale_y = 0.85
            sub.label(
                text="Active: slot %02d%s"
                % (active + 1, "  (top ranked)" if active == best else "")
            )
    except Exception:  # noqa: BLE001
        pass


def draw_empty_state(layout: UILayout) -> None:
    """Draw the "nothing generated yet" message.

    Args:
        layout: Layout to draw into.
    """
    box = layout.box()
    col = box.column(align=True)
    col.label(text="No candidates yet.", icon="INFO")
    col.label(text="Pick a mesh object and hit")
    col.label(text="Generate Grid to explore unwraps.")


def draw_object_state(layout: UILayout, context: Context) -> bool:
    """Draw the active-object banner and report whether the add-on can run.

    Args:
        layout: Layout to draw into.
        context: Current context.

    Returns:
        ``True`` when the active object is a usable mesh.
    """
    obj = getattr(context, "active_object", None) or getattr(
        context, "object", None
    )
    if obj is None:
        row = layout.row()
        row.alert = True
        row.label(text="No active object.", icon=ICON_WARN)
        return False
    if getattr(obj, "type", "") != "MESH":
        row = layout.row()
        row.alert = True
        row.label(text="'%s' is not a mesh." % obj.name, icon=ICON_WARN)
        return False
    row = layout.row(align=True)
    row.label(text=obj.name, icon="OUTLINER_OB_MESH")
    return True


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class _UVVarViewportPanel:
    """Shared panel configuration for the 3D viewport sidebar."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Return whether the panel should be drawn at all."""
        return getattr(context, "scene", None) is not None


class UVVAR_PT_main(_UVVarViewportPanel, Panel):
    """Root panel: object state, active candidate and the apply/cycle controls."""

    bl_idname = "UVVAR_PT_main"
    bl_label = "Unwrap Variations"
    bl_order = 0

    def draw_header(self, context: Context) -> None:
        """Draw the small icon in the panel header."""
        try:
            self.layout.label(text="", icon="GROUP_UVS")
        except Exception:  # noqa: BLE001
            pass

    def draw(self, context: Context) -> None:
        """Draw the root panel body."""
        layout = self.layout
        try:
            props = get_props(context)
            if props is None:
                box = layout.box()
                box.alert = True
                box.label(text="Scene properties not registered.", icon=ICON_WARN)
                return

            usable = draw_object_state(layout, context)
            items = candidate_items(props)
            count = len(items)
            active = _active_index(props, count)

            if not usable:
                return

            if count == 0:
                draw_empty_state(layout)
                op = draw_op(
                    layout, OP_GENERATE, text="Generate Grid", icon="SEQUENCE"
                )
                set_op_props(op, count=max(1, _i(props, "candidate_count", 8)))
                return

            item = items[active] if 0 <= active < count else None

            head = layout.box().column(align=True)
            head.label(
                text="Slot %02d of %d  -  %s"
                % (active + 1, count, _s(item, "label", "?")),
                icon=ICON_BEST if active == _best_index(items) else ICON_PLAIN,
            )
            draw_bar(
                head,
                "Score",
                _f(item, "score") / 100.0,
                "%.1f" % _f(item, "score"),
            )

            nav = layout.row(align=True)
            nav.scale_y = 1.2
            prev_op = draw_op(nav, OP_CYCLE, text="", icon="TRIA_LEFT")
            set_op_props(prev_op, direction="PREV", delta=-1, offset=-1)
            draw_prop(nav, props, "active_index", text="")
            next_op = draw_op(nav, OP_CYCLE, text="", icon="TRIA_RIGHT")
            set_op_props(next_op, direction="NEXT", delta=1, offset=1)

            apply_row = layout.row(align=True)
            apply_row.scale_y = 1.3
            op = draw_op(
                apply_row, OP_APPLY, text="Apply To UV Map", icon="CHECKMARK"
            )
            set_op_props(op, index=active)
        except Exception:  # noqa: BLE001 - last line of defence.
            _draw_panic(layout)


class UVVAR_PT_generate(_UVVarViewportPanel, Panel):
    """Generation controls: grid size, seed, scoring preset, branch."""

    bl_idname = "UVVAR_PT_generate"
    bl_parent_id = "UVVAR_PT_main"
    bl_label = "Generate"
    bl_order = 1

    def draw(self, context: Context) -> None:
        """Draw the generation controls."""
        layout = self.layout
        try:
            props = get_props(context)
            if props is None:
                layout.label(text="Properties unavailable.", icon=ICON_WARN)
                return

            col = layout.column(align=True)
            draw_prop(col, props, "candidate_count", text="Candidates")
            draw_prop(col, props, "seed", text="Seed")
            draw_prop(col, props, "weight_preset", text="Optimise For")

            row = layout.row(align=True)
            row.scale_y = 1.35
            op = draw_op(row, OP_GENERATE, text="Generate Grid", icon="SEQUENCE")
            set_op_props(
                op,
                count=max(1, _i(props, "candidate_count", 8)),
                seed=_i(props, "seed", 0),
            )

            layout.separator()
            items = candidate_items(props)
            branch_col = layout.column(align=True)
            branch_col.enabled = len(items) > 0
            draw_prop(branch_col, props, "mutation_strength", text="Variation")
            branch_row = branch_col.row(align=True)
            branch_row.scale_y = 1.2
            op = draw_op(
                branch_row, OP_BRANCH, text="More Like This", icon="DUPLICATE"
            )
            set_op_props(
                op,
                count=max(1, _i(props, "candidate_count", 8)),
                strength=_f(props, "mutation_strength", 0.35),
                index=_active_index(props, len(items)),
            )
        except Exception:  # noqa: BLE001
            _draw_panic(layout)


class UVVAR_PT_candidates(_UVVarViewportPanel, Panel):
    """The candidate grid."""

    bl_idname = "UVVAR_PT_candidates"
    bl_parent_id = "UVVAR_PT_main"
    bl_label = "Candidates"
    bl_order = 2

    def draw_header(self, context: Context) -> None:
        """Show the candidate count in the sub-panel header."""
        try:
            count = len(candidate_items(get_props(context)))
            if count:
                self.layout.label(text="%d" % count)
        except Exception:  # noqa: BLE001
            pass

    def draw(self, context: Context) -> None:
        """Draw the grid of selectable candidate slots."""
        layout = self.layout
        try:
            props = get_props(context)
            items = candidate_items(props)
            if props is not None:
                draw_prop(layout, props, "auto_preview", text="Auto Preview")
            draw_candidate_grid(
                layout, props, items, width=region_width(context)
            )
        except Exception:  # noqa: BLE001
            _draw_panic(layout)


class UVVAR_PT_metrics(_UVVarViewportPanel, Panel):
    """Per-candidate quality readout for the active slot."""

    bl_idname = "UVVAR_PT_metrics"
    bl_parent_id = "UVVAR_PT_main"
    bl_label = "Metrics"
    bl_order = 3

    def draw(self, context: Context) -> None:
        """Draw the metric bars and warnings for the active candidate."""
        layout = self.layout
        try:
            props = get_props(context)
            items = candidate_items(props)
            count = len(items)
            if count == 0:
                layout.label(text="Nothing generated yet.", icon="INFO")
                return
            active = _active_index(props, count)
            draw_metric_readout(layout, items[active] if active >= 0 else None)
        except Exception:  # noqa: BLE001
            _draw_panic(layout)


class UVVAR_PT_overlay(_UVVarViewportPanel, Panel):
    """Stretch / angle / checker viewport overlay controls."""

    bl_idname = "UVVAR_PT_overlay"
    bl_parent_id = "UVVAR_PT_main"
    bl_label = "Overlay"
    bl_order = 4

    def draw_header(self, context: Context) -> None:
        """Show a dot when the overlay is currently drawing."""
        try:
            from . import stretch_overlay

            if stretch_overlay.is_enabled():
                self.layout.label(text="", icon="RADIOBUT_ON")
        except Exception:  # noqa: BLE001
            pass

    def draw(self, context: Context) -> None:
        """Draw overlay toggles, mode selector and the colour-ramp legend."""
        layout = self.layout
        try:
            props = get_props(context)
            try:
                from . import stretch_overlay
            except Exception:  # noqa: BLE001
                stretch_overlay = None  # type: ignore[assignment]

            enabled = bool(
                stretch_overlay.is_enabled() if stretch_overlay is not None else False
            )

            row = layout.row(align=True)
            row.scale_y = 1.25
            if op_exists(OP_OVERLAY):
                op = draw_op(
                    row,
                    OP_OVERLAY,
                    text="Stretch Overlay",
                    icon="OVERLAY",
                    depress=enabled,
                )
                set_op_props(op, enable=not enabled)
            elif stretch_overlay is not None:
                op = draw_op(
                    row,
                    stretch_overlay.OP_OVERLAY_CONTROL,
                    text="Stretch Overlay",
                    icon="OVERLAY",
                    depress=enabled,
                )
                set_op_props(op, action="TOGGLE")

            mode = "STRETCH"
            if props is not None and hasattr(props, "overlay_mode"):
                sub = layout.column(align=True)
                draw_prop(sub, props, "overlay_mode", text="")
                mode = _s(props, "overlay_mode", "STRETCH")
            elif stretch_overlay is not None:
                mode = stretch_overlay.get_mode()
                mode_row = layout.row(align=True)
                for name in stretch_overlay.OVERLAY_MODES:
                    op = draw_op(
                        mode_row,
                        stretch_overlay.OP_OVERLAY_CONTROL,
                        text=name.title(),
                        depress=(name == mode),
                    )
                    set_op_props(op, action="SET_MODE", mode=name)

            draw_prop(
                layout, props, "overlay_opacity", text="Opacity", optional=True
            )

            if stretch_overlay is not None:
                stretch_overlay.draw_legend(layout, mode)
                layout.separator()
                box = layout.box()
                col = box.column(align=True)
                col.label(text="Checker Material", icon="TEXTURE")
                active_obj = getattr(context, "active_object", None)
                on = stretch_overlay.checker_material_is_active(active_obj)
                sub = col.row(align=True)
                sub.scale_y = 1.15
                op = draw_op(
                    sub,
                    stretch_overlay.OP_CHECKER_MATERIAL,
                    text="Show Texel Density" if not on else "Restore Materials",
                    icon="TEXTURE_DATA",
                    depress=on,
                )
                set_op_props(op, action="TOGGLE")
                note = col.column(align=True)
                note.scale_y = 0.85
                if on:
                    note.label(text="Original materials are saved.")
                else:
                    note.label(text="Swaps in a generated checker so")
                    note.label(text="texel density is visible directly.")

                refresh = layout.row(align=True)
                op = draw_op(
                    refresh,
                    stretch_overlay.OP_OVERLAY_CONTROL,
                    text="Rebuild Overlay",
                    icon="FILE_REFRESH",
                )
                set_op_props(op, action="REFRESH")
        except Exception:  # noqa: BLE001
            _draw_panic(layout)


class UVVAR_PT_advanced(_UVVarViewportPanel, Panel):
    """Backends, evaluation switches and destructive housekeeping."""

    bl_idname = "UVVAR_PT_advanced"
    bl_parent_id = "UVVAR_PT_main"
    bl_label = "Advanced"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context: Context) -> None:
        """Draw the advanced controls."""
        layout = self.layout
        try:
            props = get_props(context)
            col = layout.column(align=True)
            draw_prop(col, props, "backend_filter", text="Backends")
            draw_prop(col, props, "compute_overlap", text="Detect Overlaps")
            draw_prop(col, props, "auto_preview", text="Auto Preview")

            layout.separator()
            info = layout.column(align=True)
            info.scale_y = 0.85
            info.label(text="Overlap detection is the slow part;")
            info.label(text="turn it off for very dense meshes.")

            layout.separator()
            danger = layout.row(align=True)
            danger.alert = True
            op = draw_op(danger, OP_CLEAR, text="Clear Candidates", icon="TRASH")
            set_op_props(op, confirm=True)
        except Exception:  # noqa: BLE001
            _draw_panic(layout)


def _draw_panic(layout: UILayout) -> None:
    """Last-ditch message used when a whole ``draw()`` body failed.

    Args:
        layout: The panel layout.
    """
    try:
        row = layout.row()
        row.alert = True
        row.label(text="UI error - see console.", icon=ICON_WARN)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: Every registrable class in this module, in registration order.  Parents must
#: precede the sub-panels that name them in ``bl_parent_id``.
classes: tuple[type, ...] = (
    UVVAR_PT_main,
    UVVAR_PT_generate,
    UVVAR_PT_candidates,
    UVVAR_PT_metrics,
    UVVAR_PT_overlay,
    UVVAR_PT_advanced,
)


def register() -> None:
    """Register every panel in this module."""
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception:  # noqa: BLE001 - a re-register must not abort the add-on.
            pass


def unregister() -> None:
    """Unregister every panel in this module, children first."""
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            pass
