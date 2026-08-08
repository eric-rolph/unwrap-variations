"""``uvvar.toggle_overlay`` -- switch the viewport overlay on and off.

The overlay itself (the GPU draw handler that tints faces by stretch, or
projects a checker) belongs to the UI layer; this module owns only the
*operator* that drives it, and the scene property that records the choice.

The two are kept apart on purpose.  The scene property is the single source of
truth: it is what the panel draws, what a ``.blend`` remembers, and what the
draw handler reads.  The operator never talks to the GPU -- it sets the
property, and the property's update callback tells the drawing module.  That
means a keymap, a panel button, a Python one-liner and a loaded file all end up
in exactly the same state, and it means the add-on still registers cleanly if
the drawing module is missing or broken.

Because the drawing module is written by another worker, this module binds to
it by capability rather than by a fixed function name: it looks for the first
of several plausible entry points and degrades to "property only" when none is
found.  That is deliberately loose coupling at a boundary where being strict
would buy nothing and cost a hard dependency.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import bpy
from bpy.props import EnumProperty
from bpy.types import Context, Operator

from .. import properties as uvvar_props
from ..preferences import get_prefs
from ..utils import compat
from ..utils import logging as uvvar_logging

__all__ = [
    "UVVAR_OT_toggle_overlay",
    "sync_overlay_state",
    "shutdown_overlay",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)

#: Entry points tried, in order, to push a mode into the drawing module.
#: The first that exists wins; all are called as ``fn(mode, context)`` with
#: unsupported arguments dropped.
_SET_MODE_NAMES: Sequence[str] = ("set_mode", "set_overlay_mode", "sync", "update", "refresh")

#: Entry points tried, in order, to tear the overlay down.
_DISABLE_NAMES: Sequence[str] = ("disable", "stop", "disable_overlay", "remove_handler",
                                 "unregister_handler", "shutdown")

#: Entry points tried, in order, to stand the overlay up.
_ENABLE_NAMES: Sequence[str] = ("enable", "start", "enable_overlay", "add_handler",
                                "register_handler")

#: The mode a plain toggle turns *on*.  Updated whenever the user picks a
#: specific mode, so toggling off and on again returns to what they were using.
_LAST_ACTIVE_MODE = "STRETCH"


def _overlay_module() -> Any:
    """Return the UI drawing module, or ``None`` when it is unavailable."""
    try:
        from ..ui import overlay

        return overlay
    except Exception:  # noqa: BLE001 - the overlay is optional.
        _LOG.debug("ui.overlay is not importable", exc_info=True)
        return None


def _call_first(module: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> bool:
    """Call the first attribute of ``module`` named in ``names``.

    Arguments that the target does not accept are dropped, so the same call
    site works against ``set_mode(mode)``, ``sync(context)`` and ``refresh()``.

    Args:
        module: The module to search; ``None`` is tolerated.
        names: Candidate attribute names, most specific first.
        *args: Positional arguments (never filtered).
        **kwargs: Keyword arguments; unsupported ones are dropped.

    Returns:
        ``True`` when something was called successfully.
    """
    if module is None:
        return False
    from .generate_grid import call_flexible

    for name in names:
        func: Callable[..., Any] | None = getattr(module, name, None)
        if not callable(func):
            continue
        try:
            call_flexible(func, *args, **kwargs)
            return True
        except Exception:  # noqa: BLE001 - a broken overlay must not break the operator.
            _LOG.debug("ui.overlay.%s failed", name, exc_info=True)
            return False
    return False


def sync_overlay_state(context: Context | None = None) -> str:
    """Push ``scene.uvvar.overlay_mode`` into the drawing module.

    Called from the property's update callback, from the operator, and from
    add-on registration, so that however the mode changed, the drawing module
    ends up agreeing with it.

    Args:
        context: The context to read; ``bpy.context`` is used when omitted.

    Returns:
        The mode that was applied.
    """
    ctx = context if context is not None else bpy.context
    props = uvvar_props.get_props(ctx)
    mode = str(getattr(props, "overlay_mode", "NONE") or "NONE")

    module = _overlay_module()
    if module is None:
        compat.tag_redraw(ctx)
        return mode

    prefs = get_prefs(ctx)
    if mode == "NONE":
        if not _call_first(module, _DISABLE_NAMES, context=ctx):
            _call_first(module, _SET_MODE_NAMES, mode, context=ctx, prefs=prefs)
    else:
        if not _call_first(module, _SET_MODE_NAMES, mode, context=ctx, prefs=prefs):
            _call_first(module, _ENABLE_NAMES, mode, context=ctx, prefs=prefs)
    compat.tag_redraw(ctx)
    return mode


def shutdown_overlay() -> None:
    """Tear the overlay down, ignoring every failure.

    Called from ``unregister``.  A GPU draw handler that survives an add-on
    unload keeps drawing with freed Python state, which is the classic route to
    a Blender crash minutes after the user thought they had disabled the
    add-on.
    """
    _call_first(_overlay_module(), _DISABLE_NAMES)


class UVVAR_OT_toggle_overlay(Operator):
    """Turn the UV inspection overlay on, off, or on to a specific mode."""

    bl_idname = "uvvar.toggle_overlay"
    bl_label = "Toggle UV Overlay"
    bl_description = "Show or hide the stretch and checker overlay for the previewed candidate"
    # No UNDO: this is a viewport display state, like Blender's own overlay
    # toggles, and pushing an undo step for it would be noise.
    bl_options = {"REGISTER"}

    mode: EnumProperty(
        name="Mode",
        description="Overlay to switch to",
        items=(
            ("TOGGLE", "Toggle", "Switch the overlay off, or back on to the last mode used"),
            ("NONE", "Off", "Hide the overlay"),
            ("STRETCH", "Stretch", "Colour faces by UV stretch"),
            ("CHECKER", "Checker", "Project a checker pattern to reveal texel density"),
        ),
        default="TOGGLE",
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled whenever the scene properties exist."""
        return uvvar_props.get_props(context) is not None

    def execute(self, context: Context) -> set[str]:
        """Set the overlay mode and tell the drawing module about it."""
        global _LAST_ACTIVE_MODE
        try:
            props = uvvar_props.get_props(context)
            if props is None:
                self.report({"ERROR"}, "Scene properties are not registered")
                return {"CANCELLED"}

            current = str(props.overlay_mode)
            if self.mode == "TOGGLE":
                target = "NONE" if current != "NONE" else _LAST_ACTIVE_MODE
            else:
                target = str(self.mode)

            if target != "NONE":
                _LAST_ACTIVE_MODE = target

            # Assigning fires the property's update callback, which is what
            # actually reaches the drawing module. sync_overlay_state is called
            # again below so that a scripted call with the value already set
            # still converges.
            props.overlay_mode = target
            sync_overlay_state(context)

            module = _overlay_module()
            if module is None and target != "NONE":
                self.report(
                    {"WARNING"},
                    "Overlay mode set to %s, but the drawing module is not available"
                    % target,
                )
                return {"FINISHED"}

            self.report({"INFO"}, "Overlay: %s" % target.lower())
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001 - nothing escapes into Blender.
            _LOG.exception("toggle_overlay failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}


#: Classes contributed by this module.
classes: tuple[type, ...] = (UVVAR_OT_toggle_overlay,)


def register() -> None:
    """Register this module's operators."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    """Unregister this module's operators and make sure the overlay is down."""
    shutdown_overlay()
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            _LOG.debug("could not unregister %s", cls.__name__, exc_info=True)
