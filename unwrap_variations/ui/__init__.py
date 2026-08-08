"""The user interface package: sidebar panels and the viewport overlay.

This file exists to answer one question -- *what does the add-on register when
it draws itself* -- and it is the reason a user sees anything at all after
installing.  The add-on's :func:`unwrap_variations.register` imports this
package and calls :func:`register` on it; without the wiring below the
operators register perfectly and the N-panel stays empty, which is
indistinguishable from a failed install.

Registration order is a contract, not a preference:

1. :mod:`.stretch_overlay` first.  It owns ``uvvar.overlay_control`` and
   ``uvvar.checker_material``, and :class:`~.viewport_panel.UVVAR_PT_overlay`
   draws buttons for both.  A panel that draws an unregistered operator falls
   back to a disabled placeholder, so the overlay panel would silently lose its
   controls if this came second.
2. :mod:`.viewport_panel` next.  It defines ``UVVAR_PT_main``, which every
   other viewport sub-panel names in ``bl_parent_id``; Blender resolves that
   reference at registration time.
3. :mod:`.uv_editor_panel` last.  It imports its drawing helpers from
   :mod:`.viewport_panel`, so that module must be importable and registered
   before it.

Unregistration walks the exact reverse, and only over the modules that actually
registered -- a partially registered UI is torn down exactly as far as it got.
A failure in any one module is logged and does not abort the others: a broken
panel must never take the operators down with it.
"""

from __future__ import annotations

from typing import Any

from ..utils import logging as uvvar_logging

__all__ = [
    "modules",
    "panel_classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)

#: Modules that registered successfully, in the order they did so.
_REGISTERED: list[Any] = []


def modules() -> list[Any]:
    """Import and return the UI modules to register, in dependency order.

    Imported inside the function rather than at module scope so that importing
    ``unwrap_variations.ui`` -- which the add-on does before it knows whether a
    UI is even wanted -- cannot drag ``bpy`` into a plain-CPython process.

    Returns:
        The modules to register, first to last.

    Raises:
        ImportError: If a UI module cannot be imported at all.
    """
    from . import stretch_overlay, uv_editor_panel, viewport_panel

    return [stretch_overlay, viewport_panel, uv_editor_panel]


def panel_classes() -> tuple[type, ...]:
    """Return every class this package registers, for tests and diagnostics."""
    found: list[type] = []
    for module in modules():
        found.extend(getattr(module, "classes", ()))
    return tuple(found)


def register() -> None:
    """Register every UI module, tolerating a failure in any one of them.

    A module that raises is logged with its traceback and skipped.  The
    alternative -- letting the exception reach
    :func:`unwrap_variations.register` -- rolls back the whole add-on, so one
    bad panel would cost the user every operator too.
    """
    if _REGISTERED:
        unregister()
    try:
        ordered = modules()
    except Exception:  # noqa: BLE001 - a missing UI module must not kill the add-on.
        _LOG.exception("UI modules could not be imported; panels will be missing")
        return
    for module in ordered:
        try:
            module.register()
        except Exception:  # noqa: BLE001
            _LOG.exception("UI module %s failed to register", module.__name__)
            continue
        _REGISTERED.append(module)
    _LOG.debug("registered %d UI module(s)", len(_REGISTERED))


def unregister() -> None:
    """Unregister the UI modules in reverse, tolerating anything they throw."""
    while _REGISTERED:
        module = _REGISTERED.pop()
        try:
            module.unregister()
        except Exception:  # noqa: BLE001 - keep going; the rest still has to come down.
            _LOG.exception("UI module %s failed to unregister", module.__name__)
