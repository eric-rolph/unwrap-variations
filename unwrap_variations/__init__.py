"""UnwrapVariations -- generate, score and compare many UV unwraps side by side.

This file is the add-on's entry point, and it has three jobs that are easy to
get subtly wrong.

**Dual install modes.**  The same directory is both a legacy add-on (a folder
under ``scripts/addons``, identified by the :data:`bl_info` dict below) and a
Blender 4.2+ extension (identified by ``blender_manifest.toml``, which sits
next to this file).  Blender 4.2 and later ignore ``bl_info`` entirely and read
the manifest instead; older builds do the reverse.  Keeping both means one
source tree installs everywhere, but it also means the package is **not always
called** ``unwrap_variations``: as an extension it is
``bl_ext.<repo>.unwrap_variations``.  Every intra-package import is therefore
relative, and anything keyed on the add-on name uses ``__package__``.

**No ``bpy`` at module scope.**  Importing any core module -- say
``unwrap_variations.core.params`` under plain pytest -- executes this file
first.  A top-level ``import bpy`` here would make the entire bpy-free core
unimportable outside Blender and take the unit-test suite with it.  ``bpy`` is
imported defensively, and every module that genuinely needs it is imported
inside :func:`register`.

**Reload without restarting.**  Add-on development is unbearable if every edit
needs a Blender restart, so the classic ``if "bpy" in locals(): reload`` dance
is implemented properly here: submodules are reloaded in dependency order, so a
module that imported a name from a sibling never ends up holding a stale copy
of it.
"""

from __future__ import annotations

from typing import Any, Callable

#: Human-readable add-on name and version, held as separate constants rather
#: than being read back out of :data:`bl_info`.
#:
#: This is not redundancy, it is a bug fix.  Verified in
#: ``scripts/modules/addon_utils.py`` on **both 4.5.4 and 5.1.2**: when an
#: add-on is enabled *as an extension*, Blender executes ``del mod.bl_info``
#: **before** calling ``register()`` -- deliberately, so that an extension
#: cannot accidentally use stale metadata instead of the manifest.  Any code in
#: ``register()`` that touches ``bl_info`` therefore dies with
#: ``NameError: name 'bl_info' is not defined``, and the add-on fails to
#: install for every extension user while working perfectly from a source
#: checkout.  These constants survive that deletion.
_ADDON_NAME = "Unwrap Variations"
_ADDON_VERSION = (0, 1, 0)

#: Legacy add-on metadata.  Read by Blender 4.0/4.1 and by any tooling that
#: predates extensions; deleted (see above) by 4.2+, which uses
#: ``blender_manifest.toml``.  The version must be kept in step with the
#: manifest and with pyproject.toml.
#:
#: **Every value here must be a literal.**  ``addon_utils.modules()`` populates
#: the Add-ons list without importing anything: it locates the ``bl_info``
#: assignment in the *source text* and runs ``ast.literal_eval`` on it.  A
#: reference to a constant -- even one defined three lines above -- makes that
#: raise ``ValueError: malformed node``, and the add-on then shows up in
#: Blender's list with no name, no description and no version.  Hence the
#: duplication with ``_ADDON_NAME``/``_ADDON_VERSION`` above, which must be
#: kept in step by hand.
bl_info = {
    "name": "Unwrap Variations",
    "author": "Eric Rolph",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport and UV Editor > Sidebar > UV Variations",
    "description": "Generate, score and compare many UV unwraps side by side",
    "warning": "",
    "doc_url": "https://github.com/eric-rolph/unwrap-variations-blender",
    "tracker_url": "https://github.com/eric-rolph/unwrap-variations-blender/issues",
    "support": "COMMUNITY",
    "category": "UV",
}

__all__ = ["bl_info", "register", "unregister"]


# ---------------------------------------------------------------------------
# Development reload
# ---------------------------------------------------------------------------

#: Submodules reloaded first, in this order, so that a module which did
#: ``from .sibling import thing`` at import time is always reloaded *after* the
#: sibling it took ``thing`` from.  Reloading them alphabetically instead is the
#: classic way an add-on ends up with two incompatible copies of one class and
#: an ``isinstance`` check that mysteriously fails.
_RELOAD_ORDER: tuple[str, ...] = (
    "utils",
    "utils.logging",
    "utils.compat",
    "core",
    "core.meshdata",
    "core.params",
    "core.metrics",
    "core.evaluator",
    "core.candidate",
    "core.bridge",
    "core.engine",
    "backends",
    "backends.base",
    "backends.seams",
    "backends.blender_native",
    "backends.xatlas_cli",
    "preferences",
    "properties",
    "operators",
    "operators.generate_grid",
    "operators.select_candidate",
    "operators.branch_variations",
    "operators.apply_candidate",
    "operators.clear_candidates",
    "operators.overlay_ops",
    "ui",
    "ui.stretch_overlay",
    "ui.viewport_panel",
    "ui.uv_editor_panel",
)


def _reload_package_modules() -> None:
    """Reload every already-imported submodule, dependencies first.

    Only modules already present in ``sys.modules`` are touched, so this is a
    no-op on a first import and cannot drag ``bpy`` in under pytest.
    """
    import importlib
    import sys

    prefix = __name__ + "."
    loaded = {name for name in sys.modules if name.startswith(prefix)}

    ordered = [prefix + suffix for suffix in _RELOAD_ORDER if prefix + suffix in loaded]
    remaining = sorted(loaded - set(ordered), key=lambda n: (n.count("."), n), reverse=True)

    for name in ordered + remaining:
        module = sys.modules.get(name)
        if module is None:
            continue
        try:
            importlib.reload(module)
        except Exception as exc:  # noqa: BLE001 - a reload failure must not abort the rest.
            print("[UVVAR] WARNING reload of %s failed: %s" % (name, exc))


# This must run *before* the guarded ``import bpy`` below: on the first
# execution of this module ``bpy`` is not yet a global, and on a reload it is.
if "bpy" in locals():  # pragma: no cover - only reached on an add-on reload.
    _reload_package_modules()

try:
    import bpy
except Exception:  # noqa: BLE001 - pragma: no cover
    # ImportError is the normal case (running outside Blender, e.g. pytest, or
    # with sys.modules["bpy"] = None as CI does). The catch is deliberately
    # wider than ImportError because a stub or partially initialised bpy can
    # raise something else entirely, and *any* failure here must degrade to
    # "not inside Blender" rather than making the bpy-free core unimportable.
    bpy = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Registration bookkeeping
# ---------------------------------------------------------------------------

#: Modules whose ``register``/``unregister`` ran successfully, in the order
#: they ran.  Unregistration walks this in reverse, so a partially completed
#: registration is torn down exactly as far as it got and no further.
_REGISTERED_MODULES: list[Any] = []

#: ``(keymap, keymap_item)`` pairs added by :func:`_register_keymaps`.  Only
#: these exact items are removed on unregister; the keymaps themselves are
#: shared with Blender and every other add-on and must never be deleted.
_KEYMAP_ITEMS: list[tuple[Any, Any]] = []

#: Keyboard shortcuts for stepping through candidates.
#:
#: ``Alt`` plus the bracket keys is chosen because the brackets are unbound in
#: the 3D View and the UV Editor in Blender's default keymap (they are brush
#: size in the paint and sculpt modes, which this add-on has no business in),
#: and because they read as "previous/next" the same way they do in most other
#: applications.
_KEYMAP_SPEC: tuple[tuple[str, str, tuple[dict[str, Any], ...]], ...] = (
    (
        "3D View",
        "VIEW_3D",
        (
            {"key": "RIGHT_BRACKET", "alt": True, "properties": {"direction": "NEXT"}},
            {"key": "LEFT_BRACKET", "alt": True, "properties": {"direction": "PREV"}},
        ),
    ),
    (
        "UV Editor",
        "EMPTY",
        (
            {"key": "RIGHT_BRACKET", "alt": True, "properties": {"direction": "NEXT"}},
            {"key": "LEFT_BRACKET", "alt": True, "properties": {"direction": "PREV"}},
        ),
    ),
)

#: ``bl_idname`` the keymap items are bound to.
_CYCLE_OPERATOR = "uvvar.cycle_candidate"


def _log() -> Any:
    """Return the package logger, importing it lazily."""
    from .utils import logging as uvvar_logging

    return uvvar_logging.get_logger(__name__)


def _module_register_steps() -> list[Any]:
    """Import and return the modules to register, in dependency order.

    Order is not cosmetic:

    * ``preferences`` first, because other modules read defaults from it;
    * ``properties`` next, because it creates ``bpy.types.Scene.uvvar`` and the
      property groups every panel and operator refers to;
    * operators next, and ``generate_grid`` before ``branch_variations``
      because the latter imports the shared modal machinery from it;
    * ``ui`` last, since panels poll operators and draw properties that must
      already exist.

    Returns:
        The modules to register.

    Raises:
        ImportError: If a required module cannot be imported.  The optional
            ``ui`` package is exempt -- a broken panel must not stop the
            operators from working.
    """
    from . import preferences, properties
    from .operators import (
        apply_candidate,
        branch_variations,
        clear_candidates,
        generate_grid,
        overlay_ops,
        select_candidate,
    )

    steps: list[Any] = [
        preferences,
        properties,
        generate_grid,
        select_candidate,
        branch_variations,
        apply_candidate,
        clear_candidates,
        overlay_ops,
    ]

    try:
        from . import ui

        if hasattr(ui, "register") or hasattr(ui, "classes"):
            steps.append(ui)
        else:
            _log().debug("ui package exposes nothing to register yet")
    except Exception as exc:  # noqa: BLE001 - the UI is optional at load time.
        _log().warning("UI package unavailable (%s); operators still registered", exc)

    return steps


def _register_module(module: Any) -> None:
    """Register one module, via its ``register()`` or its ``classes`` tuple."""
    hook: Callable[[], None] | None = getattr(module, "register", None)
    if callable(hook):
        hook()
        return
    for cls in getattr(module, "classes", ()):
        bpy.utils.register_class(cls)


def _unregister_module(module: Any) -> None:
    """Unregister one module, tolerating anything it throws."""
    hook: Callable[[], None] | None = getattr(module, "unregister", None)
    if callable(hook):
        try:
            hook()
        except Exception:  # noqa: BLE001 - keep going; the rest still has to come down.
            _log().exception("unregister failed for %s", getattr(module, "__name__", module))
        return
    for cls in reversed(tuple(getattr(module, "classes", ()))):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001
            _log().debug("could not unregister %s", getattr(cls, "__name__", cls))


# ---------------------------------------------------------------------------
# Keymaps
# ---------------------------------------------------------------------------


def _register_keymaps() -> None:
    """Bind the candidate-cycling shortcuts, remembering exactly what we added.

    Silently does nothing when there is no add-on key configuration, which is
    the case in some headless contexts.  That is not an error: an add-on that
    refuses to register because it cannot bind a shortcut is far worse than one
    without shortcuts.
    """
    window_manager = getattr(bpy.context, "window_manager", None)
    keyconfigs = getattr(window_manager, "keyconfigs", None)
    addon_config = getattr(keyconfigs, "addon", None)
    if addon_config is None:
        _log().debug("no add-on key configuration available; shortcuts skipped")
        return

    for keymap_name, space_type, entries in _KEYMAP_SPEC:
        try:
            keymap = addon_config.keymaps.new(name=keymap_name, space_type=space_type)
        except Exception:  # noqa: BLE001 - an unknown keymap name is not fatal.
            _log().debug("could not open keymap %r", keymap_name, exc_info=True)
            continue
        for entry in entries:
            try:
                item = keymap.keymap_items.new(
                    _CYCLE_OPERATOR,
                    entry["key"],
                    "PRESS",
                    ctrl=bool(entry.get("ctrl", False)),
                    alt=bool(entry.get("alt", False)),
                    shift=bool(entry.get("shift", False)),
                )
                for name, value in entry.get("properties", {}).items():
                    setattr(item.properties, name, value)
                _KEYMAP_ITEMS.append((keymap, item))
            except Exception:  # noqa: BLE001
                _log().debug("could not bind %s in %r", _CYCLE_OPERATOR, keymap_name,
                             exc_info=True)
    _log().debug("registered %d keymap item(s)", len(_KEYMAP_ITEMS))


def _unregister_keymaps() -> None:
    """Remove exactly the keymap items :func:`_register_keymaps` added.

    Removing the *keymaps* would be a bug: they are shared with Blender's own
    defaults and with every other add-on.  Only our own items go, and the list
    is emptied even if a removal fails, so a second unregister cannot try to
    remove an already-freed item and crash.
    """
    for keymap, item in reversed(_KEYMAP_ITEMS):
        try:
            keymap.keymap_items.remove(item)
        except Exception:  # noqa: BLE001 - the item may already be gone.
            pass
    _KEYMAP_ITEMS.clear()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the whole add-on.

    On failure, everything already registered is rolled back before the error
    is re-raised, so a broken build leaves Blender clean rather than half
    populated with classes that can never be unregistered.

    Raises:
        Exception: Whatever the failing step raised, after rollback.
    """
    if bpy is None:  # pragma: no cover - defensive; Blender always has bpy.
        raise RuntimeError("UnwrapVariations requires Blender (bpy is not importable)")

    from .utils import logging as uvvar_logging

    uvvar_logging.configure(debug=False)
    log = uvvar_logging.get_logger(__name__)

    if _REGISTERED_MODULES:
        log.debug("register() called while already registered; unregistering first")
        unregister()

    try:
        steps = _module_register_steps()
    except Exception:
        log.exception("UnwrapVariations could not import its modules")
        raise

    try:
        for module in steps:
            _register_module(module)
            _REGISTERED_MODULES.append(module)
        _register_keymaps()
        log.info(
            "registered %s %s on Blender %s (package %s)",
            _ADDON_NAME,
            ".".join(str(part) for part in _ADDON_VERSION),
            bpy.app.version_string,
            __package__,
        )
    except Exception:
        log.exception("UnwrapVariations registration failed; rolling back")
        _teardown()
        raise


def unregister() -> None:
    """Unregister the whole add-on, in exact reverse order.

    Every step is individually guarded.  A failure part-way through
    unregistration is the one thing that genuinely breaks a Blender session --
    classes that are still registered but whose Python is gone -- so the
    priority here is to keep going, not to report neatly.
    """
    if bpy is None:  # pragma: no cover
        return
    _teardown()
    try:
        from .utils import logging as uvvar_logging

        uvvar_logging.get_logger(__name__).info("unregistered Unwrap Variations")
        uvvar_logging.shutdown()
    except Exception:  # noqa: BLE001 - logging is the least important thing here.
        pass


def _teardown() -> None:
    """Undo whatever :func:`register` managed to do, defensively.

    Shared by the normal unregister path and by the rollback inside
    :func:`register`, which is why it must tolerate any degree of partial
    registration.
    """
    _unregister_keymaps()
    while _REGISTERED_MODULES:
        _unregister_module(_REGISTERED_MODULES.pop())
