"""Add-on preferences: defaults, budgets, external tools and overlay colours.

These are the settings that belong to the *installation* rather than to a
scene: how many candidates a fresh Generate should produce, how much of a
``.blend`` the add-on is allowed to occupy, where the optional ``xatlas``
binary lives, and what the viewport overlay looks like.

Two details are worth knowing before editing this file:

* ``bl_idname`` **must** equal the add-on's package name, and that name differs
  between install modes (``unwrap_variations`` for a legacy add-on,
  ``bl_ext.<repo>.unwrap_variations`` for a 4.2+ extension).  It is taken from
  :data:`unwrap_variations.utils.compat.ADDON_PACKAGE`, which derives it at
  import time, so both modes work without a second code path.
* Preferences may legitimately be **absent**.  Registering the classes directly
  from a script -- which is exactly what the headless test harness does --
  never creates an entry in ``bpy.context.preferences.addons``.  Every consumer
  therefore goes through :func:`get_prefs`, which falls back to a plain object
  carrying the same defaults instead of raising ``KeyError``.
"""

from __future__ import annotations

import os
from typing import Any

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Context

from .utils import logging as uvvar_logging
from .utils.compat import ADDON_PACKAGE as _DERIVED_PACKAGE
from .utils.compat import XATLAS_ENV_VAR, xatlas_binary

__all__ = [
    "UVVAR_AddonPreferences",
    "ADDON_PACKAGE",
    "get_prefs",
    "prefs_dict",
    "max_persist_bytes",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)

#: The add-on's package name, and therefore the key its preferences live under.
#: This module sits directly inside the add-on package, so ``__package__`` *is*
#: that name -- ``unwrap_variations`` for a legacy add-on install and
#: ``bl_ext.<repo>.unwrap_variations`` for a 4.2+ extension install.  It is
#: taken from ``__package__`` rather than hard-coded because the literal string
#: is wrong in exactly the case that matters: a real user's extension install.
ADDON_PACKAGE: str = __package__ or _DERIVED_PACKAGE

#: Default persistence budget in megabytes.  Matches
#: ``core.candidate.DEFAULT_MAX_BYTES`` and ARCHITECTURE.md section 7.
DEFAULT_BUDGET_MB = 64

#: Seed behaviour choices for a fresh Generate.
SEED_MODES = (
    (
        "FIXED",
        "Fixed",
        "Always start from the seed stored in the scene, so a Generate is exactly repeatable",
    ),
    (
        "RANDOM",
        "Random",
        "Pick a new random seed for every Generate, for a different set of ideas each time",
    ),
    (
        "INCREMENT",
        "Increment",
        "Advance the scene seed by one on every Generate, walking the space in steps",
    ),
)


def _on_debug_changed(self: "UVVAR_AddonPreferences", _context: Context) -> None:
    """Apply the debug-logging preference to the package logger immediately."""
    uvvar_logging.set_debug(bool(self.debug_logging))
    _LOG.debug("debug logging enabled")


def _on_xatlas_changed(self: "UVVAR_AddonPreferences", _context: Context) -> None:
    """Publish the xatlas path into the environment for the CLI backend.

    The out-of-process backend reads ``UVVAR_XATLAS_BIN`` so that it works
    identically when driven from a plain script with no add-on preferences.
    Mirroring the preference into the environment keeps that single source of
    truth without the backend having to know about ``bpy`` at all.
    """
    path = (self.xatlas_bin or "").strip()
    try:
        if path:
            os.environ[XATLAS_ENV_VAR] = bpy.path.abspath(path)
        else:
            os.environ.pop(XATLAS_ENV_VAR, None)
    except Exception:  # noqa: BLE001 - a preference edit must never raise.
        _LOG.debug("could not publish %s", XATLAS_ENV_VAR, exc_info=True)


class UVVAR_AddonPreferences(AddonPreferences):
    """User-level settings for UnwrapVariations.

    ``bl_idname`` is ``__package__``: Blender looks preferences up by the
    module name it enabled, which differs between a legacy add-on install and
    an extension install.  Hard-coding ``"unwrap_variations"`` would leave every
    extension user with silently unreachable preferences.
    """

    bl_idname = ADDON_PACKAGE

    # -- generation defaults -------------------------------------------------

    default_candidate_count: IntProperty(
        name="Default Candidates",
        description="How many variations a fresh Generate produces",
        default=10,
        min=1,
        soft_max=24,
        max=64,
    )

    default_seed_mode: EnumProperty(
        name="Seed Behaviour",
        description="What happens to the seed each time Generate runs",
        items=SEED_MODES,
        default="FIXED",
    )

    default_seed: IntProperty(
        name="Default Seed",
        description="Seed a newly opened scene starts from",
        default=0,
        min=0,
        max=2 ** 30,
    )

    default_mutation_strength: FloatProperty(
        name="Default Branch Strength",
        description="How far a branch strays from its parent, 0 = tiny tweaks, 1 = wild",
        default=0.35,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )

    # -- cost controls -------------------------------------------------------

    compute_overlap: BoolProperty(
        name="Compute Overlap Metric",
        description=(
            "Measure overlapping UV area. This is the most expensive metric "
            "(a spatial-hash pass over every triangle); turn it off for a much "
            "faster generate on dense meshes"
        ),
        default=True,
    )

    persist_candidates: BoolProperty(
        name="Save Candidates In .blend",
        description=(
            "Store generated candidates in the mesh so they survive save and "
            "reload. Disable to keep .blend files small"
        ),
        default=True,
    )

    persist_budget_mb: IntProperty(
        name="Persistence Budget (MB)",
        description=(
            "Maximum size of the candidate blob written into a .blend. Past "
            "this only metadata is stored and you are told about it, so a big "
            "exploration can never silently bloat the file"
        ),
        default=DEFAULT_BUDGET_MB,
        min=0,
        soft_max=512,
        max=4096,
    )

    backend_timeout_s: FloatProperty(
        name="Backend Timeout (s)",
        description="Wall-clock budget handed to each backend for one candidate",
        default=60.0,
        min=1.0,
        soft_max=600.0,
        max=3600.0,
    )

    # -- external tools ------------------------------------------------------

    xatlas_bin: StringProperty(
        name="xatlas Binary",
        description=(
            "Path to an xatlas command-line executable. Leave empty to use the "
            "UVVAR_XATLAS_BIN environment variable. xatlas does not ship inside "
            "Blender, so this backend is optional"
        ),
        default="",
        subtype="FILE_PATH",
        update=_on_xatlas_changed,
    )

    # -- overlay appearance --------------------------------------------------

    overlay_color_low: FloatVectorProperty(
        name="Low Distortion",
        description="Colour used where the unwrap is close to isometric",
        subtype="COLOR",
        size=3,
        default=(0.10, 0.45, 0.90),
        min=0.0,
        max=1.0,
    )

    overlay_color_high: FloatVectorProperty(
        name="High Distortion",
        description="Colour used where stretch is worst",
        subtype="COLOR",
        size=3,
        default=(0.95, 0.20, 0.10),
        min=0.0,
        max=1.0,
    )

    overlay_color_flipped: FloatVectorProperty(
        name="Flipped Faces",
        description="Colour marking triangles that are mirrored in UV space",
        subtype="COLOR",
        size=3,
        default=(1.00, 0.85, 0.10),
        min=0.0,
        max=1.0,
    )

    overlay_color_seam: FloatVectorProperty(
        name="Seams",
        description="Colour used to draw the seams a candidate produced",
        subtype="COLOR",
        size=3,
        default=(0.95, 0.35, 0.85),
        min=0.0,
        max=1.0,
    )

    overlay_opacity: FloatProperty(
        name="Overlay Opacity",
        description="Blend strength of the viewport overlay",
        default=0.65,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )

    overlay_checker_size: IntProperty(
        name="Checker Size",
        description="Squares per UV unit in the checker overlay",
        default=16,
        min=1,
        soft_max=64,
        max=256,
    )

    # -- diagnostics ---------------------------------------------------------

    debug_logging: BoolProperty(
        name="Verbose Logging",
        description="Print detailed progress and timing to the system console",
        default=False,
        update=_on_debug_changed,
    )

    # -- drawing -------------------------------------------------------------

    def draw(self, context: Context) -> None:
        """Lay the preferences out in the Add-ons panel."""
        layout = self.layout

        box = layout.box()
        box.label(text="Generation Defaults", icon="PRESET")
        col = box.column(align=True)
        col.prop(self, "default_candidate_count")
        col.prop(self, "default_seed_mode")
        row = col.row()
        row.enabled = self.default_seed_mode != "RANDOM"
        row.prop(self, "default_seed")
        col.prop(self, "default_mutation_strength")

        box = layout.box()
        box.label(text="Cost And Storage", icon="MEMORY")
        col = box.column(align=True)
        col.prop(self, "compute_overlap")
        col.prop(self, "backend_timeout_s")
        col.prop(self, "persist_candidates")
        sub = col.row()
        sub.enabled = self.persist_candidates
        sub.prop(self, "persist_budget_mb")

        box = layout.box()
        box.label(text="External Backends", icon="PLUGIN")
        box.prop(self, "xatlas_bin")
        resolved = xatlas_binary(self.xatlas_bin)
        if resolved:
            state = "found" if os.path.isfile(resolved) else "NOT FOUND"
            box.label(text="Resolved: %s (%s)" % (resolved, state), icon="FILE_TICK"
                      if state == "found" else "ERROR")
        else:
            box.label(text="xatlas backend disabled (no path, no %s)" % XATLAS_ENV_VAR,
                      icon="INFO")

        box = layout.box()
        box.label(text="Overlay Colours", icon="COLOR")
        grid = box.grid_flow(row_major=True, columns=2, even_columns=True)
        grid.prop(self, "overlay_color_low")
        grid.prop(self, "overlay_color_high")
        grid.prop(self, "overlay_color_flipped")
        grid.prop(self, "overlay_color_seam")
        col = box.column(align=True)
        col.prop(self, "overlay_opacity")
        col.prop(self, "overlay_checker_size")

        box = layout.box()
        box.label(text="Diagnostics", icon="CONSOLE")
        box.prop(self, "debug_logging")


class _FallbackPrefs:
    """Stand-in preferences used when the add-on is not formally installed.

    Registering the classes straight from a script (the headless test harness,
    a developer's ``exec(open(...).read())`` loop) leaves
    ``bpy.context.preferences.addons`` without an entry for us.  Rather than
    scattering ``if prefs is None`` through every operator, :func:`get_prefs`
    hands back one of these, carrying exactly the class defaults.
    """

    default_candidate_count = 10
    default_seed_mode = "FIXED"
    default_seed = 0
    default_mutation_strength = 0.35
    compute_overlap = True
    persist_candidates = True
    persist_budget_mb = DEFAULT_BUDGET_MB
    backend_timeout_s = 60.0
    xatlas_bin = ""
    overlay_color_low = (0.10, 0.45, 0.90)
    overlay_color_high = (0.95, 0.20, 0.10)
    overlay_color_flipped = (1.00, 0.85, 0.10)
    overlay_color_seam = (0.95, 0.35, 0.85)
    overlay_opacity = 0.65
    overlay_checker_size = 16
    debug_logging = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Return a marker showing these are defaults, not user settings."""
        return "<UVVAR fallback preferences>"


_FALLBACK = _FallbackPrefs()


def get_prefs(context: Context | None = None) -> Any:
    """Return the add-on preferences, or a defaults-only stand-in.

    Args:
        context: Optional context; ``bpy.context`` is used when omitted.

    Returns:
        The registered :class:`UVVAR_AddonPreferences`, or a
        :class:`_FallbackPrefs` exposing the same attribute names with their
        default values.  Never ``None``, never raises.
    """
    ctx = context if context is not None else bpy.context
    try:
        entry = ctx.preferences.addons.get(ADDON_PACKAGE)
        if entry is not None and entry.preferences is not None:
            return entry.preferences
    except Exception:  # noqa: BLE001 - preferences access must never break an operator.
        _LOG.debug("preferences lookup failed for %r", ADDON_PACKAGE, exc_info=True)
    return _FALLBACK


def prefs_dict(context: Context | None = None) -> dict[str, Any]:
    """Return the preferences as a plain dict, for logging and diagnostics."""
    prefs = get_prefs(context)
    out: dict[str, Any] = {}
    for name in dir(_FallbackPrefs):
        if name.startswith("_"):
            continue
        try:
            value = getattr(prefs, name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, (int, float, bool, str)):
            out[name] = value
        else:
            try:
                out[name] = tuple(value)
            except Exception:  # noqa: BLE001
                pass
    return out


def max_persist_bytes(context: Context | None = None) -> int:
    """Return the persistence budget in bytes, honouring the on/off switch.

    Returns:
        ``0`` when persistence is disabled, otherwise the megabyte preference
        converted to bytes.
    """
    prefs = get_prefs(context)
    if not bool(getattr(prefs, "persist_candidates", True)):
        return 0
    return int(max(0, int(getattr(prefs, "persist_budget_mb", DEFAULT_BUDGET_MB)))) * 1024 * 1024


#: Classes this module contributes, in registration order.
classes: tuple[type, ...] = (UVVAR_AddonPreferences,)


def register() -> None:
    """Register the preferences class and apply its side effects."""
    for cls in classes:
        bpy.utils.register_class(cls)
    prefs = get_prefs()
    uvvar_logging.set_debug(bool(getattr(prefs, "debug_logging", False)))
    path = (getattr(prefs, "xatlas_bin", "") or "").strip()
    if path:
        try:
            os.environ[XATLAS_ENV_VAR] = bpy.path.abspath(path)
        except Exception:  # noqa: BLE001
            _LOG.debug("could not publish %s at register", XATLAS_ENV_VAR, exc_info=True)


def unregister() -> None:
    """Unregister the preferences class, tolerating a partial registration."""
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001 - keep going; a leftover class is worse.
            _LOG.debug("could not unregister %s", cls.__name__, exc_info=True)
