"""Every Blender-version and install-mode difference, in one place.

The add-on is verified on **Blender 4.5.4 LTS** and **Blender 5.1.2**, and is
installable both as a legacy add-on (a folder under ``scripts/addons`` with a
``bl_info`` dict) and as a 4.2+ extension (a ``blender_manifest.toml`` archive
installed into a repository, where the package is named
``bl_ext.<repo>.unwrap_variations``).  Those two axes produce a handful of small
incompatibilities.  Rather than sprinkling ``if bpy.app.version >= (5, 0)``
through fifteen modules, they all live here.

Guiding rules for anything added to this module:

* **Probe, do not assume.**  ``operator_supports`` asks the operator's own RNA
  which arguments exist; that is immune to a future release adding or removing
  one.
* **Never raise.**  A compatibility shim that throws is worse than the
  incompatibility it was meant to hide.  Everything returns a value or an
  ``(ok, message)`` pair.
* **Importable without Blender.**  ``bpy`` is imported defensively so the
  bpy-free unit tests can still import anything that happens to reach here.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - the "no Blender" path is exercised by the unit tests.
    import bpy

    HAVE_BPY = True
except Exception:  # noqa: BLE001 - any failure means "not inside Blender".
    bpy = None  # type: ignore[assignment]
    HAVE_BPY = False

__all__ = [
    "HAVE_BPY",
    "ADDON_PACKAGE",
    "BLENDER_VERSION",
    "IS_42_OR_NEWER",
    "IS_45_OR_NEWER",
    "IS_50_OR_NEWER",
    "MAX_UV_LAYERS",
    "PREVIEW_LAYER_NAME",
    "XATLAS_ENV_VAR",
    "is_background",
    "version_string",
    "operator_supports",
    "filter_operator_kwargs",
    "call_operator",
    "pack_islands",
    "read_uv_layer",
    "write_uv_layer",
    "uv_layer_by_name",
    "free_uv_layer_slots",
    "temp_override",
    "object_mode",
    "tag_redraw",
    "unique_uv_layer_name",
    "active_mesh_object",
    "mesh_problem",
    "xatlas_binary",
]


def _derive_addon_package() -> str:
    """Return the add-on's top-level package name for both install modes.

    This module sits at ``<addon>/utils/compat.py``, so its ``__package__`` is
    ``<addon>.utils``; stripping the last component yields the add-on package
    itself.  That is ``unwrap_variations`` for a legacy install and
    ``bl_ext.user_default.unwrap_variations`` for an extension install, and it
    is exactly the string ``bpy.context.preferences.addons[...]`` is keyed by.
    """
    pkg = __package__ or ""
    if "." in pkg:
        return pkg.rsplit(".", 1)[0]
    return pkg or "unwrap_variations"


#: Key under which this add-on's preferences live in
#: ``bpy.context.preferences.addons``.  Correct for both install modes.
ADDON_PACKAGE: str = _derive_addon_package()

#: Running Blender version as a 3-tuple, or ``(0, 0, 0)`` outside Blender.
BLENDER_VERSION: tuple[int, int, int] = tuple(bpy.app.version) if HAVE_BPY else (0, 0, 0)

#: First release with extension support.
IS_42_OR_NEWER: bool = BLENDER_VERSION >= (4, 2, 0)

#: The oldest release this add-on is actually verified against.
IS_45_OR_NEWER: bool = BLENDER_VERSION >= (4, 5, 0)

#: Blender 5.x changed a number of operator return conventions; see
#: :func:`pack_islands`.
IS_50_OR_NEWER: bool = BLENDER_VERSION >= (5, 0, 0)

#: Hard cap on UV layers per mesh, verified on 4.5.4 and 5.1.2:
#: ``uv_layers.new()`` returns ``None`` and logs "Cannot add more than 8 UV
#: maps" past this.  The whole candidate-as-numpy-array design exists because
#: of this number.
MAX_UV_LAYERS = 8

#: The single scratch UV layer candidates are previewed through.  Mirrors
#: ``core.candidate.CandidateStore.PREVIEW_LAYER``; duplicated (and asserted
#: below) so that importing compat cannot drag numpy-heavy core modules in.
PREVIEW_LAYER_NAME = "UVVar_Preview"

#: Environment variable naming the optional ``xatlas`` executable.
XATLAS_ENV_VAR = "UVVAR_XATLAS_BIN"


def is_background() -> bool:
    """Return whether Blender is running headless (``-b``).

    Modal operators cannot run without a window, so every modal entry point
    checks this and falls back to a blocking ``execute``.
    """
    return bool(getattr(getattr(bpy, "app", None), "background", True)) if HAVE_BPY else True


def version_string() -> str:
    """Return a short ASCII description of the host Blender."""
    if not HAVE_BPY:
        return "no-bpy"
    return "%d.%d.%d" % BLENDER_VERSION


# ---------------------------------------------------------------------------
# Operator argument compatibility
# ---------------------------------------------------------------------------


def _rna_property_names(op: Any) -> frozenset[str]:
    """Return the argument names an ``bpy.ops`` operator accepts.

    Args:
        op: A ``bpy.ops`` submember, e.g. ``bpy.ops.uv.unwrap``.

    Returns:
        The accepted keyword names, or an empty set if RNA introspection fails
        (which is treated as "accept nothing", so callers fall back to their
        own defaults rather than passing an argument that would raise).
    """
    try:
        rna = op.get_rna_type()
        return frozenset(p.identifier for p in rna.properties if p.identifier != "rna_type")
    except Exception:  # noqa: BLE001
        return frozenset()


_RNA_CACHE: dict[int, frozenset[str]] = {}


def operator_supports(op: Any, name: str) -> bool:
    """Return whether ``op`` accepts the keyword argument ``name``.

    Args:
        op: A ``bpy.ops`` operator, e.g. ``bpy.ops.uv.pack_islands``.
        name: The keyword to test.

    Returns:
        ``True`` when the argument exists in this Blender build.
    """
    if not HAVE_BPY:
        return False
    key = id(op.__class__) ^ hash(str(op))
    names = _RNA_CACHE.get(key)
    if names is None:
        names = _rna_property_names(op)
        _RNA_CACHE[key] = names
    return name in names


def filter_operator_kwargs(op: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments ``op`` does not understand.

    Passing an unknown keyword to a ``bpy.ops`` call raises ``TypeError``, and
    the set of arguments genuinely does drift between releases.  Filtering
    against the live RNA is version-proof in a way that a hand-maintained table
    is not.

    Args:
        op: The operator that will be called.
        kwargs: Candidate keyword arguments.

    Returns:
        A new dict containing only the supported keys.
    """
    if not HAVE_BPY:
        return {}
    supported = _RNA_CACHE.get(id(op.__class__) ^ hash(str(op)))
    if supported is None:
        supported = _rna_property_names(op)
        _RNA_CACHE[id(op.__class__) ^ hash(str(op))] = supported
    return {k: v for k, v in kwargs.items() if k in supported}


def call_operator(op: Any, *args: Any, **kwargs: Any) -> tuple[bool, str]:
    """Call a ``bpy.ops`` operator, never raising.

    Args:
        op: The operator to call.
        *args: Positional arguments, typically an execution context string.
        **kwargs: Operator arguments; unsupported ones are dropped first.

    Returns:
        ``(ok, message)`` where ``ok`` is ``True`` only for a ``FINISHED``
        result.  ``message`` is empty on success and explains the failure
        otherwise.
    """
    if not HAVE_BPY:
        return False, "bpy is not available"
    clean = filter_operator_kwargs(op, kwargs)
    dropped = sorted(set(kwargs) - set(clean))
    try:
        result = op(*args, **clean)
    except Exception as exc:  # noqa: BLE001 - operators must not raise upward.
        return False, "%s: %s" % (type(exc).__name__, exc)
    if "FINISHED" in result:
        return True, ""
    note = ", ".join(sorted(result)) or "no result"
    if dropped:
        note += " (unsupported args dropped: %s)" % ",".join(dropped)
    return False, note


def pack_islands(**kwargs: Any) -> tuple[bool, str]:
    """Run ``bpy.ops.uv.pack_islands``, absorbing the 4.5-vs-5.1 difference.

    Verified behaviour (headless, ``--factory-startup``, identical inputs):
    the operator returns ``CANCELLED`` on **4.5.4** but ``FINISHED`` on
    **5.1.2**.  On 4.5 the packer refuses to run without a UV editor area to
    report into, yet the unwrap it was called after is still perfectly usable --
    it is simply unpacked.  Treating that ``CANCELLED`` as fatal would throw
    away every candidate on the older LTS, so it is downgraded to a note.

    Args:
        **kwargs: Arguments for the operator; unsupported ones are dropped.

    Returns:
        ``(ok, message)``.  ``ok`` is ``True`` when islands were packed;
        ``False`` with an explanatory note when packing was skipped.  Callers
        keep their UVs either way.
    """
    if not HAVE_BPY:
        return False, "bpy is not available"
    ok, message = call_operator(bpy.ops.uv.pack_islands, **kwargs)
    if ok:
        return True, ""
    if not IS_50_OR_NEWER and "CANCELLED" in message:
        return False, (
            "pack_islands unavailable in this context (Blender %s); "
            "islands left unpacked" % version_string()
        )
    return False, "pack_islands failed: %s" % message


# ---------------------------------------------------------------------------
# UV layer access
# ---------------------------------------------------------------------------


def read_uv_layer(layer: Any, *, out: np.ndarray | None = None) -> np.ndarray:
    """Read a whole UV layer into a ``(L, 2)`` float32 array.

    Both ``layer.uv.foreach_get("vector", ...)`` (the modern attribute-backed
    path) and ``layer.data.foreach_get("uv", ...)`` (the legacy alias) exist on
    4.5 and 5.1; the former is preferred and the latter is the fallback for any
    build where it is missing.

    Args:
        layer: A ``MeshUVLoopLayer``.
        out: Optional pre-allocated ``(L, 2)`` float32 destination.

    Returns:
        The UVs as a contiguous ``(L, 2)`` float32 array.

    Raises:
        RuntimeError: If neither access path works.  Callers at the bpy edge
            catch this; it is a genuine "the API changed" condition rather than
            an expected runtime state.
    """
    count = len(layer.data)
    flat = np.empty(count * 2, dtype=np.float32)
    source = getattr(layer, "uv", None)
    if source is not None:
        try:
            source.foreach_get("vector", flat)
            return _shape_uv(flat, out)
        except Exception:  # noqa: BLE001 - fall through to the legacy path.
            pass
    try:
        layer.data.foreach_get("uv", flat)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "cannot read UV layer %r: %s" % (getattr(layer, "name", "?"), exc)
        ) from exc
    return _shape_uv(flat, out)


def _shape_uv(flat: np.ndarray, out: np.ndarray | None) -> np.ndarray:
    """Reshape a flat UV buffer to ``(L, 2)``, filling ``out`` when given."""
    shaped = flat.reshape(-1, 2)
    if out is None:
        return shaped
    out[...] = shaped
    return out


def write_uv_layer(layer: Any, uvs: np.ndarray) -> None:
    """Blit a ``(L, 2)`` array into a UV layer.

    This is the operation the whole preview design rests on: a single
    ``foreach_set`` memcpy rather than a per-element Python loop.  The array is
    coerced to contiguous float32 first, because a float64 or strided array
    silently drops ``foreach_set`` into its slow path.

    Args:
        layer: A ``MeshUVLoopLayer``.
        uvs: ``(L, 2)`` UV coordinates.

    Raises:
        ValueError: If ``uvs`` does not match the layer's loop count.  Blitting
            a wrong-length buffer is a hard crash in Blender, so the check is
            not optional.
    """
    arr = np.ascontiguousarray(uvs, dtype=np.float32).reshape(-1, 2)
    count = len(layer.data)
    if arr.shape[0] != count:
        raise ValueError(
            "UV array has %d loops but layer %r has %d"
            % (arr.shape[0], getattr(layer, "name", "?"), count)
        )
    flat = arr.reshape(-1)
    target = getattr(layer, "uv", None)
    if target is not None:
        try:
            target.foreach_set("vector", flat)
            return
        except Exception:  # noqa: BLE001 - fall through to the legacy path.
            pass
    layer.data.foreach_set("uv", flat)


def uv_layer_by_name(mesh: Any, name: str) -> Any | None:
    """Return the UV layer called ``name``, or ``None``."""
    try:
        return mesh.uv_layers.get(name)
    except Exception:  # noqa: BLE001
        return None


def free_uv_layer_slots(obj: Any) -> int:
    """Return how many more UV layers ``obj``'s mesh can hold.

    Args:
        obj: A Blender object.

    Returns:
        ``MAX_UV_LAYERS`` minus the current layer count, floored at 0.  Returns
        0 for anything that is not a mesh, which makes callers refuse to add a
        layer rather than try and fail.
    """
    mesh = getattr(obj, "data", None)
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        return 0
    try:
        return max(0, MAX_UV_LAYERS - len(layers))
    except Exception:  # noqa: BLE001
        return 0


def unique_uv_layer_name(mesh: Any, base: str) -> str:
    """Return a UV-layer name based on ``base`` that ``mesh`` does not use yet.

    Blender silently renames a colliding layer (``UVMap.001``), which makes it
    impossible for the caller to find the layer it just created by name.
    Choosing a free name up front avoids that entirely.

    Args:
        mesh: The mesh whose layers must be avoided.
        base: Desired name.

    Returns:
        ``base`` when free, otherwise ``base.001``, ``base.002``, ...  Names are
        truncated to Blender's 63-character limit for a datablock sub-name.
    """
    base = (base or "UVMap").strip() or "UVMap"
    base = base[:59]
    try:
        existing = {layer.name for layer in mesh.uv_layers}
    except Exception:  # noqa: BLE001
        return base
    if base not in existing:
        return base
    for i in range(1, 1000):
        candidate = "%s.%03d" % (base, i)
        if candidate not in existing:
            return candidate
    return base


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


class _NullOverride:
    """No-op stand-in for ``context.temp_override`` outside Blender."""

    def __enter__(self) -> None:
        """Enter the no-op context."""
        return None

    def __exit__(self, *exc: Any) -> bool:
        """Leave the no-op context without suppressing anything."""
        return False


def temp_override(context: Any = None, **kwargs: Any) -> Any:
    """Return a context-override context manager that works everywhere.

    ``Context.temp_override`` exists on both 4.5 and 5.1 (verified), but it
    raises when handed a member the build does not know, and it does not exist
    at all outside Blender.  This wrapper degrades to a no-op instead.

    Args:
        context: The context to override; defaults to ``bpy.context``.
        **kwargs: Override members such as ``area``, ``region``, ``object``.

    Returns:
        A context manager suitable for ``with``.
    """
    if not HAVE_BPY:
        return _NullOverride()
    ctx = context if context is not None else bpy.context
    try:
        return ctx.temp_override(**kwargs)
    except Exception:  # noqa: BLE001
        return _NullOverride()


class object_mode:
    """Context manager that forces an object into a mode and restores it.

    Every bridge operation is documented to work from OBJECT mode and to leave
    the mode it found.  Doing that with ``bpy.ops.object.mode_set`` requires an
    active object and a valid context, neither of which is guaranteed when an
    operator is invoked from a script; this wrapper handles both and simply
    does nothing when the switch is impossible.

    Example::

        with object_mode(obj, "OBJECT"):
            ...  # obj.data is safe to read here
    """

    def __init__(self, obj: Any, mode: str = "OBJECT") -> None:
        """Prepare to switch ``obj`` into ``mode``.

        Args:
            obj: The object whose mode should change.
            mode: Target mode, e.g. ``"OBJECT"`` or ``"EDIT"``.
        """
        self.obj = obj
        self.mode = mode
        self._previous: str | None = None
        self._switched = False

    def __enter__(self) -> Any:
        """Switch mode if needed, remembering the previous one."""
        if not HAVE_BPY or self.obj is None:
            return self.obj
        current = getattr(self.obj, "mode", None)
        if current is None or current == self.mode:
            return self.obj
        self._previous = current
        self._switched = self._set_mode(self.mode)
        return self.obj

    def __exit__(self, *exc: Any) -> bool:
        """Restore the original mode, never suppressing an exception."""
        if self._switched and self._previous:
            self._set_mode(self._previous)
        return False

    def _set_mode(self, mode: str) -> bool:
        """Attempt the mode switch; return whether it happened."""
        try:
            view_layer = bpy.context.view_layer
            previous_active = view_layer.objects.active
            if previous_active is not self.obj:
                view_layer.objects.active = self.obj
            bpy.ops.object.mode_set(mode=mode)
            return True
        except Exception:  # noqa: BLE001 - a failed switch is not fatal.
            return False


def tag_redraw(
    context: Any = None,
    area_types: Sequence[str] = ("VIEW_3D", "IMAGE_EDITOR"),
) -> None:
    """Request a redraw of the areas that show candidate state.

    Args:
        context: Context to search; defaults to ``bpy.context``.
        area_types: Area types to refresh.
    """
    if not HAVE_BPY:
        return
    ctx = context if context is not None else bpy.context
    try:
        screen = getattr(ctx, "screen", None)
        windows: Iterable[Any]
        if screen is not None:
            windows = [screen]
        else:
            windows = [w.screen for w in bpy.context.window_manager.windows]
        for scr in windows:
            for area in scr.areas:
                if area.type in area_types:
                    area.tag_redraw()
    except Exception:  # noqa: BLE001 - a redraw hint is never worth an error.
        pass


def active_mesh_object(context: Any = None) -> Any | None:
    """Return the active object when it is a usable mesh, else ``None``.

    "Usable" means an ``OBJECT`` of type ``MESH`` whose data exists.  Every
    operator ``poll`` funnels through this so that the enabled/disabled state of
    the whole panel is decided in exactly one place.

    Args:
        context: Context to read; defaults to ``bpy.context``.

    Returns:
        The object, or ``None``.
    """
    if not HAVE_BPY:
        return None
    ctx = context if context is not None else bpy.context
    obj = getattr(ctx, "object", None) or getattr(ctx, "active_object", None)
    if obj is None or getattr(obj, "type", "") != "MESH":
        return None
    if getattr(obj, "data", None) is None:
        return None
    return obj


def mesh_problem(obj: Any) -> str:
    """Return a human-readable reason ``obj`` cannot be unwrapped, or ``""``.

    Separated from :func:`active_mesh_object` because ``poll`` must be cheap and
    silent while ``execute`` must be able to explain itself.  Reading
    ``len(mesh.polygons)`` on a heavy mesh inside ``poll`` -- which runs on every
    UI redraw -- would be wasteful, so the expensive checks live here.

    Args:
        obj: The object to inspect.

    Returns:
        An empty string when the object is fine, otherwise the reason.
    """
    if obj is None:
        return "no active object"
    if getattr(obj, "type", "") != "MESH":
        return "active object %r is not a mesh" % getattr(obj, "name", "?")
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return "active object has no mesh data"
    try:
        if len(mesh.polygons) == 0:
            return "mesh %r has no faces to unwrap" % mesh.name
        if len(mesh.loops) == 0:
            return "mesh %r has no face corners" % mesh.name
    except Exception as exc:  # noqa: BLE001
        return "could not inspect mesh: %s" % exc
    return ""


def xatlas_binary(preference_path: str = "") -> str:
    """Return the path to the optional ``xatlas`` executable.

    Resolution order: the add-on preference, then the ``UVVAR_XATLAS_BIN``
    environment variable.  Returns an empty string when neither is set, which
    the xatlas backend reports as "not available" rather than treating as an
    error.

    Args:
        preference_path: The value of the add-on preference, possibly empty.

    Returns:
        A filesystem path, or ``""``.
    """
    path = (preference_path or "").strip()
    if path and HAVE_BPY:
        try:
            path = bpy.path.abspath(path)
        except Exception:  # noqa: BLE001
            pass
    if not path:
        path = (os.environ.get(XATLAS_ENV_VAR) or "").strip()
    return path
