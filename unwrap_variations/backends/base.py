"""Backend contract, result types and registry for UnwrapVariations.

A *backend* is anything that can turn a mesh plus an :class:`UnwrapParams`
recipe into a ``(L, 2)`` array of per-loop UV coordinates.  The built-in backend
drives Blender's own operators; optional backends shell out to third-party
tools such as ``xatlas``.

Two rules make the whole search loop safe, and every backend must obey them:

1. :meth:`UnwrapBackend.unwrap` **never raises**.  A failure is reported as
   ``UnwrapResult(ok=False, message=...)`` so that one bad candidate out of ten
   costs the user one slot, not the whole run and certainly not Blender.
2. :meth:`UnwrapBackend.unwrap` **leaves the scene exactly as it found it** --
   mode, selection, seams, active UV layer and every UV layer's contents.

This module imports only stdlib, numpy and the bpy-free core, so it can be
imported and unit-tested outside Blender.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..core.meshdata import UVMeshSample
from ..core.params import METHODS, UnwrapParams

__all__ = [
    "UnwrapRequest",
    "UnwrapResult",
    "UnwrapBackend",
    "register_backend",
    "get_backend",
    "available_backends",
    "backend_ids",
    "BUILTIN_BACKEND_MODULES",
]


#: Sibling modules that are imported on demand so their ``@register_backend``
#: decorators run.  Keeping this list here -- rather than in ``__init__`` --
#: means merely importing :mod:`unwrap_variations.backends.base` outside Blender
#: does not drag :mod:`bpy` in until somebody actually asks for a backend.
BUILTIN_BACKEND_MODULES: tuple[str, ...] = ("blender_native", "xatlas_cli")


# ---------------------------------------------------------------------------
# Data carried across the backend boundary
# ---------------------------------------------------------------------------


@dataclass
class UnwrapRequest:
    """Everything a backend needs to produce one candidate UV layout.

    Attributes:
        obj_name: Name of the Blender object to unwrap.  Backends that touch
            :mod:`bpy` look the object up by name rather than holding a
            reference, because Blender may reallocate an ``Object`` struct
            behind Python's back (an undo push invalidates old pointers).
        params: The recipe to realise.
        sample: A pre-extracted, bpy-free snapshot of the mesh.  The engine
            extracts this exactly once per generation run and shares it across
            every candidate, so backends must treat it as read-only.  Backends
            that do not need Blender at all (``xatlas``) work purely from it.
        timeout_s: Wall-clock budget.  Out-of-process backends must honour it;
            in-process ones use it only as advice.
    """

    obj_name: str
    params: UnwrapParams
    sample: UVMeshSample
    timeout_s: float = 60.0

    @property
    def n_loops(self) -> int:
        """Number of loops the returned UV array must have."""
        return self.sample.n_loops


@dataclass
class UnwrapResult:
    """The outcome of one backend invocation.

    Attributes:
        uvs: ``(L, 2)`` float array of per-loop UVs, or ``None`` when the
            attempt failed.  ``L`` matches ``UnwrapRequest.sample.n_loops``.
        ok: ``True`` only when ``uvs`` is usable.
        message: Human-readable detail.  Populated on failure with the reason,
            and on success with any non-fatal notes (for example "packing
            skipped").  Shown in the UI next to the candidate.
        seconds: Wall-clock time the attempt took.
        backend: Id of the backend that produced this result.
    """

    uvs: np.ndarray | None
    ok: bool
    message: str = ""
    seconds: float = 0.0
    backend: str = ""

    @classmethod
    def failure(cls, message: str, *, backend: str = "", seconds: float = 0.0) -> "UnwrapResult":
        """Return a failed result carrying ``message``."""
        return cls(uvs=None, ok=False, message=str(message), seconds=float(seconds),
                   backend=backend)

    @classmethod
    def success(
        cls,
        uvs: np.ndarray,
        *,
        backend: str = "",
        seconds: float = 0.0,
        message: str = "",
    ) -> "UnwrapResult":
        """Return a successful result wrapping ``uvs`` as a ``(L, 2)`` float32 array."""
        arr = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
        return cls(uvs=arr, ok=True, message=str(message), seconds=float(seconds),
                   backend=backend)

    def validated(self, n_loops: int) -> "UnwrapResult":
        """Return this result, downgraded to a failure if its shape is wrong.

        The engine trusts ``uvs`` to be blittable straight into a UV layer with
        ``foreach_set``; a wrong length there is a hard Blender crash, so the
        check happens here rather than at the memcpy.
        """
        if not self.ok or self.uvs is None:
            return self
        if self.uvs.ndim != 2 or self.uvs.shape[1] != 2 or self.uvs.shape[0] != int(n_loops):
            shape = "x".join(str(s) for s in self.uvs.shape)
            return UnwrapResult.failure(
                "backend returned UVs of shape %s, expected %dx2" % (shape, int(n_loops)),
                backend=self.backend,
                seconds=self.seconds,
            )
        return self


# ---------------------------------------------------------------------------
# The backend interface
# ---------------------------------------------------------------------------


class UnwrapBackend(ABC):
    """Abstract base for anything that can generate a UV layout.

    Subclasses set the three class variables, implement :meth:`unwrap`, and
    register themselves with the :func:`register_backend` decorator.

    Attributes:
        id: Stable identifier stored in :attr:`UnwrapParams.backend` and in
            persisted candidates.  Never change it once shipped.
        label: Human-readable name for the UI.
        requires_bpy: ``True`` when the backend can only run inside Blender.
    """

    id: ClassVar[str] = ""
    label: ClassVar[str] = ""
    requires_bpy: ClassVar[bool] = True

    # -- capability reporting ------------------------------------------------

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return ``(usable, reason)`` for this backend right now.

        The UI greys out unavailable backends and shows ``reason`` verbatim, so
        the string must be actionable ("xatlas binary not found; set
        UVVAR_XATLAS_BIN") rather than a bare "unavailable".  When the backend
        is usable, ``reason`` may be an empty string or a short note.

        This is a classmethod because the UI has to ask before it is willing to
        construct anything.  It must be cheap and must never raise.
        """
        return True, ""

    def supported_methods(self) -> tuple[str, ...]:
        """Return the :data:`unwrap_variations.core.params.METHODS` this backend honours.

        The engine skips any candidate whose ``method`` is not listed, rather
        than handing the backend a recipe it cannot follow.
        """
        return tuple(METHODS)

    # -- the actual work -----------------------------------------------------

    @abstractmethod
    def unwrap(self, req: UnwrapRequest) -> UnwrapResult:
        """Produce one UV layout for ``req``.

        Args:
            req: The object, recipe and mesh snapshot to work from.

        Returns:
            An :class:`UnwrapResult`.  On success ``uvs`` has shape
            ``(req.sample.n_loops, 2)``.

        Implementations **must not raise** and **must not leave the Blender
        scene modified**, no matter how the attempt fails.
        """
        raise NotImplementedError

    def safe_unwrap(self, req: UnwrapRequest) -> UnwrapResult:
        """Call :meth:`unwrap` behind a belt-and-braces guard.

        :meth:`unwrap` is already required not to raise; this wrapper exists so
        that a third-party backend which breaks that promise still cannot abort
        a generation run.  It also fills in ``backend``/``seconds`` and checks
        the returned shape.
        """
        start = time.perf_counter()
        try:
            result = self.unwrap(req)
        except BaseException as exc:  # noqa: BLE001 - containment is the point
            return UnwrapResult.failure(
                "%s raised %s: %s" % (self.id, type(exc).__name__, exc),
                backend=self.id,
                seconds=time.perf_counter() - start,
            )
        if not isinstance(result, UnwrapResult):
            return UnwrapResult.failure(
                "%s returned %r, expected UnwrapResult" % (self.id, type(result).__name__),
                backend=self.id,
                seconds=time.perf_counter() - start,
            )
        if not result.backend:
            result.backend = self.id
        if not result.seconds:
            result.seconds = time.perf_counter() - start
        return result.validated(req.n_loops)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Return a short debug representation."""
        return "<%s id=%r>" % (type(self).__name__, self.id)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Registered backend classes, keyed by :attr:`UnwrapBackend.id`.
_REGISTRY: dict[str, type[UnwrapBackend]] = {}

#: Lazily constructed singletons, keyed by id.  Backends are stateless, so one
#: instance per id is enough and saves re-probing availability.
_INSTANCES: dict[str, UnwrapBackend] = {}

#: Guard so the on-demand import of the built-in modules happens only once.
_BUILTINS_LOADED = False

#: Import failures from :func:`_load_builtin_backends`, surfaced by
#: :func:`available_backends` instead of being swallowed.
_IMPORT_ERRORS: dict[str, str] = {}


def register_backend(cls: type) -> type:
    """Class decorator registering ``cls`` under its :attr:`UnwrapBackend.id`.

    Args:
        cls: An :class:`UnwrapBackend` subclass with a non-empty ``id``.

    Returns:
        ``cls`` unchanged, so the decorator is transparent.

    Raises:
        TypeError: If ``cls`` is not an :class:`UnwrapBackend` subclass.
        ValueError: If ``cls.id`` is empty.

    Re-registering an existing id replaces the previous class; that is what
    makes an add-on reload work rather than accumulating stale duplicates.
    """
    if not isinstance(cls, type) or not issubclass(cls, UnwrapBackend):
        raise TypeError("register_backend expects an UnwrapBackend subclass, got %r" % (cls,))
    backend_id = getattr(cls, "id", "")
    if not backend_id:
        raise ValueError("backend %s has an empty id" % cls.__name__)
    _REGISTRY[backend_id] = cls
    _INSTANCES.pop(backend_id, None)
    return cls


def _load_builtin_backends() -> None:
    """Import the sibling backend modules so their decorators run.

    Import errors are recorded rather than raised: a broken optional backend
    must not stop the built-in one from working.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import importlib

    for name in BUILTIN_BACKEND_MODULES:
        try:
            importlib.import_module("." + name, __package__)
        except BaseException as exc:  # noqa: BLE001 - report, never propagate
            _IMPORT_ERRORS[name] = "%s: %s" % (type(exc).__name__, exc)


def backend_ids() -> list[str]:
    """Return every registered backend id, built-ins included, sorted stably.

    ``blender_native`` sorts first because it is the default; the rest follow
    alphabetically.
    """
    _load_builtin_backends()
    ids = sorted(_REGISTRY)
    if "blender_native" in ids:
        ids.remove("blender_native")
        ids.insert(0, "blender_native")
    return ids


def get_backend(backend_id: str) -> UnwrapBackend:
    """Return the singleton instance of the backend registered as ``backend_id``.

    Args:
        backend_id: The id to look up.

    Returns:
        A ready-to-use :class:`UnwrapBackend`.

    Raises:
        KeyError: If no backend with that id is registered.  This is a
            programming error (a typo in a recipe), not a runtime condition, so
            unlike :meth:`UnwrapBackend.unwrap` it does raise.
    """
    _load_builtin_backends()
    instance = _INSTANCES.get(backend_id)
    if instance is not None:
        return instance
    cls = _REGISTRY.get(backend_id)
    if cls is None:
        known = ", ".join(backend_ids()) or "<none>"
        raise KeyError("unknown backend %r; registered: %s" % (backend_id, known))
    instance = cls()
    _INSTANCES[backend_id] = instance
    return instance


def available_backends() -> list[tuple[str, str, bool, str]]:
    """Return ``(id, label, ok, why)`` for every registered backend.

    ``ok`` comes from :meth:`UnwrapBackend.is_available`; ``why`` is its
    explanation, which the UI shows as a tooltip on a greyed-out entry.  A
    backend whose module failed to import at all is reported here too, so a
    broken optional dependency is visible rather than silently missing.
    """
    _load_builtin_backends()
    rows: list[tuple[str, str, bool, str]] = []
    for backend_id in backend_ids():
        cls = _REGISTRY[backend_id]
        label = getattr(cls, "label", "") or backend_id
        try:
            ok, why = cls.is_available()
            ok = bool(ok)
            why = str(why)
        except BaseException as exc:  # noqa: BLE001 - a probe must never break the UI
            ok, why = False, "availability check raised %s: %s" % (type(exc).__name__, exc)
        rows.append((backend_id, label, ok, why))
    for name, err in sorted(_IMPORT_ERRORS.items()):
        rows.append((name, name, False, "module failed to import -- " + err))
    return rows


# ---------------------------------------------------------------------------
# Shared helpers for backend implementations
# ---------------------------------------------------------------------------


@dataclass
class _Notes:
    """Accumulator for non-fatal messages, joined into ``UnwrapResult.message``."""

    items: list[str] = field(default_factory=list)

    def add(self, text: str, *args: Any) -> None:
        """Append one note, ``%``-formatted against ``args`` when given."""
        self.items.append(text % args if args else text)

    def text(self) -> str:
        """Return the notes as one ``'; '``-joined string."""
        return "; ".join(self.items)

    def __bool__(self) -> bool:
        """Return whether any note was recorded."""
        return bool(self.items)
