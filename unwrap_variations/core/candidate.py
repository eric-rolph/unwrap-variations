"""Candidate records and their per-object store, including .blend persistence.

A *candidate* is one complete unwrap result: the recipe that produced it, the
raw UV array, the quality metrics and a score.  The UVs are held as a plain
``numpy`` array rather than as a Blender UV layer because Blender caps a mesh at
**8 UV layers** (verified on 4.5.4 and 5.1.2 -- ``uv_layers.new()`` simply
returns ``None`` past that).  A 10- or 24-slot exploration grid therefore cannot
live in UV layers at all.  Keeping candidates as arrays also means switching
preview is a single ``foreach_set`` memcpy (measured at 0.4 ms for one million
loops) instead of a layer rebuild.

:class:`CandidateStore` owns the runtime set for one object and can serialise it
into a mesh ID-property so the exploration survives save/load.  The blob format
is deliberately explicit and version-tagged: a ``.blend`` written by a future
version must fail *cleanly* in an older add-on rather than crash it.

This module imports **no** :mod:`bpy`.  The two functions that touch a Blender
datablock (:meth:`CandidateStore.save_to_id`, :meth:`CandidateStore.load_from_id`)
use only duck-typed attribute access, which keeps the whole module unit-testable
in plain CPython with a tiny fake object.
"""

from __future__ import annotations

import json
import logging
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from .params import UnwrapParams

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime.
    from .metrics import ScoreWeights, UVMetrics

# ``core.metrics`` is written in parallel with this module.  Importing it
# defensively means candidate.py stays importable (and testable) even while that
# file is briefly missing; every use site degrades to the cached score instead.
try:  # pragma: no cover - trivial import shim.
    from .metrics import ScoreWeights as _ScoreWeights  # noqa: F401
    from .metrics import UVMetrics as _UVMetrics
    from .metrics import score as _score_fn

    HAVE_METRICS = True
except Exception:  # noqa: BLE001 - any failure means "not available yet".
    _UVMetrics = None  # type: ignore[assignment]
    _ScoreWeights = None  # type: ignore[assignment]
    _score_fn = None  # type: ignore[assignment]
    HAVE_METRICS = False

__all__ = [
    "Candidate",
    "CandidateStore",
    "STORE_MAGIC",
    "STORE_VERSION",
    "DEFAULT_MAX_BYTES",
]

_LOG = logging.getLogger("unwrap_variations.candidate")

#: Leading bytes of every persisted blob.  Present so a foreign or corrupt
#: ID-property of the same name is rejected instantly instead of being fed to
#: zlib and producing a confusing traceback.
STORE_MAGIC = b"UVVARST1"

#: Bumped whenever the on-disk layout changes incompatibly.  Loading refuses
#: anything newer than it understands.
STORE_VERSION = 1

#: Default persistence budget (64 MB), matching the add-on preference default.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024

#: zlib level used for the payload.  Measured inside Blender on 8 MB of *real*
#: unwrapped UVs (a 1,000,000-loop grid, ANGLE_BASED): level 1 = 0.06 s / 31.1 %,
#: level 3 = 0.07 s / 29.9 %, level 6 = 0.15 s / 27.0 %, level 9 = 0.22 s /
#: 27.0 %.  Real UV layouts are far from random -- adjacent loops share
#: coordinates -- so the higher levels genuinely pay, unlike on synthetic random
#: data where every level lands near 90 %.  Level 6 is chosen because this runs
#: on save, not on the interactive preview path, and the whole point of
#: persisting at all is to not bloat the user's ``.blend``: 13 % smaller output
#: for 0.09 s per 8 MB is the right side of that trade.  Level 9 buys nothing.
_ZLIB_LEVEL = 6

#: How far over budget the *raw* payload may be before compression is skipped
#: outright.  The best ratio ever measured is 27 %, so anything beyond about 3.7x
#: the budget cannot fit however well it compresses; 8x leaves a wide margin so
#: that a store which *would* have fitted is never wrongly demoted to metadata.
#: This exists only to avoid spending seconds compressing a hopeless payload.
_COMPRESSION_BAILOUT = 8

#: Payload dtype.  Explicit little-endian so a blob written on one platform
#: reads back identically on another, rather than depending on native order.
_UV_DTYPE = np.dtype("<f4")


@dataclass
class Candidate:
    """One unwrap result: recipe, UVs, metrics and score.

    Attributes:
        index: Position in the store; also the value the UI list mirrors.
        params: The recipe that produced ``uvs``.
        uvs: ``(L, 2)`` float32 UV per loop, or ``None`` when the backend
            failed.  Always stored C-contiguous float32 so that previewing is a
            straight memcpy into the UV layer with no dtype conversion.
        metrics: Quality measurements, or ``None`` when evaluation was skipped.
        score: Cached 0..100 score under the weights last used for ranking.
        ok: ``False`` when the backend failed; such candidates are kept so the
            user can see *what* was tried, not silently dropped.
        message: Human-readable failure or warning text.
        parent: ``signature`` of the candidate this one was branched from.
        generation: Branch depth; 0 for a top-level grid.
        seconds: Wall time the backend spent producing this candidate.
    """

    index: int
    params: UnwrapParams
    uvs: np.ndarray | None
    metrics: "UVMetrics | None"
    score: float = 0.0
    ok: bool = True
    message: str = ""
    parent: str | None = None
    generation: int = 0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        """Normalise ``uvs`` to contiguous float32 ``(L, 2)``.

        Done once, at construction, because every preview afterwards depends on
        the array already being in the exact layout ``foreach_set`` can memcpy.
        A float64 or strided array would silently fall back to a per-element
        conversion loop and destroy the "instant preview" promise.
        """
        self.uvs = _coerce_uvs(self.uvs)

    @property
    def signature(self) -> str:
        """Return the stable 12-character signature of this candidate's recipe."""
        try:
            return self.params.signature()
        except Exception:  # noqa: BLE001 - identity must never break the UI.
            return "?" * 12

    @property
    def n_loops(self) -> int:
        """Return the loop count of the stored UVs, or 0 when there are none."""
        return 0 if self.uvs is None else int(self.uvs.shape[0])

    def label(self) -> str:
        """Return a short list-row label, e.g. ``"03  ABF sharp m0.01"``."""
        return "%02d  %s" % (self.index, self.params.describe())

    # -- serialisation -------------------------------------------------------

    def header_dict(self) -> dict[str, Any]:
        """Return the JSON-safe part of this candidate (everything but the UVs).

        The UV payload is deliberately excluded: it is many megabytes of float32
        and belongs in the binary section of the blob, not in JSON, where it
        would inflate roughly 6x as text.
        """
        metrics: dict[str, Any] | None = None
        if self.metrics is not None:
            try:
                metrics = self.metrics.to_dict()
            except Exception:  # noqa: BLE001
                metrics = None
        return {
            "index": int(self.index),
            "params": self.params.to_dict(),
            "metrics": metrics,
            "score": float(self.score),
            "ok": bool(self.ok),
            "message": str(self.message),
            "parent": self.parent,
            "generation": int(self.generation),
            "seconds": float(self.seconds),
        }

    @classmethod
    def from_header_dict(
        cls, data: Mapping[str, Any], uvs: np.ndarray | None = None
    ) -> "Candidate":
        """Rebuild a candidate from :meth:`header_dict` plus its UV slice.

        Tolerant by design -- a ``.blend`` written by a newer add-on may carry
        keys this build has never heard of, and that must not stop the file from
        opening.
        """
        raw_metrics = data.get("metrics")
        metrics = None
        if raw_metrics and HAVE_METRICS and _UVMetrics is not None:
            try:
                metrics = _UVMetrics.from_dict(raw_metrics)
            except Exception:  # noqa: BLE001
                _LOG.debug("could not rebuild metrics from stored dict", exc_info=True)
        return cls(
            index=int(data.get("index", 0)),
            params=UnwrapParams.from_dict(data.get("params") or {}),
            uvs=uvs,
            metrics=metrics,
            score=float(data.get("score", 0.0) or 0.0),
            ok=bool(data.get("ok", True)),
            message=str(data.get("message", "") or ""),
            parent=data.get("parent") or None,
            generation=int(data.get("generation", 0) or 0),
            seconds=float(data.get("seconds", 0.0) or 0.0),
        )


def _coerce_uvs(uvs: Any) -> np.ndarray | None:
    """Return ``uvs`` as a contiguous ``(L, 2)`` float32 array, or ``None``.

    Never raises: a malformed array becomes ``None`` (a failed candidate) rather
    than an exception escaping into a Blender operator.
    """
    if uvs is None:
        return None
    try:
        arr = np.asarray(uvs)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=_UV_DTYPE)
        arr = np.ascontiguousarray(arr, dtype=_UV_DTYPE).reshape(-1, 2)
        return arr
    except Exception:  # noqa: BLE001
        _LOG.warning("candidate UV array rejected; storing as failed", exc_info=True)
        return None


@dataclass
class _StoreHeader:
    """Decoded JSON header of a persisted store."""

    version: int
    obj_name: str
    loop_count: int
    has_uvs: bool
    candidates: list[dict[str, Any]] = field(default_factory=list)
    created: float = 0.0
    note: str = ""


class CandidateStore:
    """The candidate set belonging to one object.

    Two layers of state, on purpose:

    * a **runtime** list of :class:`Candidate` holding live numpy arrays -- this
      is what preview and ranking read, and it is never serialised on the hot
      path;
    * an **optional** persisted blob in a mesh ID-property so the exploration
      survives save/load.

    Persistence is opt-in and budgeted.  Ten candidates on a one-million-loop
    mesh is 80 MB of raw float32; writing that into every ``.blend`` without
    asking would be indefensible, so :meth:`save_to_id` drops the UV payload and
    keeps metadata only when the budget is exceeded, and says so.
    """

    #: Name of the single scratch UV layer used for previewing candidates.
    PREVIEW_LAYER = "UVVar_Preview"

    #: ID-property key on the *mesh* datablock holding the persisted blob.
    ID_KEY = "uvvar_store_v1"

    def __init__(self, obj_name: str, loop_count: int) -> None:
        """Create an empty store bound to one object and loop count.

        Args:
            obj_name: Name of the owning object, for diagnostics.
            loop_count: Loop count the candidate UV arrays must match.  Stored
                so a later topology change can be *detected* rather than
                silently blitting the wrong number of UVs into the mesh.
        """
        self.obj_name = str(obj_name or "")
        self.loop_count = int(loop_count or 0)
        self._items: list[Candidate] = []
        #: Reason for the most recent failed or partial operation.  Present
        #: because the contract's ``-> bool`` return has no room to explain
        #: *why* a load was rejected, and the user has to be told.
        self.last_message = ""
        #: True when the last :meth:`load_from_id` found a blob whose loop count
        #: no longer matches the mesh.
        self.was_stale = False

    # -- basic container behaviour ------------------------------------------

    def __len__(self) -> int:
        """Return the number of candidates held."""
        return len(self._items)

    def __iter__(self):
        """Iterate the candidates in index order."""
        return iter(self._items)

    def __bool__(self) -> bool:
        """Return whether the store holds any candidate at all."""
        return bool(self._items)

    def set(self, cands: Sequence[Candidate]) -> None:
        """Replace the whole set, renumbering ``index`` to match position."""
        self._items = list(cands)
        self._renumber()

    def add(self, c: Candidate) -> None:
        """Append one candidate, assigning it the next index."""
        c.index = len(self._items)
        self._items.append(c)

    def extend(self, cands: Iterable[Candidate]) -> None:
        """Append several candidates, renumbering as they land."""
        for c in cands:
            self.add(c)

    def get(self, index: int) -> Candidate | None:
        """Return the candidate at ``index``, or ``None`` if out of range.

        Returns ``None`` rather than raising because the UI's active index can
        legitimately point past the end after a Clear, and an ``IndexError``
        surfacing from a draw callback would spam the console every redraw.
        """
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(self._items):
            return self._items[idx]
        return None

    def all(self) -> list[Candidate]:
        """Return a shallow copy of the candidate list in index order."""
        return list(self._items)

    def clear(self) -> None:
        """Drop every candidate and release the UV arrays."""
        self._items = []
        self.last_message = ""
        self.was_stale = False

    def _renumber(self) -> None:
        """Rewrite ``index`` on every candidate to match list position."""
        for i, c in enumerate(self._items):
            c.index = i

    # -- ranking -------------------------------------------------------------

    def ranked(self, weights: "ScoreWeights") -> list[Candidate]:
        """Return the candidates best-first under ``weights``.

        Re-scores from the stored metrics rather than trusting the cached
        ``score``, because the user can change the weight preset at any time and
        the ranking has to follow immediately.  The freshly computed value is
        written back into :attr:`Candidate.score` so the UI list mirrors what it
        just sorted by.

        Failed candidates (``ok=False`` or no metrics) always sort last, in
        stable index order, so a broken backend never displaces a real result.
        """
        for c in self._items:
            if c.metrics is not None and _score_fn is not None:
                try:
                    c.score = float(_score_fn(c.metrics, weights))
                except Exception:  # noqa: BLE001 - scoring must not break the UI.
                    _LOG.debug("score() failed for candidate %d", c.index, exc_info=True)

        def key(c: Candidate) -> tuple[int, float, int]:
            usable = 1 if (c.ok and c.metrics is not None) else 0
            return (-usable, -float(c.score), c.index)

        return sorted(self._items, key=key)

    def best(self, weights: "ScoreWeights") -> Candidate | None:
        """Return the single highest-scoring candidate, or ``None`` when empty."""
        order = self.ranked(weights)
        return order[0] if order else None

    # -- size accounting -----------------------------------------------------

    def payload_bytes(self) -> int:
        """Return the uncompressed size in bytes of all UV arrays.

        This is *not* the persisted size -- real UV layouts compress to roughly
        27 % -- so :meth:`save_to_id` uses it only to recognise a payload that is
        hopelessly over budget, never to make the budget decision itself.
        """
        total = 0
        for c in self._items:
            if c.uvs is not None:
                total += int(c.uvs.size) * _UV_DTYPE.itemsize
        return total

    # -- persistence ---------------------------------------------------------

    @classmethod
    def for_object(cls, obj: Any) -> "CandidateStore":
        """Build a store for ``obj``, loading any persisted blob it carries.

        This is the one bpy-flavoured entry point, but it only duck-types
        ``obj.data`` and ``len(mesh.loops)``, so a test can pass a stub.

        Args:
            obj: A Blender object (expected to wrap a mesh).

        Returns:
            A store, empty if the object had no blob or the blob was stale.
        """
        name = str(getattr(obj, "name", "") or "")
        mesh = getattr(obj, "data", None)
        loops = getattr(mesh, "loops", None)
        try:
            loop_count = len(loops) if loops is not None else 0
        except Exception:  # noqa: BLE001
            loop_count = 0
        store = cls(name, loop_count)
        store.load_from_id(obj)
        return store

    def save_to_id(self, obj: Any, *, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[bool, str]:
        """Persist this store into ``obj.data[ID_KEY]``.

        The blob is ``STORE_MAGIC`` + a version byte + one zlib stream holding a
        length-prefixed JSON header followed by the raw float32 UV payload.  One
        compressed stream (rather than two) keeps decoding to a single
        ``zlib.decompress`` on load, which is the path that runs when a user
        opens a file.

        Verified on Blender 4.5.4 and 5.1.2: a ``bytes`` ID-property of 1 KB
        through 64 MB survives a real ``save_as_mainfile`` /``open_mainfile``
        cycle byte-for-byte, so this genuinely persists rather than merely
        existing in memory.

        Args:
            obj: The object whose mesh receives the blob.
            max_bytes: Hard budget for the *final* blob.  Over budget, the UV
                payload is dropped and only metadata is written, so a huge
                exploration can never silently inflate the user's ``.blend``.

        Returns:
            ``(True, message)`` when everything including the UVs was written;
            ``(False, reason)`` when the store was reduced to metadata or could
            not be written at all.
        """
        mesh = getattr(obj, "data", None)
        if mesh is None:
            return False, "object has no mesh data; nothing persisted"

        budget = int(max_bytes) if max_bytes and max_bytes > 0 else DEFAULT_MAX_BYTES

        raw = self.payload_bytes()
        # The budget applies to the *compressed* blob, so the only sound test is
        # to compress and measure.  This pre-check merely avoids burning seconds
        # on a payload so far over budget that no ratio could rescue it.
        want_uvs = raw <= budget * _COMPRESSION_BAILOUT
        reason = ""
        if not want_uvs:
            reason = (
                "UV payload %s cannot fit the %s budget even compressed; stored metadata only"
                % (_human_bytes(raw), _human_bytes(budget))
            )

        try:
            blob = self._encode(include_uvs=want_uvs, note=reason)
        except Exception as exc:  # noqa: BLE001 - never raise into Blender.
            _LOG.warning("candidate store encode failed", exc_info=True)
            self.last_message = "could not encode candidate store: %s" % exc
            return False, self.last_message

        # Exact post-check: a pathological store (thousands of candidates) can
        # be over budget on metadata alone.
        if want_uvs and len(blob) > budget:
            reason = (
                "compressed store %s exceeds the %s budget; stored metadata only"
                % (_human_bytes(len(blob)), _human_bytes(budget))
            )
            want_uvs = False
            try:
                blob = self._encode(include_uvs=False, note=reason)
            except Exception as exc:  # noqa: BLE001
                self.last_message = "could not encode candidate store: %s" % exc
                return False, self.last_message

        if len(blob) > budget:
            self.clear_id(obj)
            self.last_message = (
                "candidate metadata alone is %s, over the %s budget; nothing persisted"
                % (_human_bytes(len(blob)), _human_bytes(budget))
            )
            return False, self.last_message

        try:
            # DELETE BEFORE WRITING. Re-assigning an existing bytes ID property
            # silently TRUNCATES the new value at its first NUL byte -- verified
            # identically on Blender 4.5.4 and 5.1.2 (a 314-byte zlib blob came
            # back as 309). Our payload is compressed, so it is full of NULs.
            # Without this, the FIRST Generate on an object persists correctly
            # and every subsequent one quietly shreds the store, which is the
            # worst possible shape for a bug: it passes every single-run test.
            # Deleting the key first round-trips exactly.
            if hasattr(mesh, "pop"):
                mesh.pop(self.ID_KEY, None)
            elif self.ID_KEY in mesh.keys():
                del mesh[self.ID_KEY]
            mesh[self.ID_KEY] = blob
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("could not write ID property", exc_info=True)
            self.last_message = "could not write ID property: %s" % exc
            return False, self.last_message

        if not want_uvs:
            self.last_message = reason
            return False, reason

        self.last_message = "persisted %d candidates, %s" % (
            len(self._items),
            _human_bytes(len(blob)),
        )
        return True, self.last_message

    def load_from_id(self, obj: Any, *, discard_stale: bool = True) -> bool:
        """Load candidates from ``obj.data[ID_KEY]`` into this store.

        **Store invalidation.**  A stored blob records the loop count it was
        built against.  If the mesh has been edited since -- a subdivide, a
        deleted face, anything that changes ``len(mesh.loops)`` -- every stored
        UV array is meaningless, and blitting one into the mesh would either
        raise or produce garbage.  Such a store is discarded and the reason is
        left in :attr:`last_message` for the caller to report.

        Args:
            obj: The object to read from.
            discard_stale: When ``True`` (default) a stale blob is also removed
                from the mesh, so it neither reappears nor keeps bloating the
                file.  Pass ``False`` from read-only contexts such as ``poll()``.

        Returns:
            ``True`` when candidates were loaded, ``False`` otherwise (no blob,
            unreadable blob, or stale blob).
        """
        self.was_stale = False
        mesh = getattr(obj, "data", None)
        if mesh is None:
            self.last_message = "object has no mesh data"
            return False

        try:
            blob = mesh.get(self.ID_KEY) if hasattr(mesh, "get") else mesh[self.ID_KEY]
        except Exception:  # noqa: BLE001
            blob = None
        if blob is None:
            self.last_message = ""
            return False

        try:
            raw = bytes(blob)
        except Exception:  # noqa: BLE001
            self.last_message = "stored candidate data is not a byte blob; ignored"
            return False

        try:
            header, uv_bytes = _decode(raw)
        except Exception as exc:  # noqa: BLE001
            # Debug, not warning: the reason is returned to the caller for the
            # UI to show, and a stack trace in the console for a file the user
            # can do nothing about is noise, not information.
            self.last_message = "candidate store is unreadable (%s); ignored" % exc
            _LOG.debug("candidate store decode failed", exc_info=True)
            return False

        try:
            mesh_loops = len(mesh.loops)
        except Exception:  # noqa: BLE001
            mesh_loops = header.loop_count

        if header.loop_count != mesh_loops:
            # clear() resets last_message, so the explanation is set afterwards.
            self.clear()
            self.was_stale = True
            self.last_message = (
                "stored candidates were built for %d loops but the mesh now has %d; "
                "discarded (re-run Generate)" % (header.loop_count, mesh_loops)
            )
            if discard_stale:
                self.clear_id(obj)
            return False

        self.loop_count = mesh_loops
        if header.obj_name:
            self.obj_name = header.obj_name

        items: list[Candidate] = []
        cursor = 0
        floats = None
        if header.has_uvs and uv_bytes:
            floats = np.frombuffer(uv_bytes, dtype=_UV_DTYPE)

        n_per = mesh_loops * 2
        for entry in header.candidates:
            uvs = None
            if floats is not None and bool(entry.get("has_uvs", False)):
                end = cursor + n_per
                if end <= floats.size:
                    # A copy, not a view: the buffer this slices from is a
                    # temporary decompression result, and a view would pin the
                    # whole multi-megabyte block alive for one candidate.
                    uvs = np.array(floats[cursor:end], dtype=_UV_DTYPE).reshape(-1, 2)
                cursor = end
            items.append(Candidate.from_header_dict(entry, uvs))

        self.set(items)
        note = header.note or ""
        self.last_message = "loaded %d candidates%s" % (
            len(items),
            (" (%s)" % note) if note else "",
        )
        return True

    def clear_id(self, obj: Any) -> None:
        """Remove the persisted blob from ``obj``'s mesh, if present."""
        mesh = getattr(obj, "data", None)
        if mesh is None:
            return
        try:
            # `in` on a Blender datablock tests its ID properties, which is
            # exactly the namespace ID_KEY lives in.
            if self.ID_KEY in mesh:
                del mesh[self.ID_KEY]
        except Exception:  # noqa: BLE001 - deletion is best effort.
            _LOG.debug("could not delete %s", self.ID_KEY, exc_info=True)

    @classmethod
    def stored_loop_count(cls, obj: Any) -> int | None:
        """Return the loop count a stored blob was built for, without decoding UVs.

        Lets a ``poll()`` or a panel cheaply answer "is this store stale?"
        without paying for a multi-megabyte decompression on every redraw.
        """
        mesh = getattr(obj, "data", None)
        if mesh is None:
            return None
        try:
            blob = mesh.get(cls.ID_KEY)
            if blob is None:
                return None
            header, _ = _decode(bytes(blob), header_only=True)
            return int(header.loop_count)
        except Exception:  # noqa: BLE001
            return None

    # -- encoding ------------------------------------------------------------

    def _encode(self, *, include_uvs: bool, note: str = "") -> bytes:
        """Serialise the store into its on-disk byte layout.

        Layout::

            STORE_MAGIC        8 bytes
            version            uint32 little-endian
            zlib stream of:
                header_len     uint32 little-endian
                header         utf-8 JSON
                payload        float32 little-endian, candidates concatenated

        The magic and version sit *outside* the compressed stream on purpose, so
        a mismatched or foreign blob is rejected before any decompression is
        attempted.
        """
        entries: list[dict[str, Any]] = []
        chunks: list[bytes] = []
        for c in self._items:
            entry = c.header_dict()
            has = bool(include_uvs and c.uvs is not None and c.uvs.size)
            entry["has_uvs"] = has
            if has:
                arr = c.uvs
                assert arr is not None  # narrowed by `has`
                chunks.append(np.ascontiguousarray(arr, dtype=_UV_DTYPE).tobytes())
            entries.append(entry)

        header = {
            "version": STORE_VERSION,
            "obj_name": self.obj_name,
            "loop_count": int(self.loop_count),
            "has_uvs": bool(include_uvs and chunks),
            "n_candidates": len(entries),
            "created": time.time(),
            "note": note,
            "candidates": entries,
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = b"".join(chunks)
        inner = struct.pack("<I", len(header_bytes)) + header_bytes + payload
        return STORE_MAGIC + struct.pack("<I", STORE_VERSION) + zlib.compress(inner, _ZLIB_LEVEL)


def _decode(raw: bytes, *, header_only: bool = False) -> tuple[_StoreHeader, bytes]:
    """Decode a persisted blob into its header and raw UV bytes.

    Args:
        raw: The full blob as read from the ID property.
        header_only: When ``True`` the UV payload is returned empty.  The zlib
            stream still has to be decompressed (it is one stream), but the
            multi-megabyte copy into a candidate array is skipped.

    Returns:
        ``(header, uv_bytes)``.

    Raises:
        ValueError: On a bad magic, an unsupported version or truncated data.
            Callers catch this; it never escapes into Blender.
    """
    if len(raw) < len(STORE_MAGIC) + 4:
        raise ValueError("blob too short")
    if raw[: len(STORE_MAGIC)] != STORE_MAGIC:
        raise ValueError("bad magic")
    (version,) = struct.unpack_from("<I", raw, len(STORE_MAGIC))
    if version > STORE_VERSION:
        raise ValueError("store version %d is newer than supported %d" % (version, STORE_VERSION))
    inner = zlib.decompress(raw[len(STORE_MAGIC) + 4 :])
    if len(inner) < 4:
        raise ValueError("truncated store")
    (hlen,) = struct.unpack_from("<I", inner, 0)
    if hlen < 0 or 4 + hlen > len(inner):
        raise ValueError("truncated store header")
    header_data = json.loads(inner[4 : 4 + hlen].decode("utf-8"))
    payload = b"" if header_only else inner[4 + hlen :]
    header = _StoreHeader(
        version=int(header_data.get("version", version)),
        obj_name=str(header_data.get("obj_name", "")),
        loop_count=int(header_data.get("loop_count", 0)),
        has_uvs=bool(header_data.get("has_uvs", False)),
        candidates=list(header_data.get("candidates", [])),
        created=float(header_data.get("created", 0.0) or 0.0),
        note=str(header_data.get("note", "") or ""),
    )
    return header, payload


def _human_bytes(n: int) -> str:
    """Return a compact human-readable byte count, ASCII only."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            if unit == "B":
                return "%d B" % int(value)
            return "%.1f %s" % (value, unit)
        value /= 1024.0
    return "%d B" % int(n)
