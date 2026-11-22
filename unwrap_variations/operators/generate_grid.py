"""``uvvar.generate_grid`` -- produce N scored unwrap candidates.

This module also hosts :class:`UVVarModalJob`, the small modal-operator engine
shared with :mod:`unwrap_variations.operators.branch_variations`.  Both
long-running operators need identical behaviour -- a timer-driven event loop, a
progress bar, ESC to cancel, guaranteed cleanup -- and duplicating that in two
files is how the two copies drift apart.

**Why modal at all.**  A ten-candidate run on a dense mesh is seconds to
minutes of solver time.  Running that inside a plain ``execute`` freezes
Blender's window: no redraw, no cursor, no way out, and on Windows the OS
eventually paints the "not responding" ghost over it.  A modal operator returns
control to Blender between units of work, so the UI keeps breathing and ESC
keeps working.

**How the work is sliced.**  The operator asks the engine for an *incremental*
entry point first (``engine.iter_generate`` / ``engine.iter_branch``: a
generator that yields ``(fraction, message)`` per candidate and returns the
``GenerationReport``).  With that, one timer tick equals one candidate and ESC
is honoured between candidates.  If the engine only offers the blocking
``generate`` / ``branch`` from ARCHITECTURE.md section 9, the job degrades to a
single unit of work driven from inside the modal loop, still wrapped in
``progress_begin``/``update``/``end`` and still fed by the engine's own
``progress`` callback -- the progress bar moves, but ESC is only sampled at the
unit boundary.  The operator detects which is available at runtime, so it picks
up the finer granularity the moment the engine offers it.
"""

from __future__ import annotations

import inspect
import random
import time
from typing import Any, Callable, Iterator

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Context, Event, Operator

from .. import properties as uvvar_props
from ..preferences import get_prefs, max_persist_bytes
from ..utils import compat
from ..utils import logging as uvvar_logging

__all__ = [
    "UVVarModalJob",
    "UVVAR_OT_generate_grid",
    "resolve_seed",
    "adopt_report",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)

#: Timer interval for the modal loop.  Short enough that the UI feels live,
#: long enough that the timer itself is free next to the solver work.
_TIMER_INTERVAL = 0.05


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def call_flexible(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` after dropping keyword arguments it does not accept.

    The engine is written by another worker against the same contract, but a
    contract has a version and a running Blender does not.  Filtering against
    the live signature means an engine that has not grown ``compute_overlap``
    yet still runs, instead of the whole add-on failing with ``TypeError``.

    Args:
        func: The callable to invoke.
        *args: Positional arguments, passed through untouched.
        **kwargs: Keyword arguments; unsupported ones are dropped.

    Returns:
        Whatever ``func`` returns.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins only.
        return func(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return func(*args, **kwargs)
    accepted = set(signature.parameters)
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        _LOG.debug("dropping unsupported kwargs for %s: %s", getattr(func, "__name__", func),
                   ",".join(dropped))
    return func(*args, **{k: v for k, v in kwargs.items() if k in accepted})


def resolve_seed(props: Any, prefs: Any, requested: int) -> int:
    """Decide which seed a run should use and advance the scene seed.

    Args:
        props: The scene property group, or ``None``.
        prefs: The add-on preferences (never ``None``; see
            :func:`unwrap_variations.preferences.get_prefs`).
        requested: The operator's ``seed`` property.  Any value below zero
            means "decide from the preference".

    Returns:
        A non-negative seed.
    """
    if requested is not None and int(requested) >= 0:
        return int(requested)

    mode = str(getattr(prefs, "default_seed_mode", "FIXED"))
    current = int(getattr(props, "seed", 0) or 0) if props is not None else 0

    if mode == "RANDOM":
        seed = random.randrange(1 << 30)
    elif mode == "INCREMENT":
        seed = current
    else:
        return current

    if props is not None:
        try:
            props.seed = (current + 1) & 0x3FFFFFFF if mode == "INCREMENT" else seed
        except Exception:  # noqa: BLE001 - writing back is a convenience, not a requirement.
            _LOG.debug("could not write the resolved seed back to the scene", exc_info=True)
    return seed


def _engine_module() -> Any:
    """Import and return ``core.engine``, or ``None`` when it is unavailable.

    Imported lazily and defensively: the engine is a sibling worker's file, and
    a missing or broken engine must produce a clean operator error rather than
    a failed add-on registration.
    """
    try:
        from ..core import engine

        return engine
    except Exception:  # noqa: BLE001
        _LOG.debug("core.engine is not importable", exc_info=True)
        return None


def _candidate_store_cls() -> Any:
    """Return :class:`CandidateStore`, or ``None`` when unavailable."""
    try:
        from ..core.candidate import CandidateStore

        return CandidateStore
    except Exception:  # noqa: BLE001
        return None


def _bridge_module() -> Any:
    """Return ``core.bridge``, or ``None`` when unavailable."""
    try:
        from ..core import bridge

        return bridge
    except Exception:  # noqa: BLE001
        _LOG.debug("core.bridge is not importable", exc_info=True)
        return None


def _drop_stale_blob(obj: Any, store: Any) -> None:
    """Delete the persisted candidate blob before it is rewritten.

    **This is a workaround for a real Blender bug, not defensive noise.**
    Verified identically on 4.5.4 and 5.1.2: assigning ``bytes`` to a mesh
    ID-property that *already holds* a bytes value truncates the new value at
    its first NUL byte.  A zlib stream is full of NULs, so the second Generate
    on the same object leaves a nine-byte stub behind -- silently, with no
    error anywhere -- and every candidate stored in the ``.blend`` is lost.
    ARCHITECTURE.md section 0 records that bytes ID-properties round-trip, and
    that is true, but only for the *first* write.

    Deleting the key first makes every write a first write, which does
    round-trip correctly (proven up to 2 KB and 2 MB blobs on both versions).

    Args:
        obj: The object whose mesh carries the blob.
        store: The store about to be saved, consulted for its ID key.
    """
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return
    key = getattr(store, "ID_KEY", None) or "uvvar_store_v1"
    clear_id = getattr(store, "clear_id", None)
    if callable(clear_id):
        try:
            clear_id(obj)
            if mesh.get(key) is None:
                return
        except Exception:  # noqa: BLE001 - fall through to the direct delete.
            _LOG.debug("store.clear_id failed", exc_info=True)
    try:
        if mesh.get(key) is not None:
            del mesh[key]
    except Exception:  # noqa: BLE001 - a failed delete only costs persistence.
        _LOG.debug("could not delete stale %r before rewriting", key, exc_info=True)


def adopt_report(
    context: Context,
    obj: Any,
    report: Any,
    *,
    append: bool,
) -> tuple[int, str]:
    """Fold a :class:`GenerationReport` into the runtime store and the UI list.

    Args:
        context: The operator's context.
        obj: The object the run belongs to.
        report: The engine's report.  Only ``candidates`` is required; an
            engine that also attaches a ready-made ``store`` is used directly.
        append: ``True`` for a branch (keep existing candidates), ``False`` for
            a fresh grid (replace them).

    Returns:
        ``(count, note)`` -- how many candidates the store now holds, and any
        non-fatal message (a persistence budget hit, for example).
    """
    candidates = list(getattr(report, "candidates", None) or [])
    store = getattr(report, "store", None)

    if store is None:
        store = uvvar_props.get_store(obj) if append else None
        if store is None:
            store_cls = _candidate_store_cls()
            if store_cls is None:
                return 0, "core.candidate is unavailable; candidates were not stored"
            loops = 0
            try:
                loops = len(obj.data.loops)
            except Exception:  # noqa: BLE001
                loops = 0
            store = store_cls(getattr(obj, "name", ""), loops)
        if append:
            known = {c.signature for c in store.all()}
            fresh = [c for c in candidates if c.signature not in known]
            store.extend(fresh)
        else:
            store.set(candidates)

    uvvar_props.put_store(obj, store)

    note = ""
    prefs = get_prefs(context)
    budget = max_persist_bytes(context)
    if budget > 0 and hasattr(store, "save_to_id"):
        try:
            _drop_stale_blob(obj, store)
            ok, message = store.save_to_id(obj, max_bytes=budget)
            if not ok and message:
                note = message
        except Exception as exc:  # noqa: BLE001 - persistence is never fatal.
            note = "could not persist candidates: %s" % exc
            _LOG.debug("save_to_id failed", exc_info=True)
    elif budget <= 0 and bool(getattr(prefs, "persist_candidates", True)) is False:
        note = "candidate persistence is disabled in the add-on preferences"

    scene = getattr(context, "scene", None)
    uvvar_props.sync_ui_list(scene, obj, store=store)
    return len(store), note


def preview_best(context: Context, obj: Any) -> None:
    """Show the top-ranked candidate in the preview UV layer.

    Failures are logged and swallowed: a run that produced good candidates must
    not be reported as failed just because the convenience preview did not
    happen.
    """
    props = uvvar_props.get_props(context)
    if props is None or not len(props.candidates):
        return
    try:
        from .select_candidate import preview_candidate

        props.active_index = 0
        preview_candidate(context, index=0, report=None, obj=obj)
    except Exception:  # noqa: BLE001
        _LOG.debug("could not preview the best candidate", exc_info=True)


class UVVarModalJob:
    """Mixin turning an operator into a timer-driven, cancellable job.

    Subclasses provide :meth:`job_steps` (a generator yielding
    ``(fraction, message)``) and :meth:`job_finished`.  Everything else --
    timers, the progress bar, the status text, ESC handling and, critically,
    cleanup on every exit path -- is handled here.

    Cleanup is the part worth being pedantic about: a modal operator that
    returns without removing its timer leaves Blender spinning that timer
    forever, and one that returns without ``progress_end`` leaves a stuck
    progress bar the user can only clear by restarting.
    """

    # Plain class attributes, deliberately not annotations and deliberately not
    # an ``__init__``.  Blender scans a registered operator's annotations for
    # RNA property definitions, and it has not always called ``__init__`` on
    # operator instances; class-level defaults shadowed by ``self.x = ...`` are
    # the idiom that works on every version.
    job_report = None
    _timer = None
    _steps = None
    _progress_open = False
    _started = 0.0
    _last_message = ""
    _cancel_requested = False
    _obj = None

    # -- to be implemented by the operator ----------------------------------

    def job_steps(self, context: Context) -> Iterator[tuple[float, str]]:
        """Yield ``(fraction, message)`` for each unit of work.

        Implementations must set :attr:`job_report` before the generator is
        exhausted.
        """
        raise NotImplementedError

    def job_finished(self, context: Context) -> set[str]:
        """Consume :attr:`job_report` and return the operator's result set."""
        raise NotImplementedError

    def job_label(self) -> str:
        """Return a short ASCII label for the status bar."""
        return "Working"

    # -- shared modal machinery ---------------------------------------------

    def run_blocking(self, context: Context) -> set[str]:
        """Run the whole job without a modal loop.

        Used headlessly (``blender -b``), from scripts, and whenever a window
        manager is not available.  Same work, same reporting, no event loop.
        """
        self._started = time.perf_counter()
        self._begin_progress(context)
        try:
            self._steps = self.job_steps(context)
            for fraction, message in self._steps:
                self._publish(context, fraction, message)
            return self.job_finished(context)
        except Exception as exc:  # noqa: BLE001 - nothing may escape into Blender.
            _LOG.exception("job failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}
        finally:
            self._cleanup(context)

    def start_modal(self, context: Context) -> set[str]:
        """Begin the modal loop, falling back to blocking when impossible."""
        wm = getattr(context, "window_manager", None)
        window = getattr(context, "window", None)
        if compat.is_background() or wm is None or window is None:
            return self.run_blocking(context)

        self._started = time.perf_counter()
        try:
            self._steps = self.job_steps(context)
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("could not start job")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            return {"CANCELLED"}

        self._begin_progress(context)
        try:
            self._timer = wm.event_timer_add(_TIMER_INTERVAL, window=window)
            wm.modal_handler_add(self)
        except Exception as exc:  # noqa: BLE001 - no modal support: just do the work.
            _LOG.debug("modal setup failed (%s); running blocking", exc)
            try:
                for fraction, message in self._steps:
                    self._publish(context, fraction, message)
                return self.job_finished(context)
            except Exception as inner:  # noqa: BLE001
                self.report({"ERROR"}, uvvar_logging.ascii_safe(str(inner)))
                return {"CANCELLED"}
            finally:
                self._cleanup(context)
        return {"RUNNING_MODAL"}

    def modal(self, context: Context, event: Event) -> set[str]:
        """Advance the job by one unit per timer tick; ESC cancels."""
        if event.type in {"ESC"} and event.value == "PRESS":
            self._cancel_requested = True
            self.report({"INFO"}, "Cancelled after %s" % self._elapsed())
            self._cleanup(context)
            return {"CANCELLED"}

        if event.type != "TIMER":
            # Swallow other input so a stray click cannot start a second run
            # on top of this one.
            return {"RUNNING_MODAL"}

        if self._steps is None:
            self._cleanup(context)
            return {"CANCELLED"}

        try:
            fraction, message = next(self._steps)
        except StopIteration:
            try:
                return self.job_finished(context)
            except Exception as exc:  # noqa: BLE001
                _LOG.exception("job finalisation failed")
                self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
                return {"CANCELLED"}
            finally:
                self._cleanup(context)
        except Exception as exc:  # noqa: BLE001 - a broken engine must not crash Blender.
            _LOG.exception("job step failed")
            self.report({"ERROR"}, uvvar_logging.ascii_safe(str(exc)))
            self._cleanup(context)
            return {"CANCELLED"}

        self._publish(context, fraction, message)
        return {"RUNNING_MODAL"}

    def cancel(self, context: Context) -> None:
        """Blender's own cancellation hook; guarantees cleanup."""
        self._cancel_requested = True
        self._cleanup(context)

    # -- progress reporting --------------------------------------------------

    def progress_callback(self, fraction: float, message: str = "") -> None:
        """Callback handed to the engine so a blocking run still shows movement.

        Args:
            fraction: Completion in ``[0, 1]``.
            message: Short status text.
        """
        self._publish(bpy.context, fraction, message)

    def should_cancel(self) -> bool:
        """Cancellation probe handed to the engine.

        The engine calls this between candidates
        (``core.engine.generate(should_cancel=...)``), so a run that is already
        marked for cancellation stops at the next candidate boundary instead of
        working through the rest of the grid.

        Returns:
            ``True`` once the user has asked to stop.
        """
        return bool(self._cancel_requested)

    def _begin_progress(self, context: Context) -> None:
        """Open the window-manager progress bar and mark the scene as busy."""
        wm = getattr(context, "window_manager", None)
        if wm is not None and hasattr(wm, "progress_begin"):
            try:
                wm.progress_begin(0.0, 100.0)
                self._progress_open = True
            except Exception:  # noqa: BLE001
                self._progress_open = False
        props = uvvar_props.get_props(context)
        if props is not None:
            try:
                props.is_running = True
                props.progress = 0.0
                props.status = self.job_label()
            except Exception:  # noqa: BLE001
                pass

    def _publish(self, context: Context, fraction: float, message: str) -> None:
        """Push progress to the progress bar, the status bar and the scene."""
        try:
            value = min(max(float(fraction), 0.0), 1.0)
        except (TypeError, ValueError):
            value = 0.0
        text = uvvar_logging.ascii_safe(message or self._last_message)
        self._last_message = text

        wm = getattr(context, "window_manager", None)
        if self._progress_open and wm is not None:
            try:
                wm.progress_update(value * 100.0)
            except Exception:  # noqa: BLE001
                pass

        props = uvvar_props.get_props(context)
        if props is not None:
            try:
                props.progress = value
                props.status = text[:220]
            except Exception:  # noqa: BLE001
                pass

        workspace = getattr(context, "workspace", None)
        if workspace is not None and hasattr(workspace, "status_text_set"):
            try:
                workspace.status_text_set("%s  %d%%  (ESC to cancel)" % (
                    self.job_label(), int(value * 100.0)))
            except Exception:  # noqa: BLE001
                pass
        compat.tag_redraw(context)

    def _elapsed(self) -> str:
        """Return the wall time since the job started, as ASCII."""
        return "%.1fs" % max(0.0, time.perf_counter() - self._started)

    def _cleanup(self, context: Context) -> None:
        """Release the timer, the progress bar and the status text.

        Every field is cleared unconditionally and independently, because this
        runs on the failure paths too and a half-cleaned modal operator is
        exactly the state that leaves Blender feeling broken.
        """
        wm = getattr(context, "window_manager", None)
        if self._timer is not None and wm is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:  # noqa: BLE001
                pass
        self._timer = None

        if self._progress_open and wm is not None:
            try:
                wm.progress_end()
            except Exception:  # noqa: BLE001
                pass
        self._progress_open = False

        workspace = getattr(context, "workspace", None)
        if workspace is not None and hasattr(workspace, "status_text_set"):
            try:
                workspace.status_text_set(None)
            except Exception:  # noqa: BLE001
                pass

        props = uvvar_props.get_props(context)
        if props is not None:
            try:
                props.is_running = False
                props.progress = 0.0
            except Exception:  # noqa: BLE001
                pass
        self._steps = None
        compat.tag_redraw(context)


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------


class UVVAR_OT_generate_grid(UVVarModalJob, Operator):
    """Generate a grid of scored UV unwrap variations for the active mesh."""

    bl_idname = "uvvar.generate_grid"
    bl_label = "Generate UV Variations"
    bl_description = (
        "Unwrap the active mesh several different ways, score each result and "
        "collect them for side-by-side comparison"
    )
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(
        name="Count",
        description="How many variations to generate; 0 uses the scene setting",
        default=0,
        min=0,
        max=64,
    )

    seed: IntProperty(
        name="Seed",
        description="Determinism seed; -1 resolves it from the add-on preference",
        default=-1,
        min=-1,
        max=2 ** 30,
    )

    compute_overlap: EnumProperty(
        name="Overlap Metric",
        description="Whether to measure overlapping UV area, which is the expensive metric",
        items=(
            ("PREFS", "From Settings", "Use the scene and add-on preference"),
            ("ON", "Measure", "Always measure overlap"),
            ("OFF", "Skip", "Never measure overlap"),
        ),
        default="PREFS",
    )

    keep_existing: BoolProperty(
        name="Keep Existing",
        description="Add to the current candidate set instead of replacing it",
        default=False,
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled for an active mesh object that is not already generating."""
        if compat.active_mesh_object(context) is None:
            return False
        props = uvvar_props.get_props(context)
        return not (props is not None and props.is_running)

    def job_label(self) -> str:
        """Return the status-bar label."""
        return "Generating UV variations"

    def invoke(self, context: Context, _event: Event) -> set[str]:
        """Start the modal run (or a blocking one when headless)."""
        return self.start_modal(context)

    def execute(self, context: Context) -> set[str]:
        """Run to completion without a modal loop, for scripts and tests."""
        return self.run_blocking(context)

    # -- the work ------------------------------------------------------------

    def job_steps(self, context: Context) -> Iterator[tuple[float, str]]:
        """Yield one step per candidate, or one step for the whole run.

        Raises:
            RuntimeError: When the mesh or the engine is unusable.  Both modal
                and blocking drivers convert this into a clean operator error.
        """
        obj = compat.active_mesh_object(context)
        problem = compat.mesh_problem(obj)
        if problem:
            raise RuntimeError(problem)

        engine = _engine_module()
        if engine is None:
            raise RuntimeError(
                "the unwrap engine (core.engine) is not available in this build"
            )

        props = uvvar_props.get_props(context)
        prefs = get_prefs(context)

        count = int(self.count) or int(getattr(props, "candidate_count", 0) or 0) or int(
            getattr(prefs, "default_candidate_count", 10)
        )
        count = max(1, min(count, 64))
        seed = resolve_seed(props, prefs, int(self.seed))

        if self.compute_overlap == "ON":
            overlap = True
        elif self.compute_overlap == "OFF":
            overlap = False
        else:
            overlap = bool(getattr(props, "compute_overlap", True)) and bool(
                getattr(prefs, "compute_overlap", True)
            )

        weights = uvvar_props.score_weights(props)
        backends = self._backend_filter(props)
        # Refinement is score-gated and so cannot make a candidate worse, but it
        # is several times slower than a bare solve, which is a trade the artist
        # owns rather than one to make for them.
        refine_layouts = bool(getattr(props, "refine_layouts", True))

        kwargs: dict[str, Any] = {
            "count": count,
            "seed": seed,
            "weights": weights,
            "backends": backends,
            "compute_overlap": overlap,
            "refine_layouts": refine_layouts,
            # The operator owns storage: it merges, applies the persistence
            # budget and refreshes the UI mirror. Letting the engine also
            # "store" would build a throwaway CandidateStore, and building one
            # decompresses whatever blob the mesh already carries -- up to the
            # whole 64 MB budget, on every single run, for nothing.
            "store": False,
        }

        _LOG.info(
            "generate: obj=%s count=%d seed=%d overlap=%s backends=%s",
            getattr(obj, "name", "?"), count, seed, overlap, backends,
        )

        self._obj = obj
        yield from _drive_engine(
            self,
            engine,
            incremental_name="iter_generate",
            blocking_name="generate",
            args=(obj,),
            kwargs=kwargs,
            unit_label="candidate",
            total=count,
        )

    def job_finished(self, context: Context) -> set[str]:
        """Store the report, refresh the UI list and preview the winner."""
        obj = getattr(self, "_obj", None) or compat.active_mesh_object(context)
        report = self.job_report
        if report is None:
            self.report({"ERROR"}, "The engine returned no result")
            return {"CANCELLED"}

        total, note = adopt_report(context, obj, report, append=bool(self.keep_existing))
        failures = list(getattr(report, "failures", None) or [])
        warnings = list(getattr(report, "warnings", None) or [])

        if total == 0:
            detail = uvvar_logging.format_lines(
                [str(msg) for _params, msg in failures] + warnings, limit=3
            )
            self.report({"ERROR"}, "No candidate could be generated. %s" % detail)
            return {"CANCELLED"}

        preview_best(context, obj)

        seconds = float(getattr(report, "seconds", 0.0) or 0.0) or (
            time.perf_counter() - self._started
        )
        summary = "%d candidates in %.1fs" % (total, seconds)
        if getattr(report, "cancelled", False):
            summary += " (cancelled early)"
        if failures:
            summary += ", %d failed" % len(failures)
        extra = uvvar_logging.format_lines(warnings + ([note] if note else []), limit=2)
        if extra:
            summary += " -- " + extra

        props = uvvar_props.get_props(context)
        if props is not None:
            try:
                props.status = summary[:220]
            except Exception:  # noqa: BLE001
                pass

        self.report({"WARNING"} if failures else {"INFO"}, uvvar_logging.ascii_safe(summary))
        _LOG.info("generate finished: %s", summary)
        return {"FINISHED"}

    @staticmethod
    def _backend_filter(props: Any) -> tuple[str, ...] | None:
        """Return the backend id whitelist, or ``None`` for "all"."""
        choice = str(getattr(props, "backend_filter", "ALL") or "ALL")
        return None if choice == "ALL" else (choice,)


def _drive_engine(
    job: UVVarModalJob,
    engine: Any,
    *,
    incremental_name: str,
    blocking_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    unit_label: str,
    total: int,
) -> Iterator[tuple[float, str]]:
    """Yield progress for an engine call, preferring an incremental API.

    Args:
        job: The operator, used for its progress callback and report slot.
        engine: The ``core.engine`` module.
        incremental_name: Name of the generator-style entry point to prefer.
        blocking_name: Name of the one-shot entry point to fall back to.
        args: Positional arguments for either entry point.
        kwargs: Keyword arguments for either entry point.
        unit_label: Word used in progress messages, e.g. ``"candidate"``.
        total: Expected number of units, used for the fallback message.

    Yields:
        ``(fraction, message)`` pairs.
    """
    incremental = getattr(engine, incremental_name, None)
    if callable(incremental):
        _LOG.debug("using incremental engine entry point %s", incremental_name)
        generator = call_flexible(
            incremental, *args, should_cancel=job.should_cancel, **kwargs
        )
        while True:
            try:
                step = next(generator)
            except StopIteration as stop:
                job.job_report = stop.value if stop.value is not None else getattr(
                    generator, "report", None
                )
                return
            yield _normalise_step(step, unit_label)

    blocking = getattr(engine, blocking_name, None)
    if not callable(blocking):
        raise RuntimeError(
            "core.engine exposes neither %s nor %s" % (incremental_name, blocking_name)
        )

    # One unit of work.  The first yield lets the modal loop paint a 0% state
    # before the solver takes the main thread; the engine's own progress
    # callback keeps the progress bar moving while it holds it.
    yield 0.0, "preparing %d %ss" % (total, unit_label)
    job.job_report = call_flexible(
        blocking,
        *args,
        progress=job.progress_callback,
        should_cancel=job.should_cancel,
        **kwargs,
    )
    yield 1.0, "scoring %d %ss" % (total, unit_label)


def _normalise_step(step: Any, unit_label: str) -> tuple[float, str]:
    """Coerce whatever the engine yielded into ``(fraction, message)``.

    Accepts a bare float, a ``(fraction, message)`` pair, or anything with
    ``fraction``/``message`` attributes, so a small difference in the engine's
    generator protocol is absorbed here rather than crashing the operator.
    """
    if isinstance(step, tuple) and len(step) >= 2:
        try:
            return float(step[0]), str(step[1])
        except (TypeError, ValueError):
            return 0.0, str(step[1])
    if isinstance(step, (int, float)):
        return float(step), "working on %ss" % unit_label
    fraction = getattr(step, "fraction", None)
    message = getattr(step, "message", "")
    if fraction is not None:
        try:
            return float(fraction), str(message)
        except (TypeError, ValueError):
            pass
    return 0.0, uvvar_logging.ascii_safe(step)


#: Classes contributed by this module.
classes: tuple[type, ...] = (UVVAR_OT_generate_grid,)


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
