"""``uvvar.branch_variations`` -- explore around one candidate.

Where :mod:`unwrap_variations.operators.generate_grid` spreads candidates as
widely as possible, branching does the opposite: it takes the one the user
liked and produces near-siblings of it, so the workflow is "give me ten ideas,
now give me ten more like that one".

The operator shares :class:`~unwrap_variations.operators.generate_grid.UVVarModalJob`
with Generate, so it is modal, shows the same progress bar and cancels with ESC
for the same reasons -- a branch run costs exactly as much solver time as the
grid it came from.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty
from bpy.types import Context, Event, Operator

from .. import properties as uvvar_props
from ..preferences import get_prefs
from ..utils import compat
from ..utils import logging as uvvar_logging
from .generate_grid import (
    UVVarModalJob,
    _drive_engine,
    _engine_module,
    adopt_report,
    preview_best,
    resolve_seed,
)

__all__ = [
    "UVVAR_OT_branch_variations",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)


class UVVAR_OT_branch_variations(UVVarModalJob, Operator):
    """Generate variations around the selected candidate."""

    bl_idname = "uvvar.branch_variations"
    bl_label = "Branch From Candidate"
    bl_description = (
        "Produce new variations clustered around the selected candidate, "
        "keeping what worked and perturbing the rest"
    )
    bl_options = {"REGISTER", "UNDO"}

    count: IntProperty(
        name="Count",
        description="How many children to produce; 0 uses the scene setting",
        default=0,
        min=0,
        max=64,
    )

    strength: FloatProperty(
        name="Strength",
        description="How far a child strays from its parent; -1 uses the scene setting",
        default=-1.0,
        min=-1.0,
        max=1.0,
    )

    seed: IntProperty(
        name="Seed",
        description="Determinism seed; -1 resolves it from the add-on preference",
        default=-1,
        min=-1,
        max=2 ** 30,
    )

    source_index: IntProperty(
        name="Parent",
        description="Candidate store index to branch from; -1 uses the active candidate",
        default=-1,
        min=-1,
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

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Enabled for an active mesh object that already has candidates."""
        obj = compat.active_mesh_object(context)
        if obj is None:
            return False
        props = uvvar_props.get_props(context)
        if props is not None and props.is_running:
            return False
        store = uvvar_props.get_store(obj)
        if store is not None and len(store):
            return True
        return bool(props is not None and len(props.candidates))

    def job_label(self) -> str:
        """Return the status-bar label."""
        return "Branching UV variations"

    def invoke(self, context: Context, _event: Event) -> set[str]:
        """Start the modal run (or a blocking one when headless)."""
        return self.start_modal(context)

    def execute(self, context: Context) -> set[str]:
        """Run to completion without a modal loop, for scripts and tests."""
        return self.run_blocking(context)

    # -- the work ------------------------------------------------------------

    def job_steps(self, context: Context) -> Iterator[tuple[float, str]]:
        """Yield one step per child, or one step for the whole run.

        Raises:
            RuntimeError: When the mesh, the store, the parent candidate or the
                engine is unusable.
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

        store = uvvar_props.store_for(obj, create=False)
        if store is None or not len(store):
            raise RuntimeError("no candidates to branch from; run Generate first")

        props = uvvar_props.get_props(context)
        prefs = get_prefs(context)

        parent = self._resolve_parent(props, store)
        if parent is None:
            raise RuntimeError("the selected candidate could not be found in the store")
        if getattr(parent, "uvs", None) is None and not getattr(parent, "ok", True):
            raise RuntimeError(
                "candidate %d failed to unwrap, so there is nothing to branch from"
                % int(getattr(parent, "index", -1))
            )

        count = int(self.count) or int(getattr(props, "candidate_count", 0) or 0) or int(
            getattr(prefs, "default_candidate_count", 10)
        )
        count = max(1, min(count, 64))

        strength = float(self.strength)
        if strength < 0.0:
            strength = float(
                getattr(props, "mutation_strength", None)
                if props is not None
                else getattr(prefs, "default_mutation_strength", 0.35)
            )
        strength = min(max(strength, 0.0), 1.0)

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

        kwargs: dict[str, Any] = {
            "count": count,
            "strength": strength,
            "seed": seed,
            "weights": weights,
            "compute_overlap": overlap,
        }

        _LOG.info(
            "branch: obj=%s parent=%s count=%d strength=%.2f seed=%d",
            getattr(obj, "name", "?"), getattr(parent, "signature", "?"), count, strength, seed,
        )

        self._obj = obj
        self._parent = parent
        yield from _drive_engine(
            self,
            engine,
            incremental_name="iter_branch",
            blocking_name="branch",
            args=(obj, parent),
            kwargs=kwargs,
            unit_label="variation",
            total=count,
        )

    def job_finished(self, context: Context) -> set[str]:
        """Merge the children into the store and refresh the UI list."""
        obj = getattr(self, "_obj", None) or compat.active_mesh_object(context)
        report = self.job_report
        if report is None:
            self.report({"ERROR"}, "The engine returned no result")
            return {"CANCELLED"}

        before = 0
        store = uvvar_props.get_store(obj)
        if store is not None:
            before = len(store)

        total, note = adopt_report(context, obj, report, append=True)
        added = max(0, total - before)
        failures = list(getattr(report, "failures", None) or [])
        warnings = list(getattr(report, "warnings", None) or [])

        if added == 0:
            detail = uvvar_logging.format_lines(
                [str(msg) for _params, msg in failures] + warnings, limit=3
            )
            self.report(
                {"WARNING"},
                uvvar_logging.ascii_safe(
                    "Branching produced no new variations. %s" % detail
                ),
            )
            return {"CANCELLED"}

        preview_best(context, obj)

        seconds = float(getattr(report, "seconds", 0.0) or 0.0) or (
            time.perf_counter() - self._started
        )
        summary = "%d new variations in %.1fs (%d total)" % (added, seconds, total)
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
        _LOG.info("branch finished: %s", summary)
        return {"FINISHED"}

    # -- helpers -------------------------------------------------------------

    def _resolve_parent(self, props: Any, store: Any) -> Any:
        """Return the candidate to branch from.

        Resolution order: an explicit ``source_index`` (a *store* index, so a
        script can address a candidate without knowing the display order), then
        the active row of the UI mirror, then the first candidate.

        Args:
            props: The scene property group, or ``None``.
            store: The candidate store.

        Returns:
            A :class:`Candidate`, or ``None``.
        """
        if int(self.source_index) >= 0:
            return store.get(int(self.source_index))

        if props is not None and len(props.candidates):
            row_index = max(0, min(int(props.active_index), len(props.candidates) - 1))
            try:
                return store.get(int(props.candidates[row_index].index))
            except Exception:  # noqa: BLE001
                _LOG.debug("could not resolve the active row", exc_info=True)
        return store.get(0)


#: Classes contributed by this module.
classes: tuple[type, ...] = (UVVAR_OT_branch_variations,)


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
