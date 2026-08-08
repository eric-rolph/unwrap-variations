"""Scene properties, the UI mirror list, and the runtime candidate cache.

Three things live here, and they are deliberately together because they are
three views of the same state:

1. :class:`UVVAR_SceneProps` -- the knobs the user turns, hung off
   ``bpy.types.Scene.uvvar``.
2. :class:`UVVAR_CandidateItem` -- a **mirror** of one candidate as plain RNA
   floats and strings.  ARCHITECTURE.md section 11 is explicit that the UI reads
   only this collection and never the numpy store; a ``UIList`` redraws on every
   mouse move, and touching a multi-megabyte numpy array from a draw callback
   would be indefensible.
3. The **runtime store cache** -- a module-level ``{object_name: CandidateStore}``
   dict.  Candidates are numpy arrays that cannot live in RNA, and re-decoding
   them from the mesh ID-property on every preview would turn an O(L) memcpy
   into a multi-second zlib decompression.  The cache is what makes
   ``uvvar.select_candidate`` instant.

The cache is keyed by object *name* rather than by the object itself because
Blender reallocates ``Object`` structs behind Python's back (an undo push
invalidates old pointers), which makes a real reference unsafe to hold.
"""

from __future__ import annotations

from typing import Any, Iterable

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Context, PropertyGroup, Scene

from .utils import logging as uvvar_logging

__all__ = [
    "UVVAR_CandidateItem",
    "UVVAR_SceneProps",
    "WEIGHT_PRESETS",
    "OVERLAY_MODES",
    "get_props",
    "get_store",
    "put_store",
    "drop_store",
    "drop_all_stores",
    "store_for",
    "sync_ui_list",
    "score_weights",
    "classes",
    "register",
    "unregister",
]

_LOG = uvvar_logging.get_logger(__name__)


# ``core.candidate`` and ``core.metrics`` are written by other workers in
# parallel.  Importing them defensively means this module -- and therefore
# add-on registration itself -- stays working while they are still landing.
try:  # pragma: no cover - import shim.
    from .core.candidate import CandidateStore

    HAVE_STORE = True
except Exception:  # noqa: BLE001
    CandidateStore = None  # type: ignore[assignment]
    HAVE_STORE = False

try:  # pragma: no cover - import shim.
    from .core.metrics import ScoreWeights

    HAVE_METRICS = True
except Exception:  # noqa: BLE001
    ScoreWeights = None  # type: ignore[assignment]
    HAVE_METRICS = False


#: Scoring presets offered in the UI.  Mirrors ``ScoreWeights.preset`` names
#: from ARCHITECTURE.md section 4.
WEIGHT_PRESETS = (
    ("balanced", "Balanced", "Even weighting across distortion, seams and packing"),
    ("low_distortion", "Low Distortion", "Prefer faithful angles and areas over tight packing"),
    ("few_seams", "Few Seams", "Prefer fewer, shorter cuts even at some distortion cost"),
    ("tight_packing", "Tight Packing", "Prefer efficient use of the UV square"),
    ("game_ready", "Game Ready",
     "Punish overlaps and flipped faces hard; favour even texel density"),
)

#: Viewport overlay modes.  Every identifier other than ``NONE`` must exist in
#: ``ui.stretch_overlay.OVERLAY_MODES``: the drawing module falls back to
#: ``STRETCH`` for anything it does not recognise, so an extra entry here shows
#: the user a mode that quietly does something else.
OVERLAY_MODES = (
    ("NONE", "Off", "No overlay"),
    ("STRETCH", "Stretch", "Colour faces by UV stretch, from low to high distortion"),
    ("ANGLE", "Angle", "Colour faces by angular shear, from square to badly sheared"),
    ("CHECKER", "Checker", "Project a checker pattern to reveal texel density and flips"),
)

#: Backing store for the dynamic backend enum.  Blender does not copy the
#: strings an ``items`` callback returns, so a list built inside the callback is
#: freed the moment the callback exits and the UI shows garbage.  Keeping the
#: tuples alive in a module global is the standard fix.
_BACKEND_ITEMS: list[tuple[str, str, str]] = [("ALL", "All Available", "Use every usable backend")]

#: ``{object_name: CandidateStore}``.  See the module docstring for why this is
#: keyed by name and why it exists at all.
_STORES: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Runtime candidate cache
# ---------------------------------------------------------------------------


def get_store(obj: Any) -> Any | None:
    """Return the cached candidate store for ``obj``, without creating one.

    Args:
        obj: A Blender object, or ``None``.

    Returns:
        The :class:`~unwrap_variations.core.candidate.CandidateStore`, or
        ``None`` when nothing is cached (or when ``core.candidate`` is not
        importable).
    """
    name = getattr(obj, "name", None)
    if not name:
        return None
    return _STORES.get(name)


def put_store(obj: Any, store: Any) -> None:
    """Cache ``store`` as the candidate set for ``obj``."""
    name = getattr(obj, "name", None)
    if name:
        _STORES[name] = store


def drop_store(obj: Any) -> None:
    """Forget the cached store for ``obj``, if any."""
    name = getattr(obj, "name", None)
    if name:
        _STORES.pop(name, None)


def drop_all_stores() -> None:
    """Forget every cached store.

    Called on file load and on unregister: a store holds numpy arrays sized to
    a specific mesh, and both events invalidate that assumption wholesale.
    """
    _STORES.clear()


def store_for(obj: Any, *, create: bool = True) -> Any | None:
    """Return the store for ``obj``, loading or creating it as needed.

    Resolution order: the runtime cache, then the persisted mesh ID-property
    (via ``CandidateStore.for_object``), then -- when ``create`` is set -- a
    fresh empty store.

    Args:
        obj: The object to fetch candidates for.
        create: When ``False``, do not build an empty store; return ``None``
            instead.  Used from ``poll`` and from draw code.

    Returns:
        A store, or ``None``.
    """
    if obj is None or not HAVE_STORE or CandidateStore is None:
        return None
    cached = get_store(obj)
    if cached is not None:
        return cached
    try:
        store = CandidateStore.for_object(obj)
    except Exception:  # noqa: BLE001 - a corrupt blob must not break the UI.
        _LOG.debug("CandidateStore.for_object failed", exc_info=True)
        store = None
    if store is None:
        if not create:
            return None
        loops = 0
        try:
            loops = len(obj.data.loops)
        except Exception:  # noqa: BLE001
            loops = 0
        try:
            store = CandidateStore(getattr(obj, "name", ""), loops)
        except Exception:  # noqa: BLE001
            return None
    if not create and len(store) == 0:
        return store if getattr(store, "last_message", "") else None
    put_store(obj, store)
    return store


def score_weights(props: "UVVAR_SceneProps | None") -> Any:
    """Return the :class:`ScoreWeights` matching the scene's weight preset.

    Falls back to whatever ``ScoreWeights()`` defaults to when the preset name
    is unknown, and to ``None`` when ``core.metrics`` is not importable -- in
    which case callers skip re-ranking rather than fail.
    """
    if not HAVE_METRICS or ScoreWeights is None:
        return None
    name = getattr(props, "weight_preset", "balanced") if props is not None else "balanced"
    try:
        return ScoreWeights.preset(name)
    except Exception:  # noqa: BLE001
        try:
            return ScoreWeights()
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Property update callbacks
# ---------------------------------------------------------------------------


def _on_active_index_changed(self: "UVVAR_SceneProps", context: Context) -> None:
    """Preview the newly selected candidate when auto-preview is on.

    The preview helper is called directly rather than through ``bpy.ops``.
    Invoking an operator from a property update callback is the classic way to
    hit "Writing to ID classes in this context is not allowed"; the helper is a
    plain function that does one ``foreach_set``, which is safe from anywhere
    the property itself can be edited.
    """
    if not bool(getattr(self, "auto_preview", True)):
        return
    try:
        from .operators.select_candidate import preview_candidate

        preview_candidate(context, index=int(self.active_index), report=None)
    except Exception:  # noqa: BLE001 - a UI click must never raise.
        _LOG.debug("auto-preview failed", exc_info=True)


def _on_weights_changed(self: "UVVAR_SceneProps", context: Context) -> None:
    """Re-rank the mirror list when the scoring preset changes."""
    try:
        obj = getattr(context, "object", None)
        sync_ui_list(context.scene if context else None, obj)
    except Exception:  # noqa: BLE001
        _LOG.debug("re-rank after weight change failed", exc_info=True)


def _on_overlay_mode_changed(self: "UVVAR_SceneProps", context: Context) -> None:
    """Push the overlay mode into the drawing module, if one is present."""
    try:
        from .operators.overlay_ops import sync_overlay_state

        sync_overlay_state(context)
    except Exception:  # noqa: BLE001
        _LOG.debug("overlay sync failed", exc_info=True)


def _backend_items(self: "UVVAR_SceneProps", context: Context | None) -> list[tuple[str, str, str]]:
    """Return the backend filter's enum items, rebuilt from the registry.

    Args:
        self: The owning property group (unused).
        context: The current context (unused).

    Returns:
        ``[(identifier, label, description), ...]``, always containing ``ALL``
        so the enum can never end up empty (an empty enum is a Blender error,
        not merely an empty menu).
    """
    items: list[tuple[str, str, str]] = [("ALL", "All Available", "Use every usable backend")]
    try:
        from .backends.base import available_backends

        for backend_id, label, ok, why in available_backends():
            text = label if ok else "%s (unavailable)" % label
            desc = why or ("%s backend" % label)
            items.append((backend_id, text, desc))
    except Exception:  # noqa: BLE001 - the enum must survive a broken backend module.
        pass
    _BACKEND_ITEMS[:] = items
    return _BACKEND_ITEMS


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------


class UVVAR_CandidateItem(PropertyGroup):
    """One row of the candidate list: a flat, RNA-native mirror of a candidate.

    Every field is a plain float, int, bool or string so that drawing a list of
    twenty-four of these costs nothing.  The authoritative data -- the UV array
    -- stays in the numpy store and is never touched from draw code.
    """

    index: IntProperty(name="Index", default=0, min=0)
    label: StringProperty(name="Label", default="")
    signature: StringProperty(name="Signature", default="")
    description: StringProperty(name="Recipe", default="")
    backend: StringProperty(name="Backend", default="")
    message: StringProperty(name="Message", default="")

    score: FloatProperty(name="Score", default=0.0)
    rank: IntProperty(name="Rank", default=0, min=0)
    ok: BoolProperty(name="Succeeded", default=True)
    generation: IntProperty(name="Generation", default=0, min=0)
    parent: StringProperty(name="Parent", default="")
    seconds: FloatProperty(name="Seconds", default=0.0)
    has_metrics: BoolProperty(name="Has Metrics", default=False)
    #: Non-dominated over (stretch, seam ratio, packing).  The panel highlights
    #: these, so the name matches what ``ui/viewport_panel.py`` reads.
    is_pareto: BoolProperty(name="Pareto Optimal", default=False)

    # -- mirrored metrics (see ARCHITECTURE.md section 4) --------------------

    angle_distortion: FloatProperty(name="Angle Distortion", default=0.0, subtype="NONE")
    angle_distortion_max: FloatProperty(name="Angle Distortion Max", default=0.0)
    area_distortion: FloatProperty(name="Area Distortion", default=0.0)
    stretch_l2: FloatProperty(name="Stretch L2", default=0.0)
    stretch_linf: FloatProperty(name="Stretch Linf", default=0.0)
    conformal_error: FloatProperty(name="Conformal Error", default=0.0)
    island_count: IntProperty(name="Islands", default=0)
    seam_length: FloatProperty(name="Seam Length", default=0.0)
    seam_ratio: FloatProperty(name="Seam Ratio", default=0.0)
    packing_efficiency: FloatProperty(name="Packing Efficiency", default=0.0)
    uv_coverage: FloatProperty(name="UV Coverage", default=0.0)
    overlap_area: FloatProperty(name="Overlap Area", default=0.0)
    flipped_tris: IntProperty(name="Flipped Triangles", default=0)
    degenerate_tris: IntProperty(name="Degenerate Triangles", default=0)
    texel_density_cv: FloatProperty(name="Texel Density CV", default=0.0)
    out_of_bounds_tris: IntProperty(name="Out Of Bounds Triangles", default=0)
    n_tris: IntProperty(name="Triangles", default=0)
    eval_seconds: FloatProperty(name="Evaluation Seconds", default=0.0)

    def fill_from(self, candidate: Any, *, rank: int = 0) -> None:
        """Copy one :class:`Candidate` into this row.

        Never raises: a candidate whose metrics failed to load still produces a
        usable row, marked with ``has_metrics = False``.

        Args:
            candidate: The source candidate.
            rank: Position in the current ranking, 1-based; 0 means unranked.
        """
        self.index = int(getattr(candidate, "index", 0))
        self.score = float(getattr(candidate, "score", 0.0) or 0.0)
        self.rank = int(rank)
        self.ok = bool(getattr(candidate, "ok", True))
        self.generation = int(getattr(candidate, "generation", 0) or 0)
        self.parent = str(getattr(candidate, "parent", "") or "")
        self.seconds = float(getattr(candidate, "seconds", 0.0) or 0.0)
        self.message = str(getattr(candidate, "message", "") or "")[:220]

        params = getattr(candidate, "params", None)
        try:
            self.description = params.describe() if params is not None else ""
        except Exception:  # noqa: BLE001
            self.description = ""
        self.backend = str(getattr(params, "backend", "") or "")
        try:
            self.signature = str(candidate.signature)
        except Exception:  # noqa: BLE001
            self.signature = ""
        try:
            self.label = candidate.label()
        except Exception:  # noqa: BLE001
            self.label = "%02d  %s" % (self.index, self.description)

        metrics = getattr(candidate, "metrics", None)
        self.has_metrics = metrics is not None
        if metrics is None:
            return
        for name in _METRIC_FIELDS:
            value = getattr(metrics, name, None)
            if value is None:
                continue
            try:
                setattr(self, name, value)
            except Exception:  # noqa: BLE001 - a metric out of RNA range is not fatal.
                _LOG.debug("could not mirror metric %s=%r", name, value)


#: Metric attribute names mirrored onto :class:`UVVAR_CandidateItem`.
_METRIC_FIELDS: tuple[str, ...] = (
    "angle_distortion",
    "angle_distortion_max",
    "area_distortion",
    "stretch_l2",
    "stretch_linf",
    "conformal_error",
    "island_count",
    "seam_length",
    "seam_ratio",
    "packing_efficiency",
    "uv_coverage",
    "overlap_area",
    "flipped_tris",
    "degenerate_tris",
    "texel_density_cv",
    "out_of_bounds_tris",
    "n_tris",
    "eval_seconds",
)


class UVVAR_SceneProps(PropertyGroup):
    """Per-scene settings and the mirrored candidate list."""

    # -- generation ----------------------------------------------------------

    candidate_count: IntProperty(
        name="Count",
        description="How many variations to generate",
        default=10,
        min=1,
        soft_max=24,
        max=64,
    )

    seed: IntProperty(
        name="Seed",
        description="Determinism seed; the same seed always produces the same grid",
        default=0,
        min=0,
        max=2 ** 30,
    )

    mutation_strength: FloatProperty(
        name="Branch Strength",
        description="How far branched variations stray from their parent",
        default=0.35,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )

    weight_preset: EnumProperty(
        name="Priority",
        description="What 'best' means when candidates are ranked",
        items=WEIGHT_PRESETS,
        default="balanced",
        update=_on_weights_changed,
    )

    backend_filter: EnumProperty(
        name="Backend",
        description="Restrict generation to one backend",
        items=_backend_items,
    )

    compute_overlap: BoolProperty(
        name="Measure Overlap",
        description=(
            "Measure overlapping UV area. The most expensive metric; turn it "
            "off for a much faster generate on dense meshes"
        ),
        default=True,
    )

    # -- browsing ------------------------------------------------------------

    active_index: IntProperty(
        name="Active Candidate",
        description="Which candidate is shown in the preview UV layer",
        default=0,
        min=0,
        update=_on_active_index_changed,
    )

    auto_preview: BoolProperty(
        name="Auto Preview",
        description="Blit a candidate into the preview UV layer as soon as it is selected",
        default=True,
    )

    sort_by_score: BoolProperty(
        name="Sort By Score",
        description="Order the list best-first instead of by generation order",
        default=True,
        update=_on_weights_changed,
    )

    show_failed: BoolProperty(
        name="Show Failed",
        description="Keep candidates whose backend failed visible in the list",
        default=True,
        update=_on_weights_changed,
    )

    # -- overlay -------------------------------------------------------------

    overlay_mode: EnumProperty(
        name="Overlay",
        description="Viewport overlay drawn over the previewed candidate",
        items=OVERLAY_MODES,
        default="NONE",
        update=_on_overlay_mode_changed,
    )

    overlay_opacity: FloatProperty(
        name="Overlay Opacity",
        description="Blend strength of the viewport overlay",
        default=0.65,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_overlay_mode_changed,
    )

    grid_columns: IntProperty(
        name="Grid Columns",
        description="Columns in the candidate grid; 0 fits them to the panel width",
        default=0,
        min=0,
        soft_max=6,
        max=12,
    )

    # -- transient status ----------------------------------------------------

    is_running: BoolProperty(
        name="Working",
        description="True while a generate or branch run is in progress",
        default=False,
        options={"SKIP_SAVE"},
    )

    progress: FloatProperty(
        name="Progress",
        description="Fraction of the current run completed",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        options={"SKIP_SAVE"},
    )

    status: StringProperty(
        name="Status",
        description="Last message from the add-on",
        default="",
        options={"SKIP_SAVE"},
    )

    # -- the mirror list -----------------------------------------------------

    candidates: CollectionProperty(type=UVVAR_CandidateItem, name="Candidates")

    mirror_object: StringProperty(
        name="Mirrored Object",
        description=(
            "Name of the object the candidate list currently mirrors. The list "
            "lives on the scene while candidates belong to an object, so this "
            "is what keeps the two from drifting apart"
        ),
        default="",
    )

    def active(self) -> UVVAR_CandidateItem | None:
        """Return the currently selected mirror row, or ``None``."""
        idx = int(self.active_index)
        if 0 <= idx < len(self.candidates):
            return self.candidates[idx]
        return None


# ---------------------------------------------------------------------------
# Mirror synchronisation
# ---------------------------------------------------------------------------


def get_props(context: Context | None = None) -> UVVAR_SceneProps | None:
    """Return ``scene.uvvar``, or ``None`` when it is not registered.

    Args:
        context: Optional context; ``bpy.context`` is used when omitted.

    Returns:
        The scene property group, or ``None``.  Draw and poll code must handle
        ``None`` because an add-on reload briefly leaves the pointer missing.
    """
    ctx = context if context is not None else bpy.context
    scene = getattr(ctx, "scene", None)
    if scene is None:
        return None
    return getattr(scene, "uvvar", None)


def sync_ui_list(scene: Scene | None, obj: Any, *, store: Any = None) -> int:
    """Rebuild ``scene.uvvar.candidates`` from the runtime store.

    Rebuilding wholesale rather than diffing is intentional: the list is at most
    a few dozen rows, and a full rebuild cannot drift out of sync with the
    store, which a partial update eventually will.

    Args:
        scene: The scene holding the property group; ``None`` is tolerated.
        obj: The object whose candidates should be mirrored.
        store: Optional store override; looked up from the cache otherwise.

    Returns:
        The number of rows written.
    """
    props = getattr(scene, "uvvar", None) if scene is not None else None
    if props is None:
        return 0

    active_signature = ""
    current = props.active()
    if current is not None:
        active_signature = current.signature

    props.candidates.clear()
    try:
        props.mirror_object = str(getattr(obj, "name", "") or "")
    except Exception:  # noqa: BLE001 - the stamp is bookkeeping, not the data.
        _LOG.debug("could not stamp mirror_object", exc_info=True)
    if store is None:
        store = get_store(obj)
    if store is None:
        props.active_index = 0
        return 0

    weights = score_weights(props)
    try:
        rank_it = weights is not None and props.sort_by_score
        items = store.ranked(weights) if rank_it else store.all()
    except Exception:  # noqa: BLE001 - ranking must never break the panel.
        _LOG.debug("ranking failed; falling back to store order", exc_info=True)
        try:
            items = store.all()
        except Exception:  # noqa: BLE001
            items = []

    if not props.show_failed:
        kept = [c for c in items if getattr(c, "ok", True)]
        items = kept or items

    pareto = _pareto_indices(items)

    written = 0
    restored = -1
    for rank, candidate in enumerate(items, start=1):
        row = props.candidates.add()
        try:
            row.fill_from(candidate, rank=rank)
            row.is_pareto = int(getattr(candidate, "index", -1)) in pareto
        except Exception:  # noqa: BLE001
            _LOG.debug("could not mirror candidate %r", candidate, exc_info=True)
        if active_signature and row.signature == active_signature:
            restored = written
        written += 1

    # Preserve the user's selection across a re-rank when possible; otherwise
    # clamp rather than leaving an index pointing past the end.
    if restored >= 0:
        _set_index_quietly(props, restored)
    elif written:
        _set_index_quietly(props, min(int(props.active_index), written - 1))
    else:
        _set_index_quietly(props, 0)
    return written


def _pareto_indices(items: Any) -> set[int]:
    """Return the store indices of the non-dominated candidates.

    The Pareto front is what tells an artist "these are the genuinely different
    good answers" as opposed to "these scored well under the current weights",
    so the panel highlights it.  Computed here, once per resync, rather than in
    draw code.

    Args:
        items: Candidates to consider.

    Returns:
        A set of ``Candidate.index`` values, empty when the front cannot be
        computed (no metrics module, no metrics, or a failure -- none of which
        is worth an error).
    """
    try:
        from .core.metrics import pareto_front

        usable = [c for c in items if getattr(c, "metrics", None) is not None]
        if not usable:
            return set()
        front = pareto_front([c.metrics for c in usable])
        return {int(usable[i].index) for i in front if 0 <= int(i) < len(usable)}
    except Exception:  # noqa: BLE001 - a highlight is never worth an exception.
        _LOG.debug("pareto front unavailable", exc_info=True)
        return set()


def _set_index_quietly(props: UVVAR_SceneProps, index: int) -> None:
    """Set ``active_index`` without triggering an auto-preview.

    A resync happens on every re-rank and on every panel refresh; firing a UV
    blit each time would make changing the sort order feel like a stutter.
    """
    auto = bool(props.auto_preview)
    try:
        props.auto_preview = False
        props.active_index = max(0, int(index))
    finally:
        props.auto_preview = auto


def candidate_rows(props: UVVAR_SceneProps | None) -> Iterable[UVVAR_CandidateItem]:
    """Iterate the mirror rows safely, yielding nothing when unregistered."""
    if props is None:
        return ()
    return props.candidates


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


#: Owner token for the message-bus subscription.  Any hashable object works;
#: it exists so ``clear_by_owner`` can remove exactly our subscription.
_MSGBUS_OWNER = object()


def on_active_object_changed() -> None:
    """Re-point the mirror list at whatever object is now active.

    Candidates belong to an *object*; the mirror list lives on the *scene*.
    Without this the panel keeps showing the previous object's grid after the
    user clicks something else -- rows that look live, are ranked, and whose
    buttons the operators then quite correctly refuse to act on.  Resyncing on
    the change is the only way the two views can agree.

    Driven from **two** places, because neither is sufficient alone.
    ``bpy.msgbus`` fires when the user clicks an object in the viewport or the
    outliner, which is the common case, but a programmatic
    ``view_layer.objects.active = obj`` does not publish and so never notifies.
    The depsgraph handler covers exactly that gap.  Both funnel through this
    one function, and the name comparison below makes the duplicate call free.

    That comparison is also what stops the depsgraph route from looping:
    writing the mirror list tags the scene, which schedules another depsgraph
    update, whose call finds the name already matching and returns immediately.

    Never raises: a callback that throws here is reported on every selection
    click for the rest of the session.
    """
    try:
        context = bpy.context
        props = get_props(context)
        if props is None:
            return
        obj = getattr(context, "object", None) or getattr(context, "active_object", None)
        name = str(getattr(obj, "name", "") or "")
        if name == str(props.mirror_object):
            return
        store = store_for(obj, create=False) if obj is not None else None
        sync_ui_list(getattr(context, "scene", None), obj, store=store)
        _LOG.debug("mirror list re-pointed at %r", name or "<nothing>")
    except Exception:  # noqa: BLE001 - a notify callback must never raise.
        _LOG.debug("active-object resync failed", exc_info=True)


@persistent
def _on_depsgraph_update(_scene: Any, _depsgraph: Any = None) -> None:
    """Catch active-object changes the message bus does not publish.

    Selecting a different object from a script -- which is what an artist's own
    tooling, and every add-on that calls ``view_layer.objects.active = obj``,
    actually does -- updates the RNA without notifying the message bus.  This
    handler is the backstop; see :func:`on_active_object_changed` for why the
    write it may trigger cannot loop.
    """
    on_active_object_changed()


def _subscribe_active_object() -> None:
    """Subscribe to active-object changes, replacing any earlier subscription."""
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.LayerObjects, "active"),
            owner=_MSGBUS_OWNER,
            args=(),
            notify=on_active_object_changed,
        )
    except Exception:  # noqa: BLE001 - no message bus (headless) is not an error.
        _LOG.debug("could not subscribe to active-object changes", exc_info=True)


def _unsubscribe_active_object() -> None:
    """Drop the active-object subscription."""
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    except Exception:  # noqa: BLE001
        _LOG.debug("could not clear the message-bus owner", exc_info=True)


@persistent
def _on_load_post(_dummy: Any) -> None:
    """Reset per-file runtime state after a new file is opened.

    The store cache is keyed by object name, and names collide freely between
    files; without this, opening a second file could hand a stale UV array --
    sized for a different mesh -- to the preview blit.  The overlay is
    resynchronised at the same time because ``overlay_mode`` is saved in the
    ``.blend`` while the GPU draw handler that implements it is not, and the
    message-bus subscription is remade because Blender drops every subscription
    when it loads a file.
    """
    drop_all_stores()
    try:
        from .operators.overlay_ops import sync_overlay_state

        sync_overlay_state(bpy.context)
    except Exception:  # noqa: BLE001 - a handler must never raise into Blender.
        _LOG.debug("overlay resync after file load failed", exc_info=True)
    _subscribe_active_object()
    _LOG.debug("cleared candidate caches on file load")


#: ``(handler list attribute, function)`` pairs installed by this module.
_HANDLERS: tuple[tuple[str, Any], ...] = (
    ("load_post", _on_load_post),
    ("depsgraph_update_post", _on_depsgraph_update),
)


def _install_handlers() -> None:
    """Add each handler exactly once and open the message-bus subscription."""
    for attribute, function in _HANDLERS:
        handlers = getattr(bpy.app.handlers, attribute, None)
        if handlers is None:
            continue
        if function not in handlers:
            handlers.append(function)
    _subscribe_active_object()


def _remove_handlers() -> None:
    """Remove every handler this module installed, and the subscription.

    Compares by name as well as by identity because a module reload creates a
    *new* function object while the old one is still in the handler list, which
    is the classic way an add-on leaks a handler across F8.
    """
    _unsubscribe_active_object()
    for attribute, function in _HANDLERS:
        handlers = getattr(bpy.app.handlers, attribute, None)
        if handlers is None:
            continue
        name = getattr(function, "__name__", "")
        for handler in list(handlers):
            if handler is function or getattr(handler, "__name__", "") == name:
                try:
                    handlers.remove(handler)
                except Exception:  # noqa: BLE001
                    _LOG.debug("could not remove %s handler", attribute, exc_info=True)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: Classes in registration order.  ``UVVAR_CandidateItem`` **must** come first:
#: ``UVVAR_SceneProps.candidates`` refers to it by type, and Blender resolves
#: that reference at registration time.
classes: tuple[type, ...] = (
    UVVAR_CandidateItem,
    UVVAR_SceneProps,
)


def register() -> None:
    """Register the property groups and attach ``Scene.uvvar``."""
    for cls in classes:
        bpy.utils.register_class(cls)
    Scene.uvvar = PointerProperty(
        type=UVVAR_SceneProps,
        name="Unwrap Variations",
        description="Settings and results for the UnwrapVariations add-on",
    )
    _install_handlers()


def unregister() -> None:
    """Detach ``Scene.uvvar``, remove handlers and unregister the groups.

    Order matters and is the exact reverse of :func:`register`: the pointer
    property has to go before the class it points at, or Blender is left with a
    dangling RNA reference that only shows itself as a crash much later.
    """
    _remove_handlers()
    drop_all_stores()
    try:
        del Scene.uvvar
    except Exception:  # noqa: BLE001 - already gone is a fine outcome.
        _LOG.debug("Scene.uvvar was not present at unregister")
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:  # noqa: BLE001 - keep going so the rest still unregisters.
            _LOG.debug("could not unregister %s", cls.__name__, exc_info=True)
