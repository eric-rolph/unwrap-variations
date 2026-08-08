"""The unwrap parameter space: definition, diverse sampling and mutation.

This module owns everything about *what knobs exist* and *how to explore them*.
It is deliberately free of :mod:`bpy` so the search strategy can be unit-tested
and reasoned about without launching Blender.

The two entry points are :func:`sample_grid` -- give me N maximally different
unwraps -- and :func:`mutate` -- give me N variations *around* this one.  That
pair is the whole "variational" workflow in miniature.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

__all__ = [
    "UnwrapParams",
    "ParamSpec",
    "PARAM_SPACE",
    "METHODS",
    "SEAM_STRATEGIES",
    "ARCHETYPES",
    "chart_recipe",
    "sample_grid",
    "mutate",
]

#: Unwrap methods understood by the built-in backend.  ``ANGLE_BASED``,
#: ``CONFORMAL`` and ``MINIMUM_STRETCH`` map onto ``bpy.ops.uv.unwrap``; the
#: projection methods map onto their own operators and ignore seams.
METHODS = (
    "ANGLE_BASED",
    "CONFORMAL",
    "MINIMUM_STRETCH",
    "SMART_PROJECT",
    "CUBE_PROJECT",
)

#: Seam generation strategies.  Mirrored in :mod:`unwrap_variations.backends.seams`.
SEAM_STRATEGIES = (
    "NONE",
    "EXISTING",
    "SHARP",
    "ANGLE",
    "CURVATURE",
    "DUAL_GRAPH",
    "CONCAVE",
    "SMART",
)

MARGIN_METHODS = ("SCALED", "ADD", "FRACTION")
ROTATE_METHODS = ("AXIS_ALIGNED", "AXIS_ALIGNED_X", "AXIS_ALIGNED_Y")
PACK_SHAPE_METHODS = ("CONCAVE", "CONVEX", "AABB")


@dataclass(frozen=True)
class UnwrapParams:
    """One complete recipe for producing a UV layout.

    Every field is JSON-serialisable so a candidate can be persisted into a
    ``.blend`` and restored later.  Angles are stored in **degrees** throughout
    the core; conversion to the radians that ``bpy.ops.uv.smart_project``
    expects happens once, at the backend boundary.
    """

    backend: str = "blender_native"
    method: str = "ANGLE_BASED"
    angle_limit_deg: float = 66.0
    island_margin: float = 0.01
    area_weight: float = 0.0
    correct_aspect: bool = True
    scale_to_bounds: bool = False
    fill_holes: bool = True
    no_flip: bool = False
    iterations: int = 10
    margin_method: str = "SCALED"
    rotate_method: str = "AXIS_ALIGNED_Y"
    pack_rotate: bool = True
    pack_shape_method: str = "CONCAVE"
    seam_strategy: str = "SHARP"
    seam_angle_deg: float = 30.0
    curvature_threshold: float = 0.35
    minimize_stretch_iters: int = 0
    average_islands_scale: bool = False
    max_stretch: float = 1.25
    seed: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- identity ------------------------------------------------------------

    def signature(self) -> str:
        """Return a stable 12-character hash of the semantic content.

        Used to de-duplicate candidates so a grid never wastes a slot on two
        identical recipes.  Floats are rounded before hashing so that values
        differing only by float noise collapse together.  Fields that do not
        affect the result for the chosen method are excluded, which is why two
        ``CUBE_PROJECT`` recipes differing only in ``seam_strategy`` correctly
        hash the same.
        """
        payload = self._semantic_dict()
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def _semantic_dict(self) -> dict[str, Any]:
        """Return only the fields that actually influence this recipe's output."""
        d: dict[str, Any] = {
            "backend": self.backend,
            "method": self.method,
            "island_margin": round(float(self.island_margin), 6),
            "margin_method": self.margin_method,
            "correct_aspect": bool(self.correct_aspect),
            "pack_rotate": bool(self.pack_rotate),
            "pack_shape_method": self.pack_shape_method,
            "average_islands_scale": bool(self.average_islands_scale),
            # Chart merging runs after every method, projections included, so
            # this belongs in the identity of every recipe.  Two candidates that
            # differ only in whether their islands are merged are genuinely
            # different layouts, and collapsing them here would mean the grid
            # could never hold both -- which
            # ``docs/ORGANIC.md`` section 5 requires it to.
            "max_stretch": round(float(self.max_stretch), 4),
            "extra": dict(self.extra),
        }
        if self.method == "SMART_PROJECT":
            d["angle_limit_deg"] = round(float(self.angle_limit_deg), 3)
            d["area_weight"] = round(float(self.area_weight), 4)
            d["rotate_method"] = self.rotate_method
            d["scale_to_bounds"] = bool(self.scale_to_bounds)
        elif self.method == "CUBE_PROJECT":
            d["scale_to_bounds"] = bool(self.scale_to_bounds)
        else:
            # Seam-driven solvers.
            d["fill_holes"] = bool(self.fill_holes)
            d["no_flip"] = bool(self.no_flip)
            d["iterations"] = int(self.iterations)
            d["seam_strategy"] = self.seam_strategy
            d["minimize_stretch_iters"] = int(self.minimize_stretch_iters)
            if self.seam_strategy in ("ANGLE", "SHARP"):
                d["seam_angle_deg"] = round(float(self.seam_angle_deg), 2)
            elif self.seam_strategy == "CURVATURE":
                d["seam_angle_deg"] = round(float(self.seam_angle_deg), 2)
                d["curvature_threshold"] = round(float(self.curvature_threshold), 4)
            elif self.seam_strategy in ("DUAL_GRAPH", "CONCAVE"):
                d["curvature_threshold"] = round(float(self.curvature_threshold), 4)
                d["seed"] = int(self.seed)
        return d

    def describe(self) -> str:
        """Return a short label for the UI, at most 40 characters."""
        short = {
            "ANGLE_BASED": "ABF",
            "CONFORMAL": "LSCM",
            "MINIMUM_STRETCH": "MinStretch",
            "SMART_PROJECT": "Smart",
            "CUBE_PROJECT": "Cube",
        }.get(self.method, self.method[:10])
        if self.method == "SMART_PROJECT":
            detail = "%.0fdeg" % self.angle_limit_deg
        elif self.method == "CUBE_PROJECT":
            detail = "proj"
        else:
            detail = {
                "NONE": "noseam",
                "EXISTING": "keep",
                "SHARP": "sharp",
                "ANGLE": "ang%.0f" % self.seam_angle_deg,
                "CURVATURE": "curv%.2f" % self.curvature_threshold,
                "DUAL_GRAPH": "dual",
                "CONCAVE": "conc%.2f" % self.curvature_threshold,
                "SMART": "smart",
            }.get(self.seam_strategy, self.seam_strategy[:6])
            charts = self.chart_clusters()
            if charts:
                detail = "chart%d" % charts
        return ("%s %s m%.3g" % (short, detail, self.island_margin))[:40]

    def chart_clusters(self) -> int:
        """Return the chart-cut cluster count this recipe asks for, or ``0``.

        The request lives in :attr:`extra` (see
        :data:`unwrap_variations.core.engine.CHART_CUT_KEY`) rather than in a
        dedicated field, because ``extra`` already participates in
        :meth:`signature` and adding a field would change the persisted record
        layout that ARCHITECTURE section 3 fixes.
        """
        extra = self.extra if isinstance(self.extra, dict) else {}
        try:
            return max(0, int(extra.get("chart_cut", 0) or 0))
        except (TypeError, ValueError):
            return 0

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of every field."""
        d = asdict(self)
        d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnwrapParams":
        """Rebuild params from a dict, tolerating unknown and missing keys.

        Forward compatibility matters here: a ``.blend`` saved by a newer
        version of the add-on must still open in an older one, so unrecognised
        keys are dropped rather than raising.
        """
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean: dict[str, Any] = {}
        for key, value in dict(data or {}).items():
            if key in known:
                clean[key] = value
        out = cls(**clean)
        return out.clamped()

    def replace(self, **kwargs: Any) -> "UnwrapParams":
        """Return a copy with ``kwargs`` overridden, clamped back into range."""
        return replace(self, **kwargs).clamped()

    def clamped(self) -> "UnwrapParams":
        """Return a copy with every field forced into its legal range.

        Called after deserialisation and after mutation so that no downstream
        code -- least of all a Blender operator -- ever sees an out-of-range
        value.
        """
        changes: dict[str, Any] = {}
        for name, spec in PARAM_SPACE.items():
            value = getattr(self, name)
            fixed = spec.clamp(value)
            if fixed != value:
                changes[name] = fixed
        if self.method not in METHODS:
            changes["method"] = "ANGLE_BASED"
        if self.seam_strategy not in SEAM_STRATEGIES:
            changes["seam_strategy"] = "SHARP"
        if not isinstance(self.extra, dict):
            changes["extra"] = dict(self.extra or {})
        return replace(self, **changes) if changes else self


@dataclass(frozen=True)
class ParamSpec:
    """Describes one mutable field of :class:`UnwrapParams`.

    Attributes:
        kind: One of ``"float"``, ``"int"``, ``"bool"`` or ``"enum"``.
        low: Lower bound for numeric kinds.
        high: Upper bound for numeric kinds.
        choices: Legal values for ``"enum"`` kind.
        weight: Relative probability that :func:`mutate` perturbs this field.
            Fields with a large visual impact get a higher weight.
        log: When ``True`` the field is perturbed multiplicatively, which suits
            values like ``island_margin`` that span orders of magnitude.
        applies_to: Methods this field affects; ``None`` means all of them.
    """

    kind: str
    low: float = 0.0
    high: float = 1.0
    choices: tuple[Any, ...] = ()
    weight: float = 1.0
    log: bool = False
    applies_to: tuple[str, ...] | None = None

    def clamp(self, value: Any) -> Any:
        """Coerce ``value`` into this spec's legal range or choice set.

        Deliberately total: every input, however malformed, yields a legal
        value.  Non-finite numbers are saturated towards the matching bound
        rather than discarded, because ``+inf`` plainly means "as large as
        allowed".  This matters on the load path -- ``json`` happily writes and
        reads back ``Infinity``, so a hand-edited or corrupted ``.blend`` can
        deliver one, and ``int(float("inf"))`` raises ``OverflowError``, which
        is neither a ``TypeError`` nor a ``ValueError``.
        """
        try:
            if self.kind == "bool":
                return bool(value)
            if self.kind == "enum":
                return value if value in self.choices else self.choices[0]

            number = float(value)
            if math.isnan(number):
                number = float(self.low)
            elif math.isinf(number):
                number = float(self.high) if number > 0 else float(self.low)

            if self.kind == "int":
                return int(min(max(int(number), self.low), self.high))
            return float(min(max(number, self.low), self.high))
        except (TypeError, ValueError, OverflowError):
            if self.kind == "enum":
                return self.choices[0]
            if self.kind == "bool":
                return False
            return type(self.low)(self.low)


#: The searchable parameter space.  Anything absent here is held fixed.
PARAM_SPACE: dict[str, ParamSpec] = {
    "method": ParamSpec("enum", choices=METHODS, weight=2.0),
    "angle_limit_deg": ParamSpec(
        "float", 1.0, 89.0, weight=2.0, applies_to=("SMART_PROJECT",)
    ),
    "island_margin": ParamSpec("float", 0.0, 0.2, weight=1.2, log=True),
    "area_weight": ParamSpec("float", 0.0, 1.0, weight=1.0, applies_to=("SMART_PROJECT",)),
    "correct_aspect": ParamSpec("bool", weight=0.4),
    "scale_to_bounds": ParamSpec(
        "bool", weight=0.3, applies_to=("SMART_PROJECT", "CUBE_PROJECT")
    ),
    "fill_holes": ParamSpec(
        "bool", weight=0.4, applies_to=("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")
    ),
    "no_flip": ParamSpec(
        "bool", weight=0.5, applies_to=("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")
    ),
    "iterations": ParamSpec(
        "int", 1, 100, weight=0.5, applies_to=("MINIMUM_STRETCH",)
    ),
    "margin_method": ParamSpec("enum", choices=MARGIN_METHODS, weight=0.4),
    "rotate_method": ParamSpec(
        "enum", choices=ROTATE_METHODS, weight=0.4, applies_to=("SMART_PROJECT",)
    ),
    "pack_rotate": ParamSpec("bool", weight=0.6),
    "pack_shape_method": ParamSpec("enum", choices=PACK_SHAPE_METHODS, weight=0.6),
    "seam_strategy": ParamSpec(
        "enum",
        choices=SEAM_STRATEGIES,
        weight=2.0,
        applies_to=("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH"),
    ),
    "seam_angle_deg": ParamSpec(
        "float",
        5.0,
        120.0,
        weight=1.5,
        applies_to=("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH"),
    ),
    "curvature_threshold": ParamSpec(
        "float",
        0.02,
        0.95,
        weight=1.2,
        applies_to=("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH"),
    ),
    "minimize_stretch_iters": ParamSpec("int", 0, 60, weight=0.8),
    "average_islands_scale": ParamSpec("bool", weight=0.5),
    # Distortion ceiling for chart merging
    # (:func:`unwrap_variations.core.charts.merge_charts`).  The range starts
    # *below* 1.0 on purpose: ``stretch_l2 >= 1.0`` is a theorem, so any value
    # under it makes every merge unsatisfiable and reproduces the unmerged
    # layout exactly.  ORGANIC.md section 5 requires that layout to stay
    # reachable from the grid so a benchmark can attribute the gain to merging
    # rather than to the whole pipeline at once, and a bound that merely
    # *discourages* merging would not prove anything.  The Halton fill therefore
    # spends 62% of its samples with merging off and 38% with it on.
    "max_stretch": ParamSpec("float", 0.0, 1.6, weight=1.5),
}


def chart_recipe(
    clusters: int,
    *,
    method: str = "ANGLE_BASED",
    chart_seed: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    """Return an archetype dict asking the engine for a ``clusters``-chart cut.

    The engine (:func:`unwrap_variations.core.engine.chart_seams`) segments the
    surface into charts and hands the resulting seams to the backend through the
    sample, so the recipe itself only has to say "keep the seams you are given"
    and carry the request in :attr:`UnwrapParams.extra`.

    Args:
        clusters: Number of seed clusters; more means more, flatter charts.
        method: Which seam-driven solver flattens the charts.
        chart_seed: Rotates the seed directions so two recipes with the same
            cluster count still produce different charts.
        **overrides: Any other :class:`UnwrapParams` field.

    Returns:
        A dict suitable for :data:`ARCHETYPES` or ``UnwrapParams.replace``.
    """
    recipe: dict[str, Any] = {
        "method": method,
        "seam_strategy": "EXISTING",
        "island_margin": 0.01,
        "extra": {"chart_cut": int(clusters), "chart_seed": int(chart_seed)},
    }
    recipe.update(overrides)
    return recipe


#: Hand-picked recipes spanning the genuinely useful corners of the space.
#: A grid seeds from these before falling back to quasi-random fill, so the
#: first few slots are always recognisable, production-sensible layouts rather
#: than random noise.
#:
#: The ``chart_recipe`` entries are the ones that carry a closed mesh.  A sphere,
#: a torus and an icosphere have no sharp edges and no boundary, so every
#: seam heuristic returns an empty cut and every seam-driven solver on them is a
#: dead slot; a chart cut is guaranteed to open them, and measured on Blender
#: 4.5.4 it is what wins on all three (icosphere 80.4 against 66.8 for the best
#: recipe available before, torus 75.3 against 53.8, sphere 81.7 against 72.7).
ARCHETYPES: tuple[dict[str, Any], ...] = (
    # Balanced conformal unwrap along sharp edges - the everyday default, and
    # the grid's pinned **unmerged control**.  ``max_stretch`` below 1.0 makes
    # chart merging unsatisfiable (``stretch_l2 >= 1.0`` is a theorem), so this
    # slot reproduces the pre-merge pipeline exactly.  ORGANIC.md section 5
    # requires the unmerged layout to stay reachable so a benchmark can
    # attribute the gain to merging; it has to be *here*, in the pinned leading
    # archetypes, because a ten-slot grid is filled entirely from this list and
    # never reaches the quasi-random fill that samples ``PARAM_SPACE``.
    # ``1.0`` would *not* do: an exactly isometric merge -- two coplanar islands
    # on a hard-surface model -- measures exactly 1.0 and is a free win worth
    # keeping, so only a bound strictly below it means "off".
    {
        "method": "ANGLE_BASED",
        "seam_strategy": "SHARP",
        "island_margin": 0.01,
        "max_stretch": 0.0,
    },
    # Classic Smart UV Project at Blender's own default angle.
    {
        "method": "SMART_PROJECT",
        "angle_limit_deg": 66.0,
        "island_margin": 0.01,
        "area_weight": 0.0,
    },
    # Six charts, conformal: the "cube map" answer, and the one that carries a
    # closed smooth surface.
    chart_recipe(6, method="ANGLE_BASED"),
    # Three big charts relaxed for stretch: the fewest cuts that still avoid a
    # folded-over layout on an organic model.
    chart_recipe(3, method="MINIMUM_STRETCH", chart_seed=3, iterations=30),
    # Twelve charts: small, flat, fold-free - the safe pick on a lumpy organic
    # model where a coarse chart folds over itself.
    chart_recipe(12, method="ANGLE_BASED", chart_seed=11),
    # Few, well-placed cuts from the dual-graph solver. On a faceted mesh this
    # is the isometric polyhedral net, and nothing else in the set produces it:
    # a cube comes out as its 7-edge cross, one island, stretch exactly 1.0.
    {
        "method": "ANGLE_BASED",
        "seam_strategy": "DUAL_GRAPH",
        "curvature_threshold": 0.35,
        "island_margin": 0.005,
    },
    # ---- end of the always-tried core (see _ARCHETYPE_CORE) ----------------
    # Concavity-weighted minimum cut: the organic-mesh answer, and the first
    # entry of the shuffled tail.  It is the only strategy here that both
    # *always* opens a closed surface and places its cuts where an artist
    # would, so it is the natural counterpart to the chart cuts above -- those
    # open a closed mesh too, but land their seams on cube-map boundaries in
    # the middle of a face (BENCHMARKS.md section 4.4).
    {
        "method": "ANGLE_BASED",
        "seam_strategy": "CONCAVE",
        "curvature_threshold": 0.35,
        "island_margin": 0.01,
    },
    # The cube-map charts relaxed for stretch: best texel-density uniformity.
    chart_recipe(6, method="MINIMUM_STRETCH", chart_seed=7, iterations=30),
    # Aggressive Smart Project: many small, low-distortion islands.
    {
        "method": "SMART_PROJECT",
        "angle_limit_deg": 30.0,
        "island_margin": 0.02,
        "area_weight": 0.5,
    },
    # Two charts - front and back.  The shortest cut that opens a closed
    # surface at all, and the pick when every seam is expensive.
    chart_recipe(2, method="MINIMUM_STRETCH", chart_seed=13, iterations=30),
    # Minimum-stretch relaxation on sharp-angle seams.
    {
        "method": "MINIMUM_STRETCH",
        "seam_strategy": "ANGLE",
        "seam_angle_deg": 40.0,
        "iterations": 30,
        "island_margin": 0.01,
    },
    # Relaxed Smart Project with a high angle limit: few, large islands.
    {
        "method": "SMART_PROJECT",
        "angle_limit_deg": 88.0,
        "island_margin": 0.01,
        "area_weight": 0.0,
    },
    # LSCM on curvature seams: organic shapes.
    {
        "method": "CONFORMAL",
        "seam_strategy": "CURVATURE",
        "curvature_threshold": 0.2,
        "island_margin": 0.01,
    },
    # Tight packing for atlases: minimal margin, concave packing, no rotation lock.
    {
        "method": "ANGLE_BASED",
        "seam_strategy": "ANGLE",
        "seam_angle_deg": 30.0,
        "island_margin": 0.001,
        "pack_shape_method": "CONCAVE",
        "average_islands_scale": True,
    },
    # Four charts, packed tight and scale-normalised: the atlas pick.
    chart_recipe(
        4,
        method="MINIMUM_STRETCH",
        chart_seed=5,
        iterations=45,
        island_margin=0.005,
        average_islands_scale=True,
    ),
    # Respect whatever seams the artist already marked.  With none marked the
    # engine escalates this to a chart cut rather than burning the slot.
    {"method": "ANGLE_BASED", "seam_strategy": "EXISTING", "island_margin": 0.01},
    # Hard-surface planar fallback.
    {"method": "CUBE_PROJECT", "island_margin": 0.02, "scale_to_bounds": False},
    # LSCM on sharp edges with post-relaxation.
    {
        "method": "CONFORMAL",
        "seam_strategy": "SHARP",
        "minimize_stretch_iters": 20,
        "island_margin": 0.01,
    },
)

#: How many leading :data:`ARCHETYPES` a grid always tries, in order, before it
#: shuffles the rest.
#:
#: The shuffle exists so that different seeds explore differently, but with more
#: archetypes than slots it also means a ten-slot grid drops six of them at
#: random -- and the recipes that actually win are then present only by luck.
#: "Best-of-N beats a single pass" must not be a coin flip on the seed, so the
#: handful that carry the common cases are pinned to the front and only the tail
#: is shuffled.  Measured across seeds 1, 2, 7 and 42 the winner is the same
#: recipe on every mesh with this in place.
_ARCHETYPE_CORE = 6

#: Cluster counts and solvers the grid's quasi-random fill draws chart cuts
#: from.  Kept separate from :data:`ARCHETYPES` because these exist to fill
#: *remaining* slots with genuinely different layouts, not to be a curated set.
_CHART_FILL_CLUSTERS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
_CHART_FILL_METHODS = ("ANGLE_BASED", "MINIMUM_STRETCH", "CONFORMAL")


def _halton(index: int, base: int) -> float:
    """Return the ``index``-th value of the Halton low-discrepancy sequence.

    Halton gives us evenly spread samples without the clumping that plain
    uniform random produces, so a 10-slot grid actually covers the space.
    """
    result = 0.0
    fraction = 1.0
    i = index + 1
    while i > 0:
        fraction /= base
        result += fraction * (i % base)
        i //= base
    return result


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _applies(name: str, method: str) -> bool:
    """Return whether parameter ``name`` has any effect for ``method``."""
    spec = PARAM_SPACE.get(name)
    if spec is None or spec.applies_to is None:
        return True
    return method in spec.applies_to


def _quasi_random_params(slot: int, seed: int, base: UnwrapParams) -> UnwrapParams:
    """Build one quasi-random parameter set for grid ``slot``."""
    rng = random.Random((seed * 7919 + slot * 104729) & 0xFFFFFFFF)
    # Offset the Halton index by the seed so different seeds explore differently.
    hi = slot + (seed % 977) * 13
    method = METHODS[int(_halton(hi, _PRIMES[0]) * len(METHODS)) % len(METHODS)]
    values: dict[str, Any] = {"method": method, "seed": rng.randrange(1 << 30)}

    dim = 1
    for name, spec in PARAM_SPACE.items():
        if name == "method":
            continue
        if not _applies(name, method):
            continue
        u = _halton(hi, _PRIMES[dim % len(_PRIMES)])
        dim += 1
        if spec.kind == "bool":
            values[name] = u > 0.5
        elif spec.kind == "enum":
            values[name] = spec.choices[int(u * len(spec.choices)) % len(spec.choices)]
        elif spec.kind == "int":
            values[name] = int(round(spec.low + u * (spec.high - spec.low)))
        elif spec.log:
            lo = max(float(spec.low), 1e-4)
            hi_v = max(float(spec.high), lo * 1.0001)
            values[name] = float(math.exp(math.log(lo) + u * (math.log(hi_v) - math.log(lo))))
        else:
            values[name] = float(spec.low + u * (spec.high - spec.low))
    return base.replace(**values)


def sample_grid(
    count: int,
    *,
    seed: int = 0,
    base: UnwrapParams | None = None,
) -> list[UnwrapParams]:
    """Return ``count`` maximally distinct unwrap recipes.

    The first slots come from :data:`ARCHETYPES` -- curated, production-sensible
    layouts.  Its leading :data:`_ARCHETYPE_CORE` entries are always taken, in
    order; the rest are shuffled by ``seed``.  Any remainder alternates between
    Halton-sequence quasi-random samples, which spread across the space far more
    evenly than uniform random, and chart cuts at increasing cluster counts,
    which are the only fill that reliably solves a closed mesh.  Results are
    de-duplicated by :meth:`UnwrapParams.signature` and are deterministic for a
    given ``seed``.

    Args:
        count: How many recipes to return.  Values below 1 yield an empty list.
        seed: Determinism seed; different seeds explore different regions.
        base: Optional starting params whose non-searched fields are inherited.

    Returns:
        A list of at most ``count`` unique :class:`UnwrapParams`.
    """
    if count <= 0:
        return []
    root = (base or UnwrapParams()).clamped()

    out: list[UnwrapParams] = []
    seen: set[str] = set()

    # Pin the leading archetypes so the grid's coverage does not depend on the
    # seed, and shuffle only the tail.  See _ARCHETYPE_CORE.
    # Never pin more than two thirds of the slots, so a small grid still gets
    # variety and a seed still means something.
    core = min(_ARCHETYPE_CORE, max(1, (count * 2) // 3), len(ARCHETYPES))
    order = list(range(core))
    tail = list(range(core, len(ARCHETYPES)))
    if seed:
        random.Random(seed).shuffle(tail)
    order.extend(tail)

    for idx in order:
        if len(out) >= count:
            break
        cand = root.replace(seed=seed, **ARCHETYPES[idx])
        sig = cand.signature()
        if sig not in seen:
            seen.add(sig)
            out.append(cand)

    slot = 0
    guard = 0
    while len(out) < count and guard < count * 200:
        guard += 1
        # Alternate between a quasi-random point and a chart cut.  Quasi-random
        # alone fills a closed mesh's spare slots with recipes that cannot
        # possibly solve it (no boundary, no sharp edges, no seams), which is
        # how a sphere used to come back with eight of ten slots filled and four
        # of those duplicates.  A chart cut always opens the surface, and a
        # different cluster count is a genuinely different layout, not a
        # different seed on the same one.
        if slot % 2 == 1:
            index = (slot // 2) + (seed % len(_CHART_FILL_CLUSTERS))
            cand = root.replace(
                seed=seed,
                **chart_recipe(
                    _CHART_FILL_CLUSTERS[index % len(_CHART_FILL_CLUSTERS)],
                    method=_CHART_FILL_METHODS[index % len(_CHART_FILL_METHODS)],
                    chart_seed=(seed * 31 + index * 17) & 0xFFFF,
                    iterations=30,
                ),
            )
        else:
            cand = _quasi_random_params(slot // 2, seed, root)
        slot += 1
        sig = cand.signature()
        if sig not in seen:
            seen.add(sig)
            out.append(cand)
    return out[:count]


def mutate(
    parent: UnwrapParams,
    *,
    count: int,
    strength: float = 0.35,
    seed: int = 0,
) -> list[UnwrapParams]:
    """Return ``count`` perturbations clustered around ``parent``.

    This is the "give me 10 more like that one" half of the workflow.  Each
    child perturbs a random subset of the applicable fields, with the number of
    perturbed fields and the size of each perturbation both scaling with
    ``strength``.  At low strength you get near-siblings; at high strength the
    method itself may flip, which keeps deep branches from stagnating.

    Args:
        parent: The recipe to vary around.
        count: How many children to produce.
        strength: Mutation magnitude in ``[0, 1]``.
        seed: Determinism seed.

    Returns:
        Up to ``count`` unique children, none identical to ``parent``.
    """
    if count <= 0:
        return []
    parent = parent.clamped()
    strength = min(max(float(strength), 0.0), 1.0)

    out: list[UnwrapParams] = []
    seen: set[str] = {parent.signature()}

    guard = 0
    child_index = 0
    while len(out) < count and guard < count * 400:
        guard += 1
        rng = random.Random((seed * 6151 + child_index * 24593 + guard) & 0xFFFFFFFF)
        child_index += 1
        cand = _mutate_once(parent, rng, strength)
        sig = cand.signature()
        if sig not in seen:
            seen.add(sig)
            out.append(cand)
    return out


#: Seam-driven solvers, mirrored from :data:`METHODS`; the chart cut only means
#: anything for these, since a projection invents its own charts.
_SOLVER_METHODS = ("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")


def _mutate_chart(
    extra: dict[str, Any], method: str, rng: random.Random, strength: float
) -> tuple[dict[str, Any], bool]:
    """Perturb, introduce or drop the chart-cut request in ``extra``.

    The chart cut is by far the biggest lever on the result -- it decides how
    many islands there are and where they are cut -- but it lives in ``extra``
    rather than in :data:`PARAM_SPACE`, so the generic field mutator cannot see
    it.  Without this, branching from a chart recipe produced ten children that
    differed only in margin and iteration count and therefore all scored within
    a fraction of a point of the parent.

    Args:
        extra: The parent's extras, already copied.
        method: The child's method.
        rng: Seeded RNG.
        strength: Mutation magnitude in ``[0, 1]``.

    Returns:
        ``(extra, holds_chart)``; ``holds_chart`` is ``True`` when the child
        should be pinned to ``seam_strategy="EXISTING"`` so the engine's cut is
        the one that reaches the solver.
    """
    if method not in _SOLVER_METHODS:
        # Carrying a dormant request would make two identical projections hash
        # differently and waste a slot on a duplicate.
        extra.pop("chart_cut", None)
        extra.pop("chart_seed", None)
        return extra, False

    current = extra.get("chart_cut")
    if not current:
        # A branch that can never reach a chart cut cannot escape a bad one.
        if rng.random() < 0.12 + 0.30 * strength:
            pool = _CHART_FILL_CLUSTERS
            extra["chart_cut"] = int(pool[rng.randrange(len(pool))])
            extra["chart_seed"] = rng.randrange(1 << 16)
            return extra, True
        extra.pop("chart_seed", None)
        return extra, False

    try:
        clusters = int(current)
    except (TypeError, ValueError):
        clusters = 6
    if rng.random() < 0.45 + 0.45 * strength:
        span = max(1, int(round(1.0 + strength * 6.0)))
        clusters += rng.randint(-span, span)
    extra["chart_cut"] = int(min(48, max(2, clusters)))
    if rng.random() < 0.35 + 0.45 * strength:
        extra["chart_seed"] = rng.randrange(1 << 16)
    return extra, True


def _mutate_once(parent: UnwrapParams, rng: random.Random, strength: float) -> UnwrapParams:
    """Apply one round of random perturbation to ``parent``."""
    method = parent.method
    # A strong mutation may jump to a different solver entirely.
    if rng.random() < 0.10 + 0.25 * strength:
        method = rng.choice([m for m in METHODS if m != method] or list(METHODS))

    extra, holds_chart = _mutate_chart(
        dict(parent.extra) if isinstance(parent.extra, dict) else {},
        method,
        rng,
        strength,
    )

    names = [n for n in PARAM_SPACE if n != "method" and _applies(n, method)]
    if holds_chart:
        # The engine supplies the seams; a strategy that recomputes them would
        # throw the chart cut away.
        names = [n for n in names if n != "seam_strategy"]
    if not names:
        return parent.replace(method=method, seed=rng.randrange(1 << 30), extra=extra)

    weights = [PARAM_SPACE[n].weight for n in names]
    # Perturb between one field and (roughly) half of them, scaled by strength.
    n_changes = 1 + int(round(strength * max(1, len(names) - 1) * 0.5))
    n_changes = min(n_changes, len(names))

    chosen: list[str] = []
    pool = list(names)
    pool_w = list(weights)
    for _ in range(n_changes):
        if not pool:
            break
        pick = rng.choices(range(len(pool)), weights=pool_w, k=1)[0]
        chosen.append(pool.pop(pick))
        pool_w.pop(pick)

    values: dict[str, Any] = {
        "method": method,
        "seed": rng.randrange(1 << 30),
        "extra": extra,
    }
    if holds_chart:
        values["seam_strategy"] = "EXISTING"
    for name in chosen:
        spec = PARAM_SPACE[name]
        current = getattr(parent, name)
        values[name] = _perturb(spec, current, rng, strength)
    return parent.replace(**values)


def _perturb(spec: ParamSpec, current: Any, rng: random.Random, strength: float) -> Any:
    """Return a perturbed value for one field."""
    if spec.kind == "bool":
        # Flip with a probability that rises with strength.
        return (not bool(current)) if rng.random() < 0.35 + 0.5 * strength else bool(current)
    if spec.kind == "enum":
        others = [c for c in spec.choices if c != current]
        if not others:
            return current
        return rng.choice(others)
    span = float(spec.high) - float(spec.low)
    if spec.log:
        lo = max(float(spec.low), 1e-4)
        hi = max(float(spec.high), lo * 1.0001)
        cur = min(max(float(current) if float(current) > 0 else lo, lo), hi)
        step = rng.gauss(0.0, 0.9 * strength + 0.05)
        value = math.exp(math.log(cur) + step * (math.log(hi) - math.log(lo)) * 0.5)
        # A log-scaled field may legitimately return to exactly zero.
        if float(spec.low) == 0.0 and rng.random() < 0.08 * strength:
            value = 0.0
        return spec.clamp(value)
    value = float(current) + rng.gauss(0.0, span * (0.08 + 0.35 * strength))
    if spec.kind == "int":
        return spec.clamp(int(round(value)))
    return spec.clamp(value)
