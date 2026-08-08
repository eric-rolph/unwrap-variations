# UnwrapVariations — Architecture & Module Contract

This document is **binding**. Every module in this repository implements the
signatures below exactly. If you believe a signature is wrong, say so — do not
silently change it, because other modules are written against it in parallel.

---

## 0. Verified Blender ground truth

These facts were probed empirically against **Blender 4.5.4 LTS** and
**Blender 5.1.2** (headless, `--factory-startup`). Do not contradict them.

| Fact | Value |
| --- | --- |
| **UV layer hard cap** | **8 per mesh.** `uv_layers.new()` returns `None` past that. A 10-candidate grid *cannot* be stored as 10 UV layers. |
| `uv.unwrap(method=...)` | `ANGLE_BASED`, `CONFORMAL`, `MINIMUM_STRETCH` (identical in 4.5 + 5.1) |
| `uv.unwrap` other args | `fill_holes`, `correct_aspect`, `use_subsurf_data`, `margin_method`, `margin`, `no_flip`, `iterations`, `use_weights`, `weight_group`, `weight_factor` |
| `uv.smart_project` args | `angle_limit` (**RADIANS**, default 1.15192 = 66°), `margin_method`, `rotate_method`, `island_margin`, `area_weight`, `correct_aspect`, `scale_to_bounds` |
| `uv.pack_islands` args | `udim_source`, `rotate`, `rotate_method`, `scale`, `merge_overlap`, `margin_method`, `margin`, `pin`, `pin_method`, `shape_method` |
| **`pack_islands` headless** | Returns `CANCELLED` on **4.5**, `FINISHED` on **5.1**. **Root cause found:** these operators act on *selected* UVs. 4.5 defaults `tool_settings.use_uv_select_sync = False` and a fresh unwrap leaves every UV deselected, so there is nothing to pack. 5.1 flipped that default to `True`. **Fix is one line: `bpy.ops.uv.select_all(action='SELECT')` before packing** -- then FINISHED on both. Affects `average_islands_scale` and `minimize_stretch` identically. |
| `numpy` inside Blender | present (1.26.4 on 4.5) |
| `xatlas` inside Blender | **absent** — must be optional/subprocess |
| ID-property blobs | `mesh["key"] = bytes(...)` round-trips as `bytes` **on the first write only** -- see the truncation row below. |
| `gpu.shader.from_builtin` | available (overlay is feasible) |

### Further probed facts (added as workers verified them)

| Fact | Detail |
| --- | --- |
| **`uv.unwrap` lies about success** | Returns `{'FINISHED'}` even when it fails. On a closed mesh with no seams it prints `Unwrap failed to solve 1 of 1 island(s)` and **leaves the previous UVs untouched**. Never treat the operator return value as success -- detect a degenerate or unchanged UV buffer instead (`engine.dead_layout_reason`). A dead slot silently inherits its predecessor's layout. |
| **4.5 cancels more than packing** | Headless on 4.5, `pack_islands`, `average_islands_scale` **and** `minimize_stretch` all return `CANCELLED`; all three return `FINISHED` on 5.1. |
| **UV operators are NON-DETERMINISTIC** | Verified on 4.5.4: 5 runs of `smart_project` on an identical fresh sphere gave **5 distinct UV buffers**. `smart_project` varies by ~3% (genuinely different chart packing, threaded); `unwrap` and even `cube_project` -- a pure analytical projection with no solver -- vary at the 1e-6 level. Consequences: (a) a seed reproduces the same *recipes*, never bit-identical *results*; (b) any benchmark must repeat and report medians with a noise band, because a single run is a sample not a measurement; (c) a margin smaller than the run-to-run spread is not a win. |
| **Re-assigning a bytes ID property TRUNCATES it** | Writing over an existing `bytes` ID property silently truncates the new value at its **first NUL byte**. Verified identically on 4.5.4 and 5.1.2: a 314-byte zlib blob came back as 309. A compressed payload is full of NULs, so the *first* Generate persists correctly and every subsequent one shreds the store -- a bug shaped to pass every single-run test. **Always `del`/`pop` the key before writing it.** |
| **`bl_info` is deleted before `register()`** | For extensions, `addon_utils` runs `del mod.bl_info` before calling `register()`, so reading `bl_info` inside `register()` raises `NameError` for every installed user while a source checkout works fine. Also, `bl_info` must contain **only literals** -- it is parsed from source text with `ast.literal_eval` and never imported, so referencing a constant yields `ValueError: malformed node`. |
| **Registered subclasses are not on `bpy.types`** | On 4.5.4, registered `PropertyGroup`/`AddonPreferences` subclasses are not reachable via `bpy.types.<Name>` or `dir(bpy.types)` (they are on 5.1.2). Use `Base.bl_rna_get_subclass_py(name)`, or a leak assertion silently passes. |
| **Packing dominates, and its cost is INVERSE to mesh size** | `uv.pack_islands` is ~99.8% of generation time, and it gets *slower* as island count falls: one cube at 6 faces = 8.46 s, 24 faces = 0.34 s, 96 faces = 0.15 s. Mitigated by substituting `shape_method='AABB'` below 256 triangles (cube 11.6 s -> 0.013 s, cone 31.6 s -> 0.017 s), which also *raised* mean packing coverage, because the exhaustive search finds nothing better on so few islands. No rotate_method constant is safe: `AXIS_ALIGNED_X` totals 1.13 s across a suite where `ANY` totals 26.12 s, yet is the worst option (7.43 s vs 0.85 s) on one cube recipe. |
| **Do not trust a `CANCELLED` operator's timing** | An earlier measurement here read "every `uv.*` op is sub-millisecond" and concluded all cost was our own overhead. That was wrong: on 4.5 `pack_islands` was returning CANCELLED, i.e. timing an operator that did nothing. Always check the return status before drawing a conclusion from a timing. |
| **`temp_override(area=...)` is impossible headless** | Raises `TypeError: Area not found in screen` -- the background window's screen holds no areas. Do not retry this route. |
| **`uv.unwrap` has `margin`, not `island_margin`** | And `cube_project` has no margin argument at all (`cube_size`, `correct_aspect`, `clip_to_bounds`, `scale_to_bounds`), so island margin reaches it only via packing. |
| **A FLOAT2 CORNER attribute consumes a UV slot** | It counts against the 8-layer cap and appears in `mesh.uv_layers`. `free_uv_layer_slots()` must account for it. |
| **Candidates are NOT comparable across Blender versions** | With a bit-identical mesh, bit-identical seam mask and the same call, `uv.unwrap` produces *different* UVs on 4.5.4 vs 5.1.2 for multi-island meshes (maxdiff 0.97, before any packing). With no seams, and for projections, it is bit-identical. Persisted candidates should record the Blender version that produced them. |
| **Only operator-signature delta across versions** | `pack_islands.udim_source` gains `CUSTOM_REGION` in 5.1. Everything else is identical. |
| **`gpu.shader.from_builtin` names** | The `3D_`/`2D_` prefixed names (`'3D_SMOOTH_COLOR'`, `'2D_IMAGE'`) **do not exist** on 4.5 or 5.1 and raise `ValueError`. Valid on both: `FLAT_COLOR`, `IMAGE`, `IMAGE_COLOR`, `SMOOTH_COLOR`, `UNIFORM_COLOR`, `POLYLINE_*`, `POINT_*`. Resolve through a preference list, not a version branch. |
| **`layout.progress()` exists on both** | But `hasattr(bpy.types.UILayout, "progress")` returns **False** -- `dir()` does not list RNA functions. Probe with `"progress" in bpy.types.UILayout.bl_rna.functions`. |
| **GPU draw is unavailable headless** | `SystemError: GPU functions for drawing are not available in background mode`. Headless overlay tests must assert graceful degradation, not drawing. |
| **Edit-mode meshes read empty** | In Edit Mode the original datablock reports a stale pre-edit triangle count and the evaluated object's `.data` has zero triangles. Only `evaluated.to_mesh()` returns real geometry. |
| **ID-prop blobs survive save/reload** | Verified byte-identical at 1 KB, 1 MB, 4 MB, 16 MB and 64 MB on both versions, with a real `save_as_mainfile` + `open_mainfile`. |
| **`layer.uv`/`"vector"` is 25-50x faster** | Than the legacy `layer.data`/`"uv"` alias (0.27 ms vs 22 ms at 1M loops). Not a style preference. |
| **Reads that allocate** | Reading `layer.pin` *allocates* `.pn.<uv>` on 4.5, corrupting an in-progress `uv_layers` iteration. Writing all-false `hide` flags allocates `.hide_vert/.hide_edge/.hide_poly` on 5.1. Read through `mesh.attributes` instead. |
| **Stale layer pointers** | Cached `MeshUVLoopLayer` pointers go stale across a removal on 4.5. Always re-resolve by name. |
| **UV sidecar split** | 4.5 has `MeshUVLoopLayer.vertex_selection`/`.edge_selection`; **5.1 removed both** (moved to mesh-level `.uv_select_vert`/`_edge`/`_face`). `layer.pin` reports length 0 on 5.1 until allocated, where `foreach_get` raises `TypeError`. |

### Verified fast data paths (all `foreach_get`-able)

```python
me.vertices.foreach_get("co", buf)            # float32, V*3
me.loops.foreach_get("vertex_index", buf)     # int32,  L
me.loop_triangles.foreach_get("loops", buf)   # int32,  T*3  (loop indices)
me.loop_triangles.foreach_get("vertices", buf) # int32, T*3  (vertex indices)
me.edges.foreach_get("vertices", buf)         # int32,  E*2
me.edges.foreach_get("use_seam", buf)         # bool,   E
uv_layer.uv.foreach_get("vector", buf)        # float32, L*2   <- preferred
uv_layer.data.foreach_get("uv", buf)          # float32, L*2   <- legacy alias
```

`me.calc_loop_triangles()` must be called before reading `loop_triangles`.

---

## 1. Design decisions (rationale)

1. **Candidate UVs are raw float arrays, not UV layers.** Because of the 8-layer
   cap, a candidate is `numpy.ndarray` shape `(L, 2)` float32. Exactly **one**
   scratch UV layer (`UVVar_Preview`) exists on the mesh; previewing a candidate
   is a single `foreach_set` — O(L) memcpy, instant even at 1M loops.
2. **The core is `bpy`-free.** `core/params.py`, `core/meshdata.py`,
   `core/metrics.py`, `core/evaluator.py`, and `backends/seams.py`'s graph
   algorithms import only stdlib + numpy. This makes them unit-testable in plain
   CPython and is why `tests/` runs without Blender.
3. **Non-destructive by default.** The user's original UV layers are never
   touched until an explicit Apply. Generation snapshots and restores mesh state
   (seams included).
4. **Every failure is contained.** A backend that raises must degrade to a
   skipped candidate, never a Blender crash and never a half-written mesh.

---

## 2. `core/meshdata.py` — the bpy-free mesh interchange (WRITTEN, do not change)

```python
@dataclass(frozen=True)
class UVMeshSample:
    verts:      np.ndarray  # (V, 3) float64 — object-space positions
    tri_verts:  np.ndarray  # (T, 3) int32   — vertex index per triangle corner
    tri_loops:  np.ndarray  # (T, 3) int32   — loop index per triangle corner
    uvs:        np.ndarray  # (L, 2) float64 — UV per loop
    edge_verts: np.ndarray  # (E, 2) int32
    edge_seam:  np.ndarray  # (E,)   bool
    loop_verts: np.ndarray  # (L,)   int32   — vertex index per loop
    name: str = ""

    @property n_verts / n_tris / n_loops / n_edges -> int
    def validate(self) -> list[str]   # non-fatal problem strings
    def with_uvs(self, uvs) -> UVMeshSample
```

Helper (bpy-free, used by evaluator and tests):

```python
def tri_areas_3d(s: UVMeshSample) -> np.ndarray      # (T,)
def tri_areas_uv(s: UVMeshSample) -> np.ndarray      # (T,) SIGNED
def degenerate_mask(s: UVMeshSample, eps=1e-12) -> np.ndarray  # (T,) bool
#   eps is RELATIVE: the floor is eps * mean(triangle area), never an absolute
#   value. An absolute 1e-12 flags every triangle of a 0.1 mm CAD import as
#   degenerate. evaluate() must use the same relative rule.
```

---

## 3. `core/params.py` — parameter space (WRITTEN, do not change)

```python
@dataclass(frozen=True)
class UnwrapParams:
    backend: str = "blender_native"
    method: str = "ANGLE_BASED"      # ANGLE_BASED|CONFORMAL|MINIMUM_STRETCH|SMART_PROJECT|CUBE_PROJECT
    angle_limit_deg: float = 66.0    # DEGREES here; converted to rad at the bpy edge
    island_margin: float = 0.01
    area_weight: float = 0.0
    correct_aspect: bool = True
    scale_to_bounds: bool = False
    fill_holes: bool = True
    no_flip: bool = False
    iterations: int = 10
    margin_method: str = "SCALED"    # SCALED|ADD|FRACTION
    rotate_method: str = "AXIS_ALIGNED_Y"
    pack_rotate: bool = True
    pack_shape_method: str = "CONCAVE"   # CONCAVE|CONVEX|AABB
    seam_strategy: str = "SHARP"     # see §5
    seam_angle_deg: float = 30.0
    curvature_threshold: float = 0.35
    minimize_stretch_iters: int = 0
    average_islands_scale: bool = False
    seed: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def signature(self) -> str                  # stable sha1 over normalised fields, 12 hex chars
    def to_dict(self) -> dict                   # JSON-safe
    @classmethod from_dict(cls, d) -> UnwrapParams   # tolerant of unknown/missing keys
    def replace(self, **kw) -> UnwrapParams
    def describe(self) -> str                   # short human label for the UI, <= 40 chars
```

`PARAM_SPACE: dict[str, ParamSpec]` describes each mutable field
(`kind` = `float|int|bool|enum`, plus `low/high/step` or `choices`, and a
`weight` for mutation probability). `ParamSpec` is a frozen dataclass.

```python
def sample_grid(n: int, *, seed: int, base: UnwrapParams | None = None) -> list[UnwrapParams]
    """N maximally-distinct candidates. Deterministic for a given seed."""

def mutate(parent: UnwrapParams, *, n: int, strength: float, seed: int) -> list[UnwrapParams]
    """N perturbations around `parent`. `strength` in [0,1]."""
```

Both **must** return de-duplicated params (unique `signature()`).

---

## 4. `core/metrics.py` + `core/evaluator.py` — scoring (bpy-free)

```python
@dataclass(frozen=True)
class UVMetrics:
    # distortion
    angle_distortion: float      # mean abs angle deviation, DEGREES, area-weighted
    angle_distortion_max: float
    area_distortion: float       # mean |log2(uv_area_norm / tri_area_norm)|
    stretch_l2: float            # Sander et al. 2001 L2 stretch, >= 1.0 ideal 1.0
    stretch_linf: float          # worst-case singular-value stretch
    conformal_error: float       # mean sigma_max/sigma_min - 1
    # topology / layout
    island_count: int
    seam_length: float           # world units
    seam_ratio: float            # seam_length / sqrt(total 3D area)  (scale-free)
    packing_efficiency: float    # used UV area / occupied AABB area, 0..1
    uv_coverage: float           # used UV area / 1.0 (unit square)
    overlap_area: float          # 0.0 when clean
    flipped_tris: int
    degenerate_tris: int
    texel_density_cv: float      # coeff. of variation of sqrt(uv/3d) per triangle
    uv_bounds: tuple[float, float, float, float]
    out_of_bounds_tris: int
    # bookkeeping
    n_tris: int
    eval_seconds: float

    def to_dict(self) / from_dict(cls, d)

@dataclass(frozen=True)
class ScoreWeights:
    angle: float = 1.0
    area: float = 1.0
    stretch: float = 1.2
    seam: float = 0.6
    packing: float = 0.8
    overlap: float = 2.0
    flipped: float = 2.0
    @classmethod def preset(cls, name: str) -> "ScoreWeights"
        # "balanced" | "low_distortion" | "few_seams" | "tight_packing" | "game_ready"

def score(m: UVMetrics, w: ScoreWeights) -> float
    """0..100, higher is better. MUST be monotonic: strictly better metrics
    never lower the score. Overlap and flipped faces apply hard penalties."""

def pareto_front(ms: Sequence[UVMetrics]) -> list[int]
    """Indices of non-dominated candidates over (stretch, seam_ratio, packing)."""
```

```python
# core/evaluator.py
def evaluate(sample: UVMeshSample, *, compute_overlap: bool = True) -> UVMetrics
```

Hard requirements for `evaluate`:
- **Never raises** on degenerate input (zero-area tris, NaN UVs, empty mesh).
  Degenerate triangles are excluded from the distortion means and reported in
  `degenerate_tris`.
- Fully vectorised with numpy. Target: **< 250 ms for 500k triangles**
  (excluding overlap), and overlap detection must use a uniform grid / spatial
  hash, never O(T²).
- Deterministic — same input, same floats.
- `stretch_l2` / `stretch_linf` follow Sander et al., *Texture Mapping
  Progressive Meshes* (SIGGRAPH 2001), §3, using the per-triangle Jacobian
  singular values, area-weighted, and normalised so a perfectly isometric
  unwrap scores exactly 1.0.

---

## 5. `backends/seams.py` — seam heuristics

```python
SEAM_STRATEGIES = ("NONE", "EXISTING", "SHARP", "ANGLE", "CURVATURE",
                   "DUAL_GRAPH", "SMART")

def compute_seams(sample: UVMeshSample, strategy: str, *,
                  angle_deg: float = 30.0,
                  curvature_threshold: float = 0.35,
                  existing: np.ndarray | None = None,
                  seed: int = 0) -> np.ndarray   # (E,) bool
```

- `ANGLE` — dihedral angle between adjacent face normals exceeds `angle_deg`.
- `CURVATURE` — normalised discrete mean curvature per edge over threshold.
- `DUAL_GRAPH` -- build the face dual graph weighted by **coplanarity**
  `c = clamp(dot(n1, n2), -1, 1)`, take a **maximum spanning forest** over `c`,
  cut the mesh edges whose dual edge is *not* in the forest, then prune
  dangling ends. Yields few, well-placed cuts.

  > **Corrected.** This originally read "weighted by coplanarity
  > `w = 1 - clamp(dot)`, take a maximum spanning tree". That is
  > self-contradictory: `1 - dot` is a *sharpness*, so maximising it keeps the
  > sharpest adjacencies in the tree and cuts across flat regions -- the exact
  > opposite of the stated intent. Maximise coplanarity `c` directly.
  > **Forest**, not tree, because a mesh may have several disconnected shells.
  >
  > The prune rule as first written was also wrong. On a *closed* component the
  > complement of a dual spanning forest is exactly a spanning tree of that
  > component's vertices, and a tree is nothing but dangling ends -- so pruning
  > to a fixpoint deletes the entire cut and leaves a surface with no disk
  > parameterisation at all. A cube would return 0 cuts instead of 7. The
  > implementation therefore protects a dangling cut sharper than
  > `curvature_threshold`, and restores a closed component's raw cut wholesale
  > if the peel empties it. Open components prune normally.
- Must handle **non-manifold** edges (>2 incident faces) and boundary edges
  (1 face) without raising — boundary edges are never seams.
- Pure numpy/stdlib. No `bpy`.

---

## 6. `backends/base.py` — the backend ABC

```python
@dataclass
class UnwrapRequest:
    obj_name: str
    params: UnwrapParams
    sample: UVMeshSample          # pre-extracted, for bpy-free backends
    timeout_s: float = 60.0

@dataclass
class UnwrapResult:
    uvs: np.ndarray | None        # (L, 2) float32/64 — None on failure
    ok: bool
    message: str = ""
    seconds: float = 0.0
    backend: str = ""

class UnwrapBackend(ABC):
    id: ClassVar[str]
    label: ClassVar[str]
    requires_bpy: ClassVar[bool] = True

    @classmethod def is_available(cls) -> tuple[bool, str]: ...
    @abstractmethod def unwrap(self, req: UnwrapRequest) -> UnwrapResult: ...
    def supported_methods(self) -> tuple[str, ...]: ...

def register_backend(cls) -> type       # decorator
def get_backend(backend_id: str) -> UnwrapBackend
def available_backends() -> list[tuple[str, str, bool, str]]   # id, label, ok, why
```

A backend **never** raises out of `unwrap()`; it returns `ok=False` with a
message. It must leave the Blender scene exactly as it found it.

---

## 7. `core/candidate.py` — storage

```python
@dataclass
class Candidate:
    index: int
    params: UnwrapParams
    uvs: np.ndarray | None        # (L, 2) float32
    metrics: UVMetrics | None
    score: float = 0.0
    ok: bool = True
    message: str = ""
    parent: str | None = None     # parent signature for branch lineage
    generation: int = 0
    seconds: float = 0.0
    @property def signature(self) -> str

class CandidateStore:
    """Per-object candidate set. Runtime numpy cache + optional persistence to
    an ID-property blob so candidates survive save/load."""
    PREVIEW_LAYER = "UVVar_Preview"
    ID_KEY = "uvvar_store_v1"

    def __init__(self, obj_name: str, loop_count: int)
    @classmethod def for_object(cls, obj) -> "CandidateStore"    # bpy edge
    def set(self, cands: Sequence[Candidate]) -> None
    def add(self, c: Candidate) -> None
    def get(self, index: int) -> Candidate | None
    def all(self) -> list[Candidate]
    def clear(self) -> None
    def ranked(self, weights: ScoreWeights) -> list[Candidate]
    def save_to_id(self, obj, *, max_bytes: int) -> tuple[bool, str]
    def load_from_id(self, obj) -> bool
```

Persistence format: zlib-compressed, version-tagged; JSON header (params +
metrics) followed by the float32 UV payload. If total size exceeds `max_bytes`
(addon preference, default 64 MB), persist metadata only and report it — never
silently bloat the .blend.

**Store invalidation:** if `len(mesh.loops)` no longer matches the stored loop
count, the store is stale — discard it and tell the user.

---

## 8. `core/bridge.py` — the only bpy⇄core boundary

```python
def extract_sample(obj, *, uv_layer_name: str | None = None) -> UVMeshSample
def apply_uvs(obj, uvs: np.ndarray, *, layer_name: str) -> None
def ensure_preview_layer(obj) -> str            # creates/reuses UVVar_Preview
def remove_preview_layer(obj) -> None
def snapshot_state(obj) -> dict                 # UV layers + seams + mode + active layer
def restore_state(obj, snap: dict) -> None
def free_uv_layer_slots(obj) -> int             # 8 - len(uv_layers)
```

`extract_sample` uses only the verified `foreach_get` fast paths and must handle
a mesh with **zero** UV layers (returns zeroed `uvs`). All of these must work
from OBJECT mode and restore the mode they found.

---

## 9. `core/engine.py` — the coordinator

```python
@dataclass
class GenerationReport:
    candidates: list[Candidate]
    seconds: float
    failures: list[tuple[UnwrapParams, str]]
    warnings: list[str]

def generate(obj, *, count: int, seed: int, weights: ScoreWeights,
             base: UnwrapParams | None = None,
             backends: Sequence[str] | None = None,
             progress: Callable[[float, str], None] | None = None,
             compute_overlap: bool = True) -> GenerationReport

def branch(obj, parent: Candidate, *, count: int, strength: float, seed: int,
           weights: ScoreWeights,
           progress: ... , ...) -> GenerationReport
```

Engine responsibilities, in order:
1. Validate the object (mesh, has polygons, has loops) → clear error if not.
2. `snapshot_state`.
3. Extract one `UVMeshSample`, reused for every candidate (extract once, not N times).
4. For each param set: apply seams → run backend → read UVs back → evaluate → score.
5. `restore_state` **in a `finally`**, always.
6. Rank, assign indices, store.

Generation must be **interruptible** and report progress; a failure in one
candidate never aborts the run.

---

## 10. Operators (`bl_idname` is contract)

| Operator | `bl_idname` | Purpose |
| --- | --- | --- |
| Generate grid | `uvvar.generate_grid` | N candidates from the param space |
| Branch | `uvvar.branch_variations` | N mutations of the active candidate |
| Preview | `uvvar.select_candidate` | blit candidate → preview layer |
| Apply | `uvvar.apply_candidate` | write candidate into a real UV map |
| Clear | `uvvar.clear_candidates` | drop store + preview layer |
| Overlay toggle | `uvvar.toggle_overlay` | stretch/checker viewport overlay |
| Cycle | `uvvar.cycle_candidate` | next/prev preview (keymap-friendly) |
| Overlay control | `uvvar.overlay_control` | set overlay mode/opacity (owned by `ui/stretch_overlay.py`) |
| Checker material | `uvvar.checker_material` | apply/restore the procedural checker material |

`uvvar.toggle_overlay` is the contract entry point and is implemented by
`ui.stretch_overlay.sync_from_props(context)`; `uvvar.overlay_control` and
`uvvar.checker_material` are the finer-grained operators behind it. All three
are reserved in the `uvvar.` namespace -- do not redefine them elsewhere. Call
`ui.stretch_overlay.invalidate()` after blitting a preview so the overlay
rebuilds its batch.

All operators: `bl_options = {"REGISTER", "UNDO"}` where mutating, correct
`poll()`, and `self.report({"ERROR"}, ...)` + `return {"CANCELLED"}` instead of
raising.

## 11. Properties

`bpy.types.Scene.uvvar` → `UVVAR_SceneProps` holding `candidate_count`,
`seed`, `mutation_strength`, `weight_preset`, `active_index`, `backend_filter`,
`overlay_mode`, `auto_preview`, `compute_overlap`, plus a
`CollectionProperty` of `UVVAR_CandidateItem` (index, label, score, and the
individual metric floats) that mirrors the store for UI drawing. The UI reads
**only** the CollectionProperty, never the numpy store.

## 11b. Runtime module name and the bpy-free precondition

Two cross-worker rules that are easy to violate silently. Both are contract.

**The installed package is not called `unwrap_variations`.** Installed as a
4.2+ extension it is `bl_ext.user_default.unwrap_variations`. Therefore:

- Every intra-package import **must be relative** (`from ..core import params`).
  An absolute `import unwrap_variations.core.params` inside the add-on raises at
  runtime once installed, even though it works fine from a source checkout.
- `__init__.py` must key preferences off `__package__` / `__name__`, never the
  literal string `"unwrap_variations"`. `AddonPreferences.bl_idname` must be
  `__package__`.
- The manifest `id` **must** equal the package directory name.

**`__init__.py` must not import `bpy` at module top level.** Importing any core
module (`import unwrap_variations.core.params`) executes the package
`__init__.py` first. A top-level `import bpy` there breaks plain-CPython pytest
and voids the bpy-free guarantee in section 1.2 for the entire core. Guard it:

```python
try:
    import bpy
except ImportError:      # running outside Blender, e.g. under pytest
    bpy = None
```

or keep every `bpy` touch inside functions. CI enforces this by poisoning
`sys.modules["bpy"]` and importing the core modules.

## 11c. Packaging

The deliverable is a Blender extension archive, not a wheel.

- `unwrap_variations/blender_manifest.toml` lives **next to `__init__.py`**,
  because the package directory *is* the extension source directory.
- Build and validate with Blender itself:
  `blender --command extension validate unwrap_variations` and
  `blender --command extension build --source-dir unwrap_variations`.
- `tags` must come from Blender's controlled 32-entry add-on vocabulary.
  `tagline` and every `[permissions]` description are capped at 64 characters
  and may not end in a period.
- Version lives in **two** places that must move together:
  `blender_manifest.toml` `version` and `pyproject.toml` `version`.

## 12. Naming, style, tests

- `UVVAR_` prefix on every registered class (`UVVAR_OT_`, `UVVAR_PT_`, `UVVAR_PG_`).
- Python 3.11+, PEP 8, 4-space indent, ~100 col, full docstrings, type hints.
- ASCII-only in source and CLI output (Windows cp1252 console).
- `tests/` runs under plain pytest with **no** Blender; anything needing `bpy`
  lives in `tools/run_blender_tests.py` and is invoked headlessly.
