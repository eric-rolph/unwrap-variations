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
| **`pack_islands` headless** | Returns `CANCELLED` on **4.5** but `FINISHED` on **5.1**. Must be wrapped and failure-tolerant. |
| `numpy` inside Blender | present (1.26.4 on 4.5) |
| `xatlas` inside Blender | **absent** — must be optional/subprocess |
| ID-property blobs | `mesh["key"] = bytes(...)` round-trips as `bytes` |
| `gpu.shader.from_builtin` | available (overlay is feasible) |

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
- `DUAL_GRAPH` — build the face dual graph weighted by coplanarity
  `w = 1 - clamp(n1·n2)`, take a **maximum spanning tree**, and cut the mesh
  edges whose dual edge is *not* in the tree, then prune the resulting seam set
  to remove dangling ends. Yields few, well-placed cuts on a connected chart.
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

## 12. Naming, style, tests

- `UVVAR_` prefix on every registered class (`UVVAR_OT_`, `UVVAR_PT_`, `UVVAR_PG_`).
- Python 3.11+, PEP 8, 4-space indent, ~100 col, full docstrings, type hints.
- ASCII-only in source and CLI output (Windows cp1252 console).
- `tests/` runs under plain pytest with **no** Blender; anything needing `bpy`
  lives in `tools/run_blender_tests.py` and is invoked headlessly.
