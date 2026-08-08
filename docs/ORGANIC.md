# Organic meshes: the binding contract

`docs/ARCHITECTURE.md` is the contract for the add-on as a whole and still
governs everything it covers. **This file is the contract for the organic-mesh
work and overrides ARCHITECTURE.md where they disagree.** Read section 0 before
writing a line of code: it is measured ground truth, not design intent, and
several of the facts in it invalidate the obvious approach.

---

## 0. Measured ground truth

Everything in this section was measured on **Blender 5.1.2**, `--factory-startup`,
against `bpy.ops.mesh.primitive_monkey_add()` with `engine.generate(count=10,
seed=1)` and the shipped `balanced` scorer. Reproduce with
`tools/diag_organic.py`. Do not take any of it on trust; re-measure if you
change the pipeline.

### 0.1 Suzanne's actual topology

Suzanne is **three connected shells**, not one, and the head is **not** a
closed sphere:

| shell | F | E | V | chi | boundary edges | non-manifold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 856 | 1297 | 441 | **0** | 26 | 0 |
| eye L | 56 | 88 | 33 | 1 | 8 | 0 |
| eye R | 56 | 88 | 33 | 1 | 8 | 0 |

Counts are over the **triangulated** mesh (968 tris). `sample.n_edges` reports
1005 because that is the quad mesh's edge count; the 468 triangulation
diagonals are not in it. Any code that reasons about cut length must use the
triangulated edge set, which is what the evaluator already does.

`chi = 0` with 26 boundary edges means the head is **genus 0 with two boundary
loops** -- the eye sockets. It is not a torus and it is not closed. The two
eyeballs are already disks (`chi = 1`).

**Consequence that drives the whole design:** the head needs exactly **one**
cut, a path joining its two boundary loops, to become a disk. The eyes need
none. The topological minimum seam length for Suzanne is on the order of one
edge path. We currently spend **31.6**. Every unit of that above the minimum is
a distortion trade we chose, not a topological necessity -- which means it is
negotiable, and that is the opening this work exploits.

### 0.2 What the 23.2 is actually made of

Best candidate, `balanced` preset, penalty `P = 1.4615`, `score = 100*exp(-P)`:

| term | penalty | share | removing it alone |
| --- | ---: | ---: | ---: |
| **seam** | 0.6448 | 44.1% | **+21.0 pts** |
| **overlap** | 0.6096 | 41.7% | **+19.5 pts** |
| angle | 0.0834 | 5.7% | +2.0 pts |
| packing | 0.0819 | 5.6% | +2.0 pts |
| area | 0.0320 | 2.2% | +0.8 pts |
| stretch | 0.0099 | 0.7% | +0.2 pts |

Measured layout: 14 islands, `seam_length` 31.613, `seam_ratio` 8.956,
`stretch_l2` 1.0164, `stretch_linf` 3.1844, `packing_efficiency` 0.6579,
`uv_coverage` 0.5618, `overlap_area` 0.0013421, **`flipped_tris` 0**,
`degenerate_tris` 0, total 3D area 12.4609 (`sqrt` = 3.5300).

**Two terms are 85.8% of the loss. The other four are worth 5.0 points
combined.** Do not spend effort on distortion: `stretch_l2` of 1.0164 is
already close to isometric and is worth 0.2 points. This work is about seams
and folds, and anything that trades seam length for distortion is very likely
a good trade at these numbers.

Ceilings, holding everything else fixed:

| fix | score becomes |
| --- | ---: |
| overlap only | 42.7 |
| seam only | 44.2 |
| **overlap + seam** | **81.3** |
| overlap + seam + packing | 88.2 |

Seam targets, assuming overlap is fixed:

| target score | required `seam_ratio` | required `seam_length` |
| --- | ---: | ---: |
| 40 | 9.85 | 34.8 (already met) |
| 50 | 6.75 | 23.8 (**-25%**) |
| 60 | 4.22 | 14.9 (**-53%**) |

### 0.3 The overlap is a global injectivity failure, not a flip

This is the single most important fact in this document, because it rules out
the obvious fix.

- `flipped_tris` is **0**. All 968 signed UV areas are **positive**. The map is
  **locally injective everywhere**.
- The overlap is nonetheless real: an independent exact convex-clip
  implementation, cross-checked per pair against a 200k-sample Monte-Carlo area
  estimate, reproduces the evaluator's figure to ten significant digits
  (`0.0013420951`). **The evaluator is not lying and must not be "fixed".**
- 16 triangle pairs overlap. **Zero of them share an edge.** 12 pairs are fully
  disjoint, 4 share a single vertex. So it is not a hinge folding at a crease;
  it is one region of a chart lapping over a distant region of the same chart.
- Location in 3D: bbox `x in [-0.737, 0.995]`, `y in [0.253, 0.414]`,
  `z in [0.312, 0.495]` -- Suzanne's **brow ridge**, on both sides. Not the
  ears.

So the chart boundary self-intersects while every triangle keeps its
orientation. A flip-only check (`signed area < 0`) **cannot see this defect**,
and a flip-only repair cannot fix it. Certification has to test the boundary
curve for self-intersection, and repair has to be a genuine global-injectivity
method.

Standard result this rests on, and which the design uses as its certificate:
**a locally injective piecewise-linear map of a disk whose boundary image is a
simple closed curve is globally injective.** Checking (no flips) AND (boundary
is simple) is therefore a complete proof of injectivity for a disk chart, not a
heuristic.

### 0.4 All of the seam length is mergeable

Measured over the winning layout, classifying every triangulated interior edge
that is split in UV:

- measured cut length: **31.613** (matches `seam_length` exactly)
- mesh boundary length, i.e. the eye sockets, which are free edges and not our
  doing: 5.573 (correctly excluded from the metric)
- cut length lying **between two different islands: 31.613 -- 100.0%**
- cut length lying **inside one island** (a slit that opens it into a disk):
  **0.000**

Every single cut edge is a chart boundary. There are no slits. So **the entire
seam budget is reachable by merging adjacent charts**, subject only to keeping
each merged chart a disk and its distortion acceptable. The largest shared
boundaries:

| island pair | shared length |
| --- | ---: |
| 10-12 | 3.419 |
| 5-11 | 2.485 |
| 10-13 | 2.239 |
| 5-10 | 2.192 |
| 11-13 | 2.146 |
| 5-13 | 1.998 |

Merging just the four largest islands `{5, 10, 11, 13}` (8.7 of the 12.46 total
area) removes 11.06 units of seam, taking `seam_ratio` from 8.956 to **5.83** --
past the score-50 threshold on its own, if distortion holds.

### 0.5 There is no scipy

Blender 5.1.2 ships **Python 3.13.9 with numpy 2.3.4 and nothing else**.
`scipy` is **not available** and must not be imported, made an optional
dependency, or vendored. Any linear solve -- and the flattener needs one -- has
to be written against numpy alone. `xatlas` is likewise absent from Blender's
interpreter; it stays a CLI-only optional backend.

This is a hard constraint, not a preference. A module that imports scipy will
pass every test you run in your own CPython and fail for every user.

### 0.6 Rules inherited from ARCHITECTURE.md that bite here

Re-stated because they have already cost this project real bugs:

- `bpy.ops.uv.unwrap` returns `{'FINISHED'}` when it fails. Never trust a
  return status; check the buffer.
- Blender's UV operators are **non-deterministic**. A single benchmark run is a
  sample, not a measurement. Medians over repeats with a noise band, always.
- A mesh holds at most **8 UV layers**; candidates are numpy arrays, not layers.
- Nothing under `core/` may import `bpy`. The new modules in this contract are
  all `core/` and all bpy-free, which is what makes them testable in plain
  CPython at speed.
- Keep source and console output **ASCII-only** (Windows cp1252).

---

## 1. Scope and the two rules

Deliver: Suzanne, and organic meshes generally, unwrapped materially better,
with the improvement measured honestly.

**Rule 1 -- do not touch the scorer.** `core/metrics.py` weights, scales and the
`_HARD_STEP` structure are **frozen for this work**. It is trivially easy to
turn 23.2 into 60 by softening the overlap step, and that is scoring theatre:
the artist's bake is just as broken. The docstring in `metrics.py` speculating
that the step "should be gated on overlap exceeding a small fraction of
coverage" is **not** authority to do it here. If you believe the scorer is
wrong, say so in your report and leave the code alone; it will be argued
separately, on its own merits, against both the old and new numbers.

**Rule 2 -- no stage may make a candidate worse.** Every refinement stage
measures before and after and keeps the better layout. A stage that cannot
improve a layout must return it untouched. This makes the whole chain safe to
enable by default and makes any regression a bug rather than a trade-off.

---

## 2. Where this plugs in

One hook, in `core/engine.py::_run_recipes`, between the `dead_layout_reason`
check (line ~916, where `previous_uvs = uvs` is assigned) and
`derive_uv_seams` (line ~919):

```
backend -> uvs -> [ refine(sample, uvs, params) ] -> derive_uv_seams -> evaluate -> score
```

`refine` is the new chain and is the **only** engine change. Everything else in
this contract is a new `core/` or `backends/` module that `refine` calls. The
chain is:

```
1. charts.merge_charts     reduce seam length          (section 5)
2. flatten.repair_folds    remove overlap / flips      (section 4)
3. repack                  reclaim packing efficiency  (section 6)
```

Merging runs first because a merged chart is a different chart and has to be
certified after the merge, not before.

---

## 3. `core/linalg.py` -- numpy-only sparse solving

Owner: worker 1.

The flattener needs to solve `L x = b` with `L` a sparse symmetric
positive-definite Laplacian, up to ~500k unknowns, with no scipy.

```python
class SparseSym:
    """Symmetric sparse matrix in COO triplets, applied as an operator."""
    def __init__(self, rows, cols, vals, n): ...
    def matvec(self, x: np.ndarray) -> np.ndarray: ...   # np.bincount based
    def diagonal(self) -> np.ndarray: ...

def pcg(A: SparseSym, b, *, x0=None, tol=1e-10, maxiter=None,
        precond="jacobi") -> tuple[np.ndarray, int, float]:
    """Jacobi-preconditioned conjugate gradient. Returns (x, iters, residual)."""
```

Requirements:

- `matvec` must be vectorised (`np.bincount` / `np.add.at` on the triplets), not
  a Python loop. Budget: a 100k-vertex cotan Laplacian matvec in **under 10 ms**.
- `pcg` must handle multiple right-hand sides (`b` shaped `(n,)` or `(n, k)`) so
  `u` and `v` solve together.
- It must **never raise** on a singular or indefinite system: detect breakdown
  (`p^T A p <= 0`, non-finite residual) and return the best iterate with the
  achieved residual so the caller can decide. A silent NaN here becomes a NaN
  UV layer and a corrupted `.blend`.
- Deterministic: identical input gives bit-identical output.

Unit tests: against dense `np.linalg.solve` on random SPD systems; on a known
Laplacian with an analytic solution; on a deliberately singular system;
on a system with a zero row; and on a 200k-unknown system for timing.

## 4. `core/flatten.py` and `core/injectivity.py` -- injective flattening

Owner: worker 2 (flatten), worker 3 (injectivity + oracle).

### 4.1 `core/injectivity.py` -- certification

```python
@dataclass(frozen=True)
class InjectivityReport:
    flipped: int                  # triangles with sign opposite the chart majority
    degenerate: int
    boundary_crossings: int       # self-intersections of the chart boundary
    overlap_area: float           # 0.0 when clean
    locally_injective: bool       # flipped == 0 and degenerate == 0
    globally_injective: bool      # locally_injective and boundary_crossings == 0
    charts_checked: int

def certify(sample, *, tri_chart=None) -> InjectivityReport: ...
def boundary_loops(sample, tri_chart) -> list[list[int]]: ...
def boundary_crossings(uv, loops) -> int: ...
```

- `boundary_crossings` is a segment-segment intersection count over each chart's
  boundary polyline, **excluding** shared endpoints of consecutive segments.
  Use a sweep or a uniform grid; `O(n^2)` over a 50k-vertex boundary is not
  acceptable. Budget: 100k boundary segments in **under 500 ms**.
- Exact predicates matter more than speed here: use the `orient2d` sign test,
  and treat exactly-touching as **not** crossing (consistent with the
  evaluator's overlap convention, section 0.3).
- This module is the runtime certificate. **A second, deliberately naive
  `O(n^2)` implementation lives in `tests/reference.py`** and the two must agree
  on every corpus mesh. Last time an independent reference caught more than all
  the self-testing combined; this is not optional.

### 4.2 `core/flatten.py` -- the flattener

```python
def tutte_embed(chart) -> np.ndarray:
    """Guaranteed flip-free initial UV: convex boundary + mean-value weights."""

def minimize_distortion(uv, chart, *, iters=32, energy="symmetric_dirichlet",
                        free_boundary=True) -> np.ndarray:
    """Local-global descent that provably never flips a triangle."""

def flatten_chart(chart, *, target=None) -> FlattenResult:
    """tutte_embed then minimize_distortion, with a certificate attached."""

def repair_folds(sample, uvs, *, tri_chart=None) -> tuple[np.ndarray, RepairReport]:
    """Re-flatten only the charts that fail certification. Never worsens."""
```

Design, and why:

- **Tutte / Floater gives the feasible start.** A disk triangulation mapped to a
  convex boundary with positive weights has a provably crossing-free embedding.
  Use **mean-value weights** (Floater 2003), which are positive for any
  triangulation -- cotangent weights go negative on obtuse triangles and forfeit
  the guarantee. This start is heavily distorted; that is fine, it is a
  starting point, not an answer.
- **Then descend without ever flipping.** Local-global on the symmetric
  Dirichlet energy `sum A_i * (||J_i||_F^2 + ||J_i^-1||_F^2)`, with a
  **maximum-step line search**: for each triangle compute the largest `t` such
  that `uv + t*d` keeps its signed area positive (a quadratic in `t`), take
  `0.8 * min_i t_i` as the step cap, then backtrack on the energy. A step that
  cannot flip any triangle cannot break local injectivity, so **the invariant
  holds by construction from the Tutte start onward** -- this is the whole
  reason for starting at Tutte rather than at the backend's layout.
- **Free the boundary, then check it.** Fixed-boundary results are far too
  distorted to ship. Run with a free boundary and then certify; per section 0.3,
  (no flips) + (simple boundary) is a complete proof. If the boundary does
  self-intersect, fall back in this order: (a) more descent iterations with a
  smaller step cap, (b) re-run with the boundary constrained to its convex hull
  ordering, (c) return the Tutte embedding, which is always valid. **Never
  return a layout that fails certification** -- return a worse-but-valid one and
  say so in the report.
- `repair_folds` is the engine-facing entry. It certifies the incoming layout
  per chart, leaves clean charts **bit-identical**, re-flattens only the failing
  ones, rescales each repaired chart to preserve its original UV area, and
  compares the result. Per Rule 2, if the repair does not reduce
  `overlap_area`, return the input unchanged.

Acceptance for this pair: on Suzanne's winning layout, `overlap_area` becomes
**exactly 0.0** with `flipped_tris` still 0, and the score reaches **at least
40** with no other change and no scorer change.

## 5. `core/charts.py` -- distortion-bounded chart merging

Owner: worker 4.

```python
def chart_graph(sample, tri_chart) -> ChartGraph:
    """Adjacency, shared boundary length and per-chart area/topology."""

def is_disk(sample, tri_mask) -> bool:
    """Exact Euler test: chi = F - E + V == 1 over the triangulated submesh."""

def merge_charts(sample, tri_chart, *, max_stretch=1.25, min_gain=0.0,
                 flatten=None) -> tuple[np.ndarray, MergeReport]:
    """Greedily merge adjacent charts while each merge stays a disk and under
    the distortion bound. Monotonically non-increasing in seam length."""
```

- Greedy by **benefit ratio**: `shared_boundary_length / max(added_distortion,
  eps)`. Recompute after each accepted merge (a merge changes its neighbours'
  ratios). A priority queue with lazy invalidation, not a full re-sort.
- **Every candidate merge must be validated by actually flattening the merged
  chart** and measuring, using `flatten.flatten_chart`. Estimating distortion
  from the two parents is exactly the shortcut that produces charts nobody would
  accept -- section 0.1 shows the topological argument permits merging the whole
  head, and only a real measurement stops it.
- Reject a merge if the union is not a disk (`is_disk` false), if measured
  `stretch_l2` exceeds `max_stretch`, or if the flattened union fails
  injectivity certification.
- `max_stretch` is the artist-facing dial and becomes a `UnwrapParams` field so
  the grid can sample it -- some candidates merge aggressively, some not at all.
  **The unmerged layout must remain reachable in the grid** so the benchmark can
  attribute the gain.

Acceptance: on Suzanne, `seam_ratio` drops from 8.956 to **at most 6.75**
(score-50 threshold) with `stretch_l2` staying under 1.25 and injectivity
holding, verified by the independent oracle.

## 6. Repacking

Owner: worker 4, jointly with section 5.

Merging changes island shapes, so the layout must be repacked or the packing
term eats part of the gain. Reuse the existing Blender packing path in
`backends/blender_native.py` -- including the `uv.select_all(action='SELECT')`
fix and the small-mesh `shape_method='AABB'` substitution, both of which are
already there and both of which are load-bearing. Do **not** write a new packer.
Per Rule 2, keep the repack only if the score improves.

## 7. `backends/seams.py` -- concavity-aware cuts

Owner: worker 5.

The existing strategies are `SHARP`, `DUAL_GRAPH`, `EXISTING` and the chart-cut
escalation. On smooth organic meshes `SHARP` selects nothing (measured: zero
edges at 30 degrees on a sphere, torus, icosphere) and normal-clustering lands
cuts on "cube map" boundaries, which are arbitrary on a face.

Add a `CONCAVE` strategy. Artists hide seams in concave valleys, and a valley is
also where the surface is least developable, so a cut there costs the least
distortion. Weight each edge by its length scaled down where the dihedral is
concave, then:

1. Compute the signed dihedral per edge (`edge_dihedral_deg` exists already;
   confirm its sign convention by measurement before relying on it).
2. Take the minimum spanning structure needed to make each shell a disk -- for
   Suzanne's head, one shortest concavity-weighted path between its two boundary
   loops (section 0.1); for a closed shell, a path joining two curvature
   extremes.
3. Add further cuts **only** where the distortion measurement demands them.

Acceptance: on Suzanne the `CONCAVE` cuts must lie predominantly in negative
dihedral regions (report the fraction), every resulting chart must be a disk by
the exact Euler test, and the strategy must win at least one slot in the grid on
at least one organic corpus mesh.

## 8. Test methods for organic meshes

Owner: worker 6. **This is a deliverable in its own right, not test scaffolding
for the others.** The user asked for testing methods explicitly.

### 8.1 `tools/organic_corpus.py` -- the corpus

Deterministic, procedural, **no downloads**, built in-script so any user can
regenerate it. Blender primitives are not organic and Suzanne alone is not a
corpus. Required meshes, each with a stated reason it is hard:

| mesh | why it is hard |
| --- | --- |
| `suzanne` | creases, ears, two boundary loops, three shells -- the reference case |
| `suzanne_sub2` | 15.5k tris; same shape, tests scaling |
| `blob_limbs` | metaball creature, 5 protrusions -- tests chart-per-limb decomposition |
| `displaced_sphere` | noise-displaced, closed genus 0 -- no sharp edges anywhere |
| `torus_knot` | genus 1, organic cross-section -- topology plus curvature |
| `hand` | 5 close-together protrusions -- tests cross-island collision |
| `folded_ribbon` | deliberately induces a boundary self-intersection |
| `pinched` | zero-area and near-degenerate faces -- robustness, must not crash |

Each entry returns a mesh plus expected invariants (shell count, chi per shell)
so a corpus regression is caught at build time.

### 8.2 Oracles

- **Injectivity oracle** (`tests/reference.py`): naive `O(n^2)` boundary
  self-intersection and all-pairs triangle overlap. Slow and obviously correct.
  Must agree with `core/injectivity.py` on every corpus mesh.
- **Topology oracle**: independent Euler characteristic and shell count from
  triangle soup, as used to produce section 0.1.
- **Metric oracle**: the existing `tests/reference.py` naive stretch
  implementation, extended to cover the new fields.

### 8.3 Visual regression -- `tools/render_uv_report.py`

Critics must be able to **look** at the result. For each corpus mesh and each of
{baseline, best-of-N}: render the UV layout to PNG with islands coloured, folds
outlined in red, and seams drawn; render the mesh with a checker texture from
three fixed camera angles. Write an index HTML. Determinism: fixed seed, fixed
camera, EEVEE, fixed resolution.

### 8.4 `tools/benchmark_organic.py`

Corpus x {`smart_project`, `unwrap+sharp`, `xatlas` where available, best-of-N
**before** this work, best-of-N **after**}. Median of >= 5 repeats with the
observed range, the same verdict rule the existing benchmark uses (margin must
exceed both the combined spread and a 1.0-point floor), on **both** Blender
4.5.4 and 5.1.2. The "before" column is mandatory: without it there is no
evidence any of this helped.

---

## 9. Definition of done

1. Suzanne's best candidate scores **>= 60** under the **unchanged** `balanced`
   scorer, on both 4.5.4 and 5.1.2, median of 5.
2. `overlap_area == 0.0` and `flipped_tris == 0` for the top-ranked candidate on
   **every** corpus mesh, confirmed by the independent oracle.
3. No regression on the existing seven-mesh benchmark: the record stays at least
   5 wins / 2 ties / 0 losses and no cell drops by more than the noise band.
4. The full existing test suite passes (534 tests at the time of writing), plus
   new tests for every module above. `ruff check .` clean.
5. `blender --command extension validate` clean, and the built extension
   installs and runs on both versions.
6. Generation time for 10 candidates on Suzanne stays under **30 s** on the
   reference machine (currently 5.6 s; the budget allows the new work to cost
   several times the unwrap itself, but not unboundedly).
7. Every number published in `docs/BENCHMARKS.md` regenerated wholesale, never
   hand-patched.
