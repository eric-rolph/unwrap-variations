# Benchmarks

> Measured **2026-08-07**, median of 5 runs, verified identical on Blender
> **4.5.4 LTS** and **5.1.2** -- every cell, both versions. Regeneration
> instructions are in [section 6](#6-regenerating-this-table).

This file is the single source of truth for every performance and quality
number the project claims. The README quotes only the headline record and two
rows from here; nothing else in the repository should carry a benchmark
figure of its own.

---

## 1. What is actually being claimed

The claim is narrow on purpose:

> **Generating N candidate unwraps and ranking them automatically produces a
> better layout than running one unwrapper once.**

That is it. Specifically **not** claimed:

- Not that this beats [xatlas](https://github.com/jpcy/xatlas) or RizomUV on
  raw distortion. It does not, and it could not: the solvers doing the work
  here are Blender's own. Measured on the same corpus with the same scorer,
  xatlas 0.0.11 is genuinely better than Blender native on distortion --
  Suzanne L2 **1.0077** against Blender's **1.0298**, torus **1.0037** against
  **1.0296**, hard-surface test mesh **1.0006** against **1.0119**. Where this
  add-on differs is that it searches, measures and ranks; a single unwrapper,
  however good, does not.
- Not that it is fast in absolute terms. Ten candidates cost roughly ten
  unwraps plus ten evaluations. The claim is that the *iteration loop* is
  faster than doing the same exploration by hand, not that the arithmetic is
  cheap.

## 2. Method

| | |
| --- | --- |
| Blender versions | 4.5.4 LTS and 5.1.2, headless, `--factory-startup` |
| Platform | Windows |
| Corpus | Seven Blender primitives, built in-script (no asset download) |
| Scorer | `core.metrics.score()` with `ScoreWeights.preset("balanced")` |
| Candidates | 10 per mesh, `seed=1` |
| Repeats | 5 per column -- see [section 3](#3-blenders-uv-operators-are-not-deterministic) for why this is mandatory |
| Script | [`tools/benchmark_bestofn.py`](../tools/benchmark_bestofn.py) |

**Every cell is identical on 4.5.4 and 5.1.2**, verified under the repeated
methodology below. That cross-version agreement is itself a result: the ranking
a user sees does not depend on which supported Blender they run.

### 2.1 A note on timing

**There is no single "generation takes N seconds" figure, and quoting one would
be misleading.** Generation cost is dominated by Blender's packer, whose cost
runs *inverse* to island count -- a layout with few large islands is far more
expensive to pack than one with many small ones. So the same candidate count on
two meshes of similar size can differ by orders of magnitude.

What can be stated, because it was measured: a packing pathology on small
meshes took a **12-triangle cube from 11.6s to 0.013s** and a **62-triangle
cone from 31.6s to 0.017s** once fixed. Suzanne, at 968 triangles, takes
**6.6s** and keeps full-quality packing because it sits above the small-mesh
threshold where the workaround applies. Generation cost is monotonic in
candidate count.

The three columns are:

| Column | What it runs |
| --- | --- |
| `smart_project` | `bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.0)` -- Blender's own defaults (66 degrees) |
| `unwrap+sharp` | Sharp edges selected at 30 degrees and marked as seams, then `bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)` |
| `best-of-10` | `engine.generate(count=10, seed=1)`, top-ranked candidate |

All three are measured by the same evaluator on the same mesh, so the
comparison is apples-to-apples. Seam length is read back *from the resulting
layout* rather than from the mesh's seam flags, because `smart_project` invents
its own charts without marking anything -- asking the mesh would report zero
seam length for a heavily cut result.

The corpus is deliberately made of primitives: they are reproducible on any
machine with no download, and between them they cover the topologies that break
unwrappers -- closed genus 0 (sphere, icosphere), genus 1 (torus), capped tubes
(cylinder, cone), pure hard-surface (cube), and one organic mesh with real
creases and ears (Suzanne).

## 3. Blender's UV operators are not deterministic

This has to come before the results, because it changes what a result means.

**Running the same Blender UV operator twice on the same mesh does not give you
the same UVs.** Verified on 4.5.4, five runs each against a freshly rebuilt,
identical 48x24 UV sphere, comparing a hash of the resulting UV buffer:

| Operator | Distinct results in 5 runs | Magnitude |
| --- | --- | --- |
| `smart_project` (angle 66, margin 0.01) | 5 of 5 | -- |
| `smart_project` (angle 30, margin 0.02) | 5 of 5 | UV sum 4248.46 to 4383.83, about **3%** |
| `unwrap(ANGLE_BASED)` | 5 of 5 | 1e-6 level |
| `cube_project` | 5 of 5 | 1e-6 level |

`cube_project` is the diagnostic one: it is a pure analytical projection with no
solver in it at all, and it *still* differs between runs. That identifies the
small variation as float and threading noise. `smart_project`'s 3% is a
different order of thing -- that is a genuinely different chart packing each
time.

This is Blender's behaviour, not a defect in this add-on, and not something the
add-on can fix. The engine reproduces the same *recipes* from a seed
faithfully; it cannot make Blender's solvers reproduce the same *results* from
a recipe.

Two consequences:

1. **For this benchmark:** a single run is a sample, not a measurement. Any
   margin smaller than the run-to-run spread is not a win. Every column below
   is therefore measured five times and reported as a median with its observed
   range.
2. **For users:** pressing Generate twice at the same seed produces slightly
   different candidates and slightly different scores. This is documented in
   the README, because a user who noticed it would reasonably assume the seed
   was broken.

## 4. Results

Score is 0-100, higher is better. Each cell is the **median of 5 runs**, with
the observed `[min-max]` range. A verdict is awarded only when the median
margin exceeds **both** the full combined spread of the two series **and** an
absolute floor of 1.0 point. Everything else is a tie.

That is a deliberately conservative rule for a benchmark whose whole purpose is
to be believed: if our best-of-N result and the baseline could have swapped
places on a different roll of Blender's non-deterministic packer, it is a tie.

<!-- BEGIN BENCHMARK TABLE - regenerate wholesale, never hand-edit a cell -->

`balanced` preset, `seed=1`, 10 candidates, 5 repeats. **Identical on Blender
4.5.4 LTS and 5.1.2 -- every cell, both versions.**

| mesh | `smart_project` | `unwrap+sharp` | **best-of-10** | verdict | grid filled |
| --- | ---: | ---: | ---: | --- | ---: |
| cylinder | 36.9 [36.9-36.9] | 0.0 [0.0-0.0] | **72.1** [72.1-72.1] | **WIN +35.3** | 10/10 |
| torus | 29.7 [29.7-29.7] | 33.2 [33.2-33.2] `*` | **50.1** [50.1-50.1] | **WIN +16.8** | 10/10 |
| suzanne | 9.4 [9.4-9.4] | 8.8 [8.8-8.8] | **23.2** [23.2-23.2] | **WIN +13.8** | 10/10 |
| sphere | 57.3 [57.3-57.3] | 45.5 [45.5-45.5] | **66.4** [66.4-66.4] | **WIN +9.1** | 10/10 |
| icosphere | 50.4 [50.4-50.4] | 63.2 [63.2-63.2] | **66.0** [66.0-66.0] | **WIN +2.8** | 10/10 |
| cone | 60.0 [60.0-60.0] | 73.7 [73.7-73.7] | 74.7 [74.7-74.7] | tie +0.9 | 10/10 |
| cube | 70.3 [70.3-70.3] | 70.3 [70.3-70.3] | 70.9 [70.9-70.9] | tie +0.6 | 10/10 |

**Record: 5 wins, 2 ties, 0 losses.** Run-to-run spread was **0.1 points or
less on every cell**, so every verdict here clears the spread rule by a wide
margin and rests on the 1.0-point floor instead.

`*` The `unwrap+sharp` baseline **did nothing measurable** on this mesh -- see
[section 4.2](#42-three-things-not-to-read-into-this-table), caveat 3. The flag
is conservative and under-reports: the sphere and icosphere rows are affected
too.

<!-- END BENCHMARK TABLE -->

### 4.1 What changed since the provisional table

An earlier provisional version of this table reported **3 wins, 3 ties, 1
loss**, with an outright loss on the icosphere and only 7-8 of 10 candidate
slots filled on closed meshes. Both problems are fixed. Stating the changes
plainly, rather than quietly swapping the table:

**The icosphere loss is gone, and every mesh now fills 10/10 slots.** The root
cause was a dead code path. `repair_recipe` -- the escalation that rewrites a
recipe which provably cannot work on a given mesh -- called
`topo.boundary_mask()`, but `boundary_mask` is a `@property` returning an
array, not a method. The call raised `TypeError` straight into a bare
`except Exception`, which swallowed it. **The entire repair path had never
fired once.** On a UV sphere that was silently wasting six slots in ten.
Alongside the fix, a chart-clustering cut was added (see
[section 4.4](#44-how-a-dead-recipe-gets-repaired)) which guarantees every
component is opened into disks.

**Absolute scores are NOT comparable with the provisional table.** The scoring
weights were retuned -- seam 0.6 to 1.8, overlap 2.0 to 1.0, packing 0.8 to
0.7 -- and because all three columns share one scorer, both baseline columns
moved too, exactly as they should. Suzanne's `smart_project` baseline reading
of 13.0 became 9.4 for that reason and not because Blender got worse. Compare
verdicts across the two tables if you like; do not compare cell values.

**The winning layouts got dramatically simpler.** Island counts of the
top-ranked candidate fell from `[276, 6, 11, 6, 6, 5, 24]` to
`[14, 3, 6, 4, 1, 3, 4]`. The 276-island Suzanne winner -- a layout that scored
well on distortion by shattering the mesh into confetti, and which nobody
would have accepted -- is now 14 islands. Raising the seam weight is what did
it, and it is the clearest evidence in this document that the weights, not just
the search, decide what "best" means.

### 4.2 Three things not to read into this table

The record is 5 wins, 2 ties, 0 losses. Three caveats limit what that means,
and none of them is a footnote.

**1. Do not quote the cube or the cone as wins.** They are +0.6 and +0.9, both
below the 1.0-point floor, and both are recorded as ties. A separate agent's
harness scored this corpus **7 wins, 0 ties, 0 losses**; the number published
here is the smaller one, from the stricter rule. The cube in particular is
effectively a coin flip: every dual arc on a cube ties, so which of the eleven
cross-net unfoldings you get is decided by the seed.

**2. Suzanne scores 23.2 out of 100. That is not a good unwrap.** It beats both
baselines by 13.8 points and it is 14 islands rather than 276, so the *relative*
result is real. But nobody should look at 23/100 and call the absolute result
acceptable. The residual is a high seam ratio plus a small genuine self-fold.
An organic mesh with ears, creases and a closed hollow interior is the hardest
thing in this corpus and the add-on does not yet solve it well.

**3. The `unwrap+sharp` column is not a real unwrap on smooth closed meshes.**
This is the caveat that most affects how the table should be read. A 30-degree
sharpness threshold selects **no edges at all** on a sphere, torus or
icosphere. `uv.unwrap` therefore receives no cuts, fails to solve, and -- per
the [section 0 behaviour](ARCHITECTURE.md) -- still reports `FINISHED` while
leaving the existing layout alone. Measured stretch L2 before and after:

| mesh | L2 before | L2 after |
| --- | ---: | ---: |
| sphere | 1.1767 | 1.1767 |
| torus | 1.5230 | 1.5230 |
| icosphere | 1.0090 | 1.0090 |

Unchanged, to four decimal places. What that column actually contains on those
three rows is **the primitive's own stock UV map**. For the icosphere that is
the shipped icosahedron-net layout, which is a genuinely good piece of work --
so the icosphere row is us beating `primitive_ico_sphere_add`'s built-in UVs,
not beating a single-pass unwrap. That cuts both ways: it overstates the
baseline on the icosphere and understates it on the torus.

The script flags these cells with `*`. Detection compares the measured *score*
rather than the raw UV bytes, because a failed unwrap can still leave the
layout flipped or rescaled by packing while every distortion metric stays
invariant. The flag is deliberately conservative and currently catches only the
torus; the sphere and icosphere differ in packing-related metrics after a
rescale and slip past it.

### 4.3 The most instructive row: the cylinder that cannot be unwrapped

`unwrap+sharp` scores **0.0** on the cylinder, with L2 stretch of **30.6**.
That is not a harness bug. Measured on Blender 4.5.4, on a 32-sided cylinder
with 96 edges:

- `edges_select_sharp(sharpness=30 degrees)` marks **64 seams**;
- **all 64 are horizontal rim edges** -- the two 32-edge circles where the caps
  meet the side;
- **zero seams run lengthwise**; nothing ever opens the tube.

The caps come off cleanly. The side is left as a *closed tube*, which is
topologically an annulus, not a disk, so no valid parameterisation exists
without one further cut along its length. ABF produces a catastrophically
stretched result -- 3 islands, L2 30.6049, Linf 248.7764, 12 flipped triangles,
overlap area 6.6e-03 -- and reports `{'FINISHED'}` anyway. That is the same
"`uv.unwrap` lies about success" behaviour recorded in
[ARCHITECTURE.md section 0](ARCHITECTURE.md).

Blender did exactly what it was asked to do. The lesson is not that the
operator is broken; it is that **"mark the sharp edges" is not a general
strategy**. This corpus catches it failing in two entirely different ways:

| Failure mode | Where | What happens |
| --- | --- | --- |
| Selects the **wrong** edges | cylinder, cone | 64 rim seams, none lengthwise. The tube is never opened, and the solver produces L2 30.6 while reporting success |
| Selects **no** edges | sphere, torus, icosphere | Nothing is cut, the solver fails outright, and the stock UV map is left in place unchanged -- see [caveat 3](#42-three-things-not-to-read-into-this-table) |

Neither failure announces itself. In both cases `uv.unwrap` returns
`{'FINISHED'}`. Trying several strategies and measuring the result is how you
catch that, and it is the clearest single argument for this add-on that the
corpus produces.

Two strategies exist for this case specifically. `DUAL_GRAPH` cuts the
*complement* of a spanning forest of the face dual graph, so by construction it
cannot leave a component un-openable. The engine's chart cut goes further and
proves disk-ness outright -- next section.

### 4.4 How a dead recipe gets repaired

Some recipes provably cannot work on a given mesh, and the honest thing to do
with a wasted candidate slot is to notice and fix it rather than fill it with a
failure. Two cases are decidable before any solver runs:

- **A closed surface with an empty seam set.** No boundary means no disk to
  flatten. The solver gives up and, because Blender reports success anyway,
  the slot is wasted silently. On a UV sphere this was six slots in ten.
- **`EXISTING` on a mesh with no marked seams.** The recipe's whole premise --
  "respect what the artist marked" -- is void.

Both escalate to a **chart cut**, which is what raised every mesh in the table
to 10/10 filled slots:

1. **Cluster the polygon normals.** Area-weighted k-means seeded from a
   Fibonacci lattice of directions -- Cohen-Steiner's variational shape
   approximation with the proxy fixed to a plane normal. It lands cuts on the
   creases of a hard-surface model and on the "cube map" boundaries of a smooth
   one.
2. **Cut between clusters, and absorb the slivers.** The boundary of a face
   partition is a set of closed loops, which is why this cut is short where a
   spanning-tree cut is long. One cluster can land in several disconnected
   places, so charts below a fraction of the mean area are merged back into
   their largest neighbour. Measured on Suzanne at 3 clusters: 52 islands
   scoring 15.6 became 8 islands.
3. **Bisect until every chart is a disk.** Clustering cannot guarantee
   disk-ness -- the top half of a torus is an annulus, and a solver handed an
   annulus either fails or folds it over itself (measured overlap 7.9e-03 on
   the icosphere). So each chart's Euler characteristic is computed exactly
   from three bincounts, `chi = F - U + Vint`, and any chart with `chi != 1` is
   split along its principal axis and retested. An annulus becomes two disks in
   one round.

The cluster count and seed rotation are derived from the recipe's own
signature, so several recipes escalating in the same run produce genuinely
different charts rather than collapsing onto one layout and being
de-duplicated away.

### 4.5 Overlap is a metric because baselines really do produce it

On Suzanne, the `smart_project` baseline leaves **42 overlapping triangle
pairs**. Overlapping UVs silently corrupt baked normal and light maps, so
overlap area is one of the axes candidates are ranked on. It carries a flat
step penalty *on top of* a proportional one, which is what makes it a hard
defect rather than merely an expensive one: a layout with any overlap at all
drops a full multiplicative notch below an otherwise-equal clean layout, no
matter how good the rest of it looks.

For contrast, xatlas produced **zero** overlaps everywhere on the same corpus.

## 5. Competitive context (measured, not quoted)

From the same corpus, scored by the same reference implementation. This is
context for where the add-on sits, not a claim about it.

| engine | Suzanne L2 | torus L2 | hard-surface L2 | overlaps |
| --- | ---: | ---: | ---: | --- |
| Blender `smart_project` | 1.0298 | 1.0296 | 1.0119 | 42 pairs on Suzanne |
| xatlas 0.0.11 | **1.0077** | **1.0037** | **1.0006** | 0 everywhere |

Blender native does win on the sphere (1.0237 against xatlas's 1.0346), and it
is far better integrated.

**xatlas 0.0.11 segfaulted** on a 130k-triangle sphere on Windows / CPython
3.13, with valid in-range indices -- a hard crash, not a Python exception. This
is stated as a factual observation about why this add-on isolates its backends
and treats a backend failure as a skipped candidate: a backend that dies must
take the candidate down, never Blender. It is not a criticism of xatlas, which
is an excellent library and beats Blender's own solver on distortion above.

## 6. Regenerating this table

The scoring weights, the seam strategies and the archetype list all feed every
column. **Regenerate the whole table; never patch a cell.** Mixing baselines
from one weighting with our results from another produces a table that is not
merely stale but wrong in a way that flatters us.

```bash
blender -b --factory-startup --python tools/benchmark_bestofn.py
```

On Windows, with an explicit path:

```bash
"C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup --python tools/benchmark_bestofn.py
```

Options go after a bare `--` so Blender does not try to parse them:

```bash
blender -b --factory-startup --python tools/benchmark_bestofn.py -- --count 20 --seed 7
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--count` | 10 | Candidates generated per mesh |
| `--seed` | 1 | Determinism seed |
| `--preset` | `balanced` | `ScoreWeights` preset applied to **all three** columns |
| `--no-recipes` | off | Skip the per-mesh winning-recipe report |

The script exits non-zero if any mesh produced no candidates at all.

Besides the table it prints two things worth reading:

- **Timings** per mesh: generation, and each baseline, so the cost of the
  search is visible next to the cost of a single pass.
- **The winning recipe** per mesh and the score spread across all candidates.
  If the same archetype always wins, the parameter search is theatre -- this is
  the check for that.

Run it on both supported Blender versions before publishing. The scores should
match; if they do not, that is a bug and it is more interesting than the
benchmark.
