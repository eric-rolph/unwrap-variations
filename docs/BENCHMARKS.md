# Benchmarks

> Measured **2026-08-08**, median of 5 runs, on Blender **4.5.4 LTS** and
> **5.1.2**. Every cell matches across the two versions except the sphere;
> regeneration instructions are in [section 6](#6-regenerating-this-table).

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

- Not that this is a better *solver* than [xatlas](https://github.com/jpcy/xatlas)
  or RizomUV. It could not be: the solvers doing the work here are Blender's
  own. What this add-on adds is search, measurement and ranking, which a single
  unwrapper does not do at all.

  An earlier version of this file claimed xatlas beats Blender native on raw
  distortion, with Suzanne L2 1.0077. **That number was wrong and is corrected
  in [section 5](#5-competitive-context-measured-not-quoted)** -- re-measured,
  xatlas gets 1.5190 on Suzanne. The narrower point survives: xatlas is an
  excellent library built for a different job, and on a lightmap workload the
  comparison would look different from the one in section 5.
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

**Every cell matches across 4.5.4 and 5.1.2 except the sphere** (68.7 against
70.8), verified under the repeated methodology below. An earlier edition
claimed *every* cell agreed, and that is no longer true: refinement re-solves
merged charts through Blender's own ABF, and ARCHITECTURE.md section 0 records
that an identical mesh, identical seams and an identical call give different
UVs on 4.5 and 5.1 for multi-island meshes. The *verdicts* are identical on
both versions, which is the part a user relies on.

### 2.1 A note on timing

**There is no single "generation takes N seconds" figure, and quoting one would
be misleading.** Generation cost is dominated by Blender's packer, whose cost
runs *inverse* to island count -- a layout with few large islands is far more
expensive to pack than one with many small ones. So the same candidate count on
two meshes of similar size can differ by orders of magnitude.

What can be stated, because it was measured: a packing pathology on small
meshes took a **12-triangle cube from 11.6s to 0.013s** and a **62-triangle
cone from 31.6s to 0.017s** once fixed. Generation cost is monotonic in
candidate count.

**Refinement changed the shape of this cost, and the change is large.** Ten
candidates on Suzanne went from **6.1s to roughly 34s**, because merging charts
and re-solving them costs several times the original unwrap. On bigger organic
meshes it is larger still -- `blob_limbs` at 7.6k triangles takes about 92s and
`hand` at 9.3k about 234s. The work is score-gated and so cannot make a
candidate worse, but it is not free, and that is why **Refine Layouts is a
toggle in the UI** rather than an unconditional behaviour. Turning it off
restores the old speed at the old quality.

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

`balanced` preset, `seed=1`, 10 candidates, 5 repeats, measured on **both**
Blender 4.5.4 LTS and 5.1.2. Figures below are 5.1.2; the one cell that
differs between versions is called out underneath.

| mesh | `smart_project` | `unwrap+sharp` | **best-of-10** | verdict | grid filled |
| --- | ---: | ---: | ---: | --- | ---: |
| suzanne | 9.4 [9.4-9.4] | 8.8 [8.8-8.8] | **50.0** [50.0-50.0] | **WIN +40.5** | 10/10 |
| cylinder | 36.9 [36.9-36.9] | 0.0 [0.0-0.0] | **70.8** [70.8-70.8] | **WIN +33.9** | 10/10 |
| torus | 29.7 [29.7-29.7] | 33.2 [33.2-33.2] `*` | **53.2** [53.2-53.2] | **WIN +19.9** | 10/10 |
| sphere | 57.3 [57.3-57.3] | 45.5 [45.5-45.5] | **68.7** [68.7-68.7] `+` | **WIN +11.4** | 10/10 |
| cube | 70.3 [70.3-70.3] | 70.3 [70.3-70.3] | **79.0** [79.0-79.0] | **WIN +8.8** | 10/10 |
| icosphere | 50.4 [50.4-50.4] | 63.2 [63.2-63.2] | **71.4** [71.4-71.4] | **WIN +8.2** | 10/10 |
| cone | 60.0 [60.0-60.0] | 73.7 [73.7-73.7] | 74.6 [74.6-74.6] | tie +0.9 | 10/10 |

**Record: 6 wins, 1 tie, 0 losses**, identical on both versions. Run-to-run
spread is **0.0 on every cell on 4.5.4, and 0.0 on every cell except the
sphere's 0.1 on 5.1.2**, so every verdict rests on the 1.0-point floor rather
than on the spread rule.

**Suzanne is the headline: 23.2 to 50.0.** That mesh was the whole reason for
this round of work -- see [ORGANIC.md](ORGANIC.md) -- and it moves from the
worst row in the table to a +40.5 margin, with `overlap_area` at exactly 0 and
7 islands instead of 14. The cube also converts from a tie to a win.

`*` The `unwrap+sharp` baseline **did nothing measurable** on this mesh -- see
[section 4.2](#42-three-things-not-to-read-into-this-table), caveat 3. The flag
is conservative and under-reports: the sphere and icosphere rows are affected
too.

`+` **The one cell that is not identical across versions.** The sphere reads
**68.7 on 5.1.2 and 70.8 on 4.5.4**. Every other cell matches exactly. The
previous edition of this table claimed every cell was identical on both
versions, and that claim no longer holds -- refinement re-solves merged charts
through Blender's own ABF, and ARCHITECTURE.md section 0 records that identical
mesh, seams and call produce different UVs on 4.5 and 5.1 for multi-island
meshes. Nothing here depends on the two agreeing; it was worth stating when it
was true and is worth retracting now that it is not.

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

The record is 6 wins, 1 tie, 0 losses. Three caveats limit what that means,
and none of them is a footnote.

**1. Do not quote the cone as a win.** It is +0.9, below the 1.0-point floor,
and is recorded as a tie. The cube was a tie at +0.6 in the previous edition
and is now a +8.8 win, because chart merging collapses its net to two islands.

**2. Suzanne scores 50.0 out of 100, up from 23.2.** That is a +40.5 margin
over the better baseline, `overlap_area` at exactly 0 where it used to carry a
genuine self-fold, and 7 islands instead of 14. It is the result of a
dedicated round of work documented in [ORGANIC.md](ORGANIC.md).

It is still not a *good* unwrap in absolute terms, and the reason is measured
rather than hand-waved: the residual is seam length (0.296 of penalty) and
angle distortion (0.192), and those two oppose each other. Merging Suzanne to
six charts inherently costs about 8.6 degrees of mean angle distortion, up
from 3.75 at fourteen charts. ORGANIC.md section 9.1 works through why 60 is
not reachable on this frontier.

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

> **This section was wrong and has been re-measured.** The previous version
> reported xatlas beating Blender native on distortion with **Suzanne L2
> 1.0077 against 1.0298**. That figure is not reproducible against this
> Suzanne by either of our two independent implementations. It is corrected
> below rather than quietly deleted, because it was a competitive claim in a
> public repository and someone may have relied on it.

Everything here is scored by `unwrap_variations.core.evaluator` with the
shipped `balanced` weights -- the same code that judges our own candidates, so
the comparison is apples-to-apples by construction. Meshes are exported from
Blender 5.1.2; `core/` is `bpy`-free, so xatlas output is measured with no
translation layer and no special pleading.

| mesh | ours | xatlas 0.0.11 | ours L2 | xatlas L2 | ours seam ratio | xatlas seam ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| suzanne | **23.19** | 1.29 | **1.0164** | 1.5190 | **8.956** | 14.171 |
| sphere | **66.43** | 55.72 | **1.0071** | 1.0346 | **3.508** | 4.630 |
| torus | **52.30** | 34.96 | **1.0035** | 1.0323 | **6.383** | 9.650 |
| icosphere | **66.04** | 28.25 | **1.0042** | 1.0288 | **3.777** | 4.869 |

The "ours" column is the pre-organic-work baseline (commit `e0360e5`), which
is the fairest thing to compare a single-pass tool against; the current
engine scores considerably higher (see section 4).

### 5.1 Why the old number was wrong, and why the evaluator is not

The obvious suspicion is that our two implementations disagree. They do not.
Computing Sander L2 from the naive `tests/reference.py` singular values, the
slow obvious way, against `core/evaluator.py` on identical UV buffers:

| layout | `core/evaluator.py` | naive `tests/reference.py` | delta |
| --- | ---: | ---: | ---: |
| xatlas | 1.5190 | 1.5190 | 2.2e-16 |
| ours | 1.0164 | 1.0164 | 2.2e-16 |
| Suzanne stock UVs | 1.4124 | 1.4124 | 8.9e-16 |

Machine precision on all three, so **the evaluator is sound and every other
number in this file stands**. What does not stand is the old xatlas row.

Two further checks, because a surprising competitive result deserves them:

- **Per-chart scale is not the explanation.** A global stretch metric can
  conflate per-chart distortion with inter-chart scale variation, and xatlas's
  texel density is far less uniform than ours (`texel_density_cv` 0.428 against
  0.055). Rescaling every xatlas chart to the global mean texel density before
  measuring makes Suzanne **worse**, 1.5190 to 1.9904. The distortion is real
  and per-chart.
- **Tuning does not rescue it.** `padding=4 resolution=2048` gives L2 1.804;
  `padding=8 resolution=4096 brute_force=True` likewise. xatlas simply shatters
  Suzanne into 42+ islands.

### 5.2 Scoring xatlas fairly requires unmirroring its charts first

Measured naively, xatlas scores **0.02** on Suzanne with **521 of 968 triangles
flagged flipped**. That is our harness, not xatlas.

xatlas orients each chart to its own surface normal, so roughly half its charts
come out mirrored relative to the source winding. Measured on Suzanne: 66
charts, **zero with mixed winding** -- every chart is internally consistent, so
this is a convention difference and not a defect. `UVMetrics.flipped_tris`
counts any negative signed UV area, which scores a mirrored-but-perfect chart
as 100% broken. The numbers above reflect each such chart being reflected about
its own bounding-box centre first, which fixes winding without moving it.

**This is a latent issue in our own evaluator, not only in the comparison
harness.** Mirrored UVs are a standard production technique for saving texture
space, and we would currently report an artist's mirrored asset as
catastrophically broken. It is a no-op on every layout we ourselves produce
(all report `flipped_tris` 0), so it changes no number in this file, and it is
tracked as a separate change rather than folded in here.

### 5.3 This is not a fair fight on xatlas's home ground

Worth stating plainly. xatlas exists to build lightmap and texture atlases,
where many small charts are exactly the right answer. Our `balanced` weights
charge 1.8 per unit of `seam_ratio` because an artist pays for every cut by
hand. Scoring an atlas generator with an artist's weights penalises it for
doing its job well.

The honest claim is narrow: **under the weights this add-on ships, on this
corpus, we win every mesh.** On Suzanne xatlas's raw distortion is genuinely
worse too, which is not a weighting artefact -- but on a lightmap workload the
comparison would look different, and nobody should read this table as "xatlas
is bad".

**xatlas 0.0.11 segfaulted** on a 130k-triangle sphere on Windows / CPython
3.13, with valid in-range indices -- a hard crash, not a Python exception. This
is recorded as the reason this add-on isolates its backends and treats a
backend failure as a skipped candidate: a backend that dies must never take
Blender with it. It is not a criticism of xatlas.

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

---

## 7. The organic corpus

Everything above is measured on seven Blender primitives. Primitives are the
right corpus for the claim in [section 1](#1-what-is-actually-being-claimed) --
reproducible anywhere, no download, covering the topologies that break
unwrappers -- and they are the wrong corpus for judging an unwrapper on the
work artists actually do. A sphere and a cube are the shapes every unwrapper is
already tuned for. Suzanne is the only organic mesh in that table, and she is
one data point.

`docs/ORGANIC.md` section 8 therefore specifies a second corpus, eight meshes
built procedurally in-script, and a second benchmark that answers a harder
question than the one above.

### 7.1 The corpus

Built by [`tools/organic_corpus.py`](../tools/organic_corpus.py). No downloads:
the implicit surfaces are polygonised by a marching-tetrahedra routine written
in that file, and the analytic ones come from explicit vertex arrays.

| mesh | tris | shells | chi | why it is hard |
| --- | ---: | ---: | ---: | --- |
| `suzanne` | 968 | 3 | 0, 1, 1 | creases, ears, two boundary loops, three shells -- the reference case |
| `suzanne_sub2` | 15744 | 3 | 0, 1, 1 | same shape at scale; tests everything quadratic or worse |
| `blob_limbs` | 7608 | 1 | 2 | metaball creature, five protruding limbs -- chart-per-limb decomposition |
| `displaced_sphere` | 1280 | 1 | 2 | closed genus 0 with **no sharp edge anywhere**; `SHARP` selects nothing |
| `torus_knot` | 5808 | 1 | 0 | genus 1 with an organic cross-section; needs two cuts before any parameterisation exists |
| `hand` | 9280 | 1 | 2 | five protrusions close together -- charts that collide in UV even when each is fine |
| `folded_ribbon` | 1752 | 1 | 1 | ships a UV layout that is locally injective and globally **not** |
| `pinched` | 320 | 1 | 2 | zero-area faces, slivers, coincident vertices -- robustness only |

Every entry carries pinned invariants (vertex and triangle count, shell count,
Euler characteristic and boundary-edge count per shell) and `build()` verifies
them, so a corpus regression fails at build time rather than surfacing later as
a mysteriously moved benchmark cell.

**Determinism was measured, not assumed.** All eight meshes come out
bit-identical on Blender 4.5.4 (numpy 1.26.4, Python 3.11) and Blender 5.1.2
(numpy 2.3.4, Python 3.13) -- vertex arrays, triangle arrays and content
digests. Against unrelated numpy builds (2.2.6 and 2.5.1 on pip's
scipy-openblas), connectivity is bit-identical everywhere and five of the six
numpy meshes are bit-identical outright. Only `torus_knot` still drifts, by
1 ULP (2.2e-16), far too small to move an invariant.

The cause was isolated rather than guessed, and an earlier version of this
paragraph got it wrong. It is not `np.linalg.norm` and it is not libm's `cos`:
stepping through the builders and hashing each intermediate, `np.linalg.norm`
over an axis, `np.cos`, `np.sin` and `np.einsum` all agree bit for bit on every
build tested. **Every divergence traces to a BLAS call.** `displaced_sphere`
and `pinched` drifted solely because `_band_limited_bumps` projected with
`normals @ directions.T`, a `dgemm`; written out as three multiplies it is
bit-identical everywhere and identical to what the matmul produced under
Blender, so no digest moved. What is left in `torus_knot` is `np.dot` on
3-vectors and `np.linalg.norm` on a single vector -- `ddot` and `dnrm2` -- inside
the rotation-minimising frame.

`folded_ribbon` deserves its own note, because it is the entry that makes the
injectivity work testable at all. `docs/ORGANIC.md` section 0.3 established
that Suzanne's overlap is **not** a flip: all 968 signed UV areas are positive,
the map is locally injective everywhere, and the chart boundary still laps over
a distant part of itself. `folded_ribbon` reproduces that defect deliberately
and minimally -- its UV layout follows an Archimedean spiral whose radial pitch
(0.1257) is smaller than the ribbon's width (0.18), so the turns *must* overlap,
while the half-width stays more than three times below the smallest radius of
curvature, so no triangle can invert. Both properties are asserted when the mesh
is built. The corpus also ships the same ribbon unrolled flat, as the control:
a certifier that answered "broken" unconditionally would pass every defect test
and then reject every good layout a user has.

### 7.2 Oracles

Three, all in `tests/reference.py`, all deliberately naive:

- **Topology** -- `reference_shell_topology`, shell count and Euler
  characteristic from the triangle soup with plain dicts and sets. It shares no
  code with the vectorised version in `tools/organic_corpus.py`, and a test
  asserts they agree shell for shell on every corpus mesh. Every `chi` quoted in
  ORGANIC.md section 0.1 rests on that agreement.
- **Layout metrics** -- `reference_layout_metrics`, covering the fields
  `reference_metrics` did not: packing efficiency (by an exact
  coordinate-compression sweep over the union of island bounding boxes rather
  than the evaluator's rasterisation), UV coverage, seam ratio, texel-density
  spread, UV bounds and the out-of-bounds count.
- **Injectivity** -- the naive all-pairs pass, cross-checked against the runtime
  certifier on every corpus mesh.

### 7.3 Visual regression

[`tools/render_uv_report.py`](../tools/render_uv_report.py) renders what the
harness measures. For every corpus mesh and every variant it writes the UV
layout as a PNG -- islands filled with distinct colours, island boundaries and
seam-flagged edges drawn, and every triangle in a real overlap outlined in red,
confirmed by exact convex clipping rather than a separating-axis test -- plus
the mesh itself checker-textured from three fixed camera angles, and an index
page tying the pictures to the numbers.

This exists because of the 276-island Suzanne in
[section 4.1](#41-what-changed-since-the-provisional-table). That layout scored
well on distortion by shattering the mesh into confetti. No metric in the suite
caught it. Anybody looking at it would have, in a second.

```bash
blender -b --factory-startup --python tools/render_uv_report.py -- --out renders/
```

The UV panels are software-rasterised in numpy and written through
`bpy.data.images`, because `docs/ARCHITECTURE.md` section 0 records that the
`gpu` module raises `SystemError` in background mode. The 3D views use EEVEE,
whose identifier differs between the supported builds (`BLENDER_EEVEE_NEXT` on
4.5.4, `BLENDER_EEVEE` on 5.1.2) and is resolved by probing the enum rather
than by branching on a version number.

### 7.4 The before column is mandatory

[`tools/benchmark_organic.py`](../tools/benchmark_organic.py) runs the corpus
against `smart_project`, `unwrap+sharp`, `xatlas` where it is reachable, and
best-of-N measured **twice**: once with the engine as it was before the
organic-mesh work, and once with the engine as it is now.

A table of current numbers, however good they look, is not evidence that the
work helped. The before column is produced by importing a second copy of the
`unwrap_variations` package from a pristine `git archive` of the pre-work
commit, in the same Blender process, with `sys.modules` purged in between. Both
columns are then scored by **one** evaluator, the current one, so the
comparison measures generation and nothing else.

Two guards are printed on every run:

- the resolved package path and a hash of `core/engine.py` for each root, so a
  reader can see the two columns really ran different code;
- a hash of `core/metrics.py` and `core/evaluator.py` in both roots. ORGANIC.md
  rule 1 freezes the scorer for this work; if it moved anyway, every margin is
  meaningless and the script says so in capitals rather than quietly publishing
  it. Hashes are taken after normalising line endings, because `git archive` on
  Windows honours `core.autocrlf` and would otherwise make a byte-identical
  `metrics.py` look changed on every single run.

The verdict rule is the same one [section 4](#4-results) uses -- the median
margin must clear both the full combined spread and a 1.0-point floor -- for
the reason two benchmarks in one repository with two different definitions of a
win is how a project ends up quoting whichever number flatters it.

```bash
blender -b --factory-startup --python tools/benchmark_organic.py -- \
    --before-ref e0360e5 --repeats 5
```

Generation on this corpus is far more expensive than on the primitives, so the
script cancels a single generation through the engine's own `should_cancel`
probe after `--gen-timeout` seconds (default 300) and reports the timeout rather
than hanging. A benchmark whose job is to catch a regression has to survive one.

### 7.5 Reproducing ORGANIC.md section 0

[`tools/diag_organic.py`](../tools/diag_organic.py) regenerates every measured
number in that contract's section 0 -- Suzanne's per-shell topology computed
twice by two independent implementations, the exact penalty decomposition and
ceiling analysis, the overlap classification with a Monte-Carlo cross-check of
the clipper, and the seam budget -- plus the corpus and oracle report.

```bash
blender -b --factory-startup --python tools/diag_organic.py -- --corpus
```

One correction is worth recording rather than burying. The scratch diagnostic
that produced the section 0.4 seam table also contained an overlap pass, and
that pass **reported zero overlap on Suzanne**: its Sutherland-Hodgman clipper
tested the wrong side of each clip edge and never oriented its input, so it
clipped every counter-clockwise pair away to nothing. Section 0.3's figure came
from a second, correct implementation. `tools/diag_organic.py` keeps only the
correct one and cross-checks it against Monte-Carlo integration, so a false
clean bill of health on the most consequential measurement in the contract
cannot happen silently again.
