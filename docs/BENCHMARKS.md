# Benchmarks

> **Status: PROVISIONAL.** The numbers in this file were measured on
> **2026-08-07** against a work-in-progress build. Two things are still moving:
> the scoring weights are being retuned -- and because every column here is
> scored by the same scorer, a weight change moves **all** of them -- and work
> is underway on the icosphere loss and on candidate diversity for closed
> meshes. Treat these figures as a direction of travel, not as a published
> result. Regeneration instructions are in
> [section 6](#6-regenerating-this-table).

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

**The scores matched between 4.5.4 and 5.1.2.** That cross-version agreement is
itself a result: the ranking a user sees does not appear to depend on which
supported Blender they run. It is subject to the same sampling caveat as
everything else here, and the cross-version comparison has not yet been re-run
under the repeated methodology.

No timing figures are published yet. A backend performance bug is being fixed
that adds roughly a second of pure overhead per candidate, so any wall-clock
number measured today would be unrepresentative of the next build. The raw
`bpy.ops.uv.*` calls themselves are sub-millisecond on these meshes.

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
margin exceeds **both** half the combined spread of the two series **and** an
absolute floor of 1.0 point. Everything else is a tie.

<!-- BEGIN BENCHMARK TABLE - regenerate wholesale, never hand-edit a cell -->

Blender 4.5.4, `balanced` preset, `seed=1`, 10 candidates, 5 repeats.

| mesh | `smart_project` | `unwrap+sharp` | **best-of-10** | verdict | noise | grid filled |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| cylinder | 25.4 [25.4-25.4] | 0.0 [0.0-0.0] | **84.6** [84.6-84.6] | **WIN +59.2** | 0.0 | 10/10 |
| suzanne | 13.0 [13.0-13.0] | 13.7 [13.7-13.7] | **42.3** [42.3-42.3] | **WIN +28.6** | 0.0 | 10/10 |
| torus | 49.1 [49.1-49.1] | 36.5 [36.5-36.5] | **54.8** [54.8-54.8] | **WIN +5.7** | 0.0 | 8/10 |
| cone | 74.4 [74.4-74.4] | 80.7 [80.7-80.7] | 81.5 [81.5-81.5] | tie +0.8 | 0.0 | 10/10 |
| cube | 88.9 [88.9-88.9] | 88.9 [88.9-88.9] | 88.9 [88.9-88.9] | tie +0.0 | 0.0 | 10/10 |
| sphere | 72.1 [72.1-72.1] | 52.8 [52.8-52.8] | 72.1 [72.1-72.7] | tie -0.0 | 0.6 | 8/10 |
| icosphere | 66.6 [66.6-66.6] | 73.3 [73.3-73.3] | 66.6 [66.6-66.6] | **LOSS -6.7** | 0.0 | 8/10 |

**Record: 3 wins, 3 ties, 1 loss.**

<!-- END BENCHMARK TABLE -->

### 4.1 What changed, and why the record got worse

An earlier single-run version of this benchmark reported **5 wins, 1 tie, 1
loss**. That was wrong, and the correction is worth stating plainly rather than
quietly swapping the table:

- **sphere** was reported as a **+0.6 win**. Its best-of-10 column varies
  between 72.1 and 72.7 across runs -- a spread of 0.6 -- so a 0.6 margin is
  indistinguishable from a different roll of Blender's packer. **Demoted to a
  tie.**
- **cone** was reported as a **+0.8 win**. That margin is stable, but 0.8
  points out of 100 is not a difference anyone can see on a model. It falls
  below the 1.0-point floor. **Demoted to a tie.**

Neither result changed. The standard for calling something a win changed, and
the honest record is 3 wins, 3 ties, 1 loss. That is the number this project
should be judged on.

### 4.2 The loss: icosphere

Best-of-10 **loses** on the icosphere, scoring 66.6 against `unwrap+sharp`'s
73.3 -- a 6.7-point deficit that is well outside the noise and is not going to
be argued away. This is a real result and it is not being buried.

A subdivided icosphere has no creases: every dihedral angle is shallow and
near-uniform, so the heuristics that make the winning rows work have nothing to
grip. Meanwhile the sharp-edge baseline degenerates into something close to
"cut nothing much and let ABF relax", which on a uniformly curved closed
surface happens to be a good answer -- note that it produces a single island at
L2 1.0090, better than anything in our grid.

Note also the **grid filled** column: the icosphere is one of three meshes where
a 10-candidate grid comes back with only 8 usable candidates. The three meshes
that fail to fill the grid -- sphere, torus, icosphere -- are exactly the three
that tie or lose. Those are closed meshes, and this is believed to be one root
cause rather than two problems: the search is not producing enough viable
diversity on closed surfaces. Work in progress.

Two honest readings, and both are true:

1. Best-of-N is a search. A search with a fixed budget over a
   heuristic-seeded space can lose to a well-chosen single pass, and this is
   what that looks like.
2. Raising the candidate count is the mitigation available today, not a fix. A
   fix means a seam strategy aimed at smooth closed meshes, which does not
   exist yet -- see [Limitations](../README.md#limitations-and-known-issues).

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
strategy**. Applied blindly it can silently produce an unusable unwrap that
still reports success. Trying several strategies and measuring the result is
how you catch that -- which is the clearest single argument for this add-on
that the corpus produces.

The `DUAL_GRAPH` strategy exists for this case specifically: it cuts the
*complement* of a spanning forest of the face dual graph, so by construction it
cannot leave a component un-openable.

### 4.4 Overlap is a metric because baselines really do produce it

On Suzanne, the `smart_project` baseline leaves **42 overlapping triangle
pairs**. Overlapping UVs silently corrupt baked normal and light maps, so
overlap area is one of the axes candidates are ranked on, and it carries a hard
step penalty rather than a proportional one -- a layout with any overlap at all
drops below an otherwise-equal clean layout.

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
