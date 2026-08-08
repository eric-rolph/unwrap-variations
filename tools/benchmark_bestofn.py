#!/usr/bin/env python3
"""Best-of-N against single-pass, repeated -- because Blender's UV ops are not deterministic.

The product premise is that generating N candidate unwraps and ranking them
automatically produces a better UV layout than running one unwrapper once. This
script is the measurement, and it is the *only* thing that may produce the
numbers in ``docs/BENCHMARKS.md``.

Why this repeats every measurement
----------------------------------

Blender's UV operators do not return the same result twice. Verified on 4.5.4,
five runs each against a freshly rebuilt, identical 48x24 UV sphere, comparing
a hash of the resulting UV buffer:

* ``smart_project`` -- 5 runs, 5 distinct results, varying by roughly **3%**.
  That is a genuinely different chart packing each time.
* ``unwrap(ANGLE_BASED)`` -- 5 runs, 5 distinct results, at the 1e-6 level.
* ``cube_project`` -- 5 runs, 5 distinct results, at the 1e-6 level. This one
  is a pure analytical projection with no solver in it at all, which is what
  identifies the small variation as float/threading noise rather than solver
  behaviour.

This is Blender's behaviour, not a defect in this add-on, and it is not
something the add-on can fix: the engine reproduces the same *recipes* from a
seed faithfully, but it cannot make Blender's solvers reproduce the same
*results* from a recipe.

The consequence is methodological. **A single run is a sample, not a
measurement.** A margin smaller than the run-to-run spread is not a win, and an
earlier single-run version of this script reported two such non-wins as wins.
So every column is measured ``--repeats`` times, reported as median with the
observed range, and a verdict is only awarded when the median margin clears
both the full combined spread and an absolute floor of 1.0 point.

Columns
-------

All three are scored by the same evaluator on the same mesh, so the comparison
is apples-to-apples:

* ``smart_project`` -- ``bpy.ops.uv.smart_project`` at Blender's own defaults.
* ``unwrap+sharp``  -- sharp edges marked as seams at 30 degrees, then
  ``bpy.ops.uv.unwrap(method='ANGLE_BASED')``.
* ``best-of-N``     -- ``engine.generate(count=N)``, top-ranked candidate.

Because every column runs through ``core.metrics.score`` with the same
``ScoreWeights`` preset, changing that preset changes **all three**. Never
hand-edit a single cell of the published table: regenerate the whole thing, or
you will end up comparing baselines from one weighting against our results from
another.

Run::

    blender.exe -b --factory-startup --python tools/benchmark_bestofn.py

Options go after a bare ``--`` so that Blender does not try to parse them::

    blender.exe -b --factory-startup --python tools/benchmark_bestofn.py -- \
        --count 10 --repeats 5 --seed 1 --preset balanced

Exit code is 0 when every mesh produced at least one candidate, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

# The add-on package lives one directory up from tools/. Resolving it from
# __file__ (rather than hard-coding a path) is what lets anyone else run this.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import bpy
except ImportError:  # pragma: no cover - the whole script needs Blender.
    print("ERROR: this script must be run inside Blender:")
    print("  blender -b --factory-startup --python tools/benchmark_bestofn.py")
    raise SystemExit(1) from None

from unwrap_variations.backends import seams as seam_mod  # noqa: E402
from unwrap_variations.core import bridge, engine  # noqa: E402
from unwrap_variations.core.evaluator import evaluate  # noqa: E402
from unwrap_variations.core.metrics import ScoreWeights, score  # noqa: E402

#: Absolute floor a median margin must clear to be called a win or a loss, in
#: score points. Without it, a mesh whose runs happen to be perfectly stable
#: (spread 0.0) would have any margin at all declared a victory -- including
#: 0.05, which is noise in the scorer rather than a difference anyone can see.
MIN_MARGIN = 1.0

#: Blender's own default Smart UV Project angle limit, in radians (66 degrees).
SMART_ANGLE_LIMIT_RAD = 1.15192

#: Sharpness used to pick seam edges for the unwrap baseline, in radians
#: (30 degrees) -- the value Blender's own "Select Sharp Edges" defaults to.
SHARP_ANGLE_RAD = 0.523599

#: The corpus. Primitives are used deliberately: they are reproducible on any
#: machine with no asset download, and between them they cover the topologies
#: that break unwrappers -- closed genus 0 (sphere, icosphere), genus 1 (torus),
#: capped tubes (cylinder, cone), pure hard-surface (cube) and an organic mesh
#: with real creases and ears (suzanne).
MESHES = (
    ("suzanne", lambda: bpy.ops.mesh.primitive_monkey_add()),
    ("sphere", lambda: bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)),
    ("torus", lambda: bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=24)),
    ("cylinder", lambda: bpy.ops.mesh.primitive_cylinder_add(vertices=32)),
    ("cube", lambda: bpy.ops.mesh.primitive_cube_add()),
    ("cone", lambda: bpy.ops.mesh.primitive_cone_add(vertices=32)),
    ("icosphere", lambda: bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3)),
)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def fresh(build):
    """Return a brand-new object built by ``build`` in an empty factory scene.

    Every measurement starts from ``read_factory_settings`` so that no seam,
    UV map or selection left behind by the previous run can leak into the next
    one. That leakage is not hypothetical: seams in particular survive an
    object rebuild if the scene is reused, and a repeated benchmark rebuilds
    the same object dozens of times.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build()
    obj = bpy.context.object
    if not len(obj.data.uv_layers):
        obj.data.uv_layers.new(name="UVMap")
    return obj


def measure_current(obj, weights):
    """Return the score of whatever UV layout ``obj`` currently carries.

    Seams are *read back from the layout* rather than taken from the mesh's
    seam flags, because ``smart_project`` invents its own charts without
    marking anything -- asking the mesh would report zero seam length for a
    heavily cut result and flatter it badly.
    """
    sample = bridge.extract_sample(obj)
    topo = seam_mod.build_edge_topology(sample)
    seams = engine.derive_uv_seams(sample, topo, sample.uvs)
    metrics = evaluate(sample.with_uvs(sample.uvs).with_seams(seams), compute_overlap=True)
    return score(metrics, weights)


def baseline_smart(obj, weights):
    """Score Blender's own Smart UV Project at its stock defaults."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=SMART_ANGLE_LIMIT_RAD, island_margin=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    return measure_current(obj, weights)


def baseline_unwrap(obj, weights):
    """Score Blender's stock angle-based Unwrap over sharp-edge seams.

    The deselect in the middle is load-bearing. ``mesh.edges_select_sharp``
    *extends* the current selection rather than replacing it, so without it
    everything stays selected and ``mark_seam`` turns **every** edge into a
    seam -- which is how an earlier version of this script reported 1152
    islands on a 1152-face sphere.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.mark_seam(clear=True)
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.edges_select_sharp(sharpness=SHARP_ANGLE_RAD)
    bpy.ops.mesh.mark_seam(clear=False)
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")
    return measure_current(obj, weights)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def band(values):
    """Format a run of samples as ``median [min-max]``."""
    return "%5.1f [%5.1f-%5.1f]" % (statistics.median(values), min(values), max(values))


def verdict(ours, baseline):
    """Decide win / tie / loss for one mesh from repeated samples.

    Args:
        ours: Best-of-N scores, one per repeat.
        baseline: The better of the two baselines, one per repeat.

    Returns:
        ``(label, margin, noise)``. ``noise`` is the combined observed spread of
        both series. A verdict is awarded only when the median margin clears
        **both** that full spread and :data:`MIN_MARGIN`, so a difference that
        could plausibly be a different roll of Blender's non-deterministic
        packer is reported as a tie.
    """
    margin = statistics.median(ours) - statistics.median(baseline)
    noise = (max(ours) - min(ours)) + (max(baseline) - min(baseline))
    if margin > noise and margin > MIN_MARGIN:
        return "WIN", margin, noise
    if margin < -noise and margin < -MIN_MARGIN:
        return "LOSS", margin, noise
    return "tie", margin, noise


def parse_args(argv):
    """Parse the arguments Blender passed through after a bare ``--``."""
    parser = argparse.ArgumentParser(
        prog="benchmark_bestofn.py",
        description="Measure best-of-N against Blender's single-pass unwrappers.",
    )
    parser.add_argument("--count", type=int, default=10, help="Candidates per mesh (default 10).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Samples per column (default 5). Blender's UV ops are non-deterministic; "
             "1 disables the statistics and is not suitable for publication.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Determinism seed (default 1).")
    parser.add_argument(
        "--preset",
        default="balanced",
        help="ScoreWeights preset used for ALL THREE columns (default balanced).",
    )
    return parser.parse_args(argv)


def main():
    """Run the benchmark and print the table. Returns a process exit code."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    repeats = max(1, args.repeats)
    weights = ScoreWeights.preset(args.preset)

    print("BEST-OF-%d vs SINGLE-PASS, median of %d runs [min-max], Blender %s"
          % (args.count, repeats, bpy.app.version_string))
    print("preset %r | seed %d" % (args.preset, args.seed))
    print("A verdict requires the median margin to exceed the FULL combined spread")
    print("and to exceed %.1f point; anything else is reported as a tie." % MIN_MARGIN)
    if repeats < 2:
        print("WARNING: --repeats 1 produces a sample, not a measurement. Do not publish it.")
    print("-" * 104)
    print("%-10s | %-20s | %-20s | %-20s | %s"
          % ("mesh", "smart_project", "unwrap+sharp", "best-of-%d (ours)" % args.count, "verdict"))
    print("-" * 104)

    record = {"WIN": 0, "tie": 0, "LOSS": 0}
    empty = []

    for name, build in MESHES:
        smart_runs, unwrap_runs, our_runs, counts = [], [], [], []
        for _ in range(repeats):
            smart_runs.append(baseline_smart(fresh(build), weights))
            unwrap_runs.append(baseline_unwrap(fresh(build), weights))
            report = engine.generate(
                fresh(build),
                count=args.count,
                seed=args.seed,
                weights=weights,
                store=False,
                compute_overlap=True,
            )
            our_runs.append(report.candidates[0].score if report.candidates else 0.0)
            counts.append(len(report.candidates))

        if not max(counts):
            print("%-10s | GENERATION PRODUCED NOTHING" % name)
            empty.append(name)
            record["LOSS"] += 1
            continue

        # The per-repeat maximum, not the maximum of the medians: on any given
        # run the user would have taken whichever baseline happened to be better.
        baseline = [max(a, b) for a, b in zip(smart_runs, unwrap_runs, strict=True)]
        label, margin, noise = verdict(our_runs, baseline)
        record[label] += 1

        print("%-10s | %s | %s | %s | %-4s %+6.1f (noise %.1f) %d/%d"
              % (name, band(smart_runs), band(unwrap_runs), band(our_runs),
                 label, margin, noise, min(counts), args.count))

    print("-" * 104)
    print("RECORD: %d wins, %d ties, %d losses"
          % (record["WIN"], record["tie"], record["LOSS"]))
    print()
    if empty:
        print("FAILED: no candidates at all for: %s" % ", ".join(empty))
        return 1
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
