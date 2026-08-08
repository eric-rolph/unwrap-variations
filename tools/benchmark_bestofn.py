#!/usr/bin/env python3
"""The decisive experiment: does best-of-N actually beat a single-pass unwrap?

The product premise is that generating N candidates and ranking them
automatically produces a better UV layout than Blender's one-shot default. If
that is not true and measurable, the add-on has no reason to exist. This script
is the measurement, and it is the *only* thing that may produce the numbers in
``docs/BENCHMARKS.md``.

Three columns, all scored by the same evaluator so the comparison is
apples-to-apples:

* ``smart_project`` -- ``bpy.ops.uv.smart_project`` at Blender's own defaults.
* ``unwrap+sharp``  -- sharp edges marked as seams at 30 degrees, then
  ``bpy.ops.uv.unwrap(method='ANGLE_BASED')``.
* ``best-of-N``     -- ``engine.generate(count=N)``, top-ranked candidate.

Because every column runs through ``core.metrics.score`` with the same
``ScoreWeights`` preset, changing that preset changes **all three** columns.
Never hand-edit a single cell of the published table: regenerate the whole
thing, or you will end up comparing baselines from one weighting against our
results from another.

Run::

    blender.exe -b --factory-startup --python tools/benchmark_bestofn.py

Options go after a bare ``--`` so that Blender does not try to parse them::

    blender.exe -b --factory-startup --python tools/benchmark_bestofn.py -- \
        --count 10 --seed 1 --preset balanced

Exit code is 0 when every mesh produced at least one candidate, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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

#: Score difference below which two layouts are called a tie rather than a win.
#: The scorer is a continuous function, so without a dead band a 0.05-point
#: difference would be reported as a victory.
TIE_BAND = 0.5

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
    object rebuild if the scene is reused.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build()
    obj = bpy.context.object
    if not len(obj.data.uv_layers):
        obj.data.uv_layers.new(name="UVMap")
    return obj


def measure_current(obj):
    """Evaluate whatever UV layout ``obj`` currently carries.

    Returns:
        ``(metrics, score)``. Seams are *read back from the layout* rather than
        taken from the mesh's seam flags -- ``smart_project`` invents its own
        charts without marking anything, so asking the mesh would report zero
        seam length for a heavily cut result.
    """
    sample = bridge.extract_sample(obj)
    topo = seam_mod.build_edge_topology(sample)
    seams = engine.derive_uv_seams(sample, topo, sample.uvs)
    metrics = evaluate(sample.with_uvs(sample.uvs).with_seams(seams), compute_overlap=True)
    return metrics, score(metrics, WEIGHTS)


def baseline_smart(obj):
    """Blender's own Smart UV Project at its stock defaults."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    start = time.perf_counter()
    bpy.ops.uv.smart_project(angle_limit=SMART_ANGLE_LIMIT_RAD, island_margin=0.0)
    seconds = time.perf_counter() - start
    bpy.ops.object.mode_set(mode="OBJECT")
    metrics, value = measure_current(obj)
    return metrics, value, seconds


def baseline_unwrap(obj):
    """Blender's stock angle-based Unwrap over sharp-edge seams.

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
    start = time.perf_counter()
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    seconds = time.perf_counter() - start
    bpy.ops.object.mode_set(mode="OBJECT")
    metrics, value = measure_current(obj)
    return metrics, value, seconds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv):
    """Parse the arguments Blender passed through after a bare ``--``."""
    parser = argparse.ArgumentParser(
        prog="benchmark_bestofn.py",
        description="Measure best-of-N against Blender's single-pass unwrappers.",
    )
    parser.add_argument("--count", type=int, default=10, help="Candidates per mesh (default 10).")
    parser.add_argument("--seed", type=int, default=1, help="Determinism seed (default 1).")
    parser.add_argument(
        "--preset",
        default="balanced",
        help="ScoreWeights preset used for ALL THREE columns (default balanced).",
    )
    parser.add_argument(
        "--no-recipes",
        action="store_true",
        help="Skip the per-mesh winning-recipe report at the end.",
    )
    return parser.parse_args(argv)


def main():
    """Run the benchmark and print the table. Returns a process exit code."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    global WEIGHTS
    WEIGHTS = ScoreWeights.preset(args.preset)

    print("=" * 108)
    print("BEST-OF-%d vs SINGLE-PASS  (score 0-100 higher better; L2 stretch >= 1.0, lower better)"
          % args.count)
    print("Blender %s | preset %r | seed %d" % (bpy.app.version_string, args.preset, args.seed))
    print("=" * 108)
    print("%-11s | %-22s | %-22s | %-24s | %s"
          % ("mesh", "smart_project", "unwrap+sharp", "best-of-%d (ours)" % args.count, "verdict"))
    print("%-11s | %-22s | %-22s | %-24s |"
          % ("", "score / L2 / isl", "score / L2 / isl", "score / L2 / isl"))
    print("-" * 108)

    wins = losses = ties = 0
    rows = []
    recipes = []
    empty = []

    for name, build in MESHES:
        obj = fresh(build)
        m_smart, sc_smart, t_smart = baseline_smart(obj)

        obj = fresh(build)
        m_unwrap, sc_unwrap, t_unwrap = baseline_unwrap(obj)

        obj = fresh(build)
        start = time.perf_counter()
        report = engine.generate(
            obj,
            count=args.count,
            seed=args.seed,
            weights=WEIGHTS,
            compute_overlap=True,
            store=False,
        )
        t_gen = time.perf_counter() - start

        if not report.candidates:
            print("%-11s | GENERATION PRODUCED NOTHING: %s" % (name, report.warnings[:2]))
            empty.append(name)
            losses += 1
            continue

        best = report.candidates[0]
        best_baseline = max(sc_smart, sc_unwrap)
        delta = best.score - best_baseline
        if delta > TIE_BAND:
            verdict = "WIN  +%.1f" % delta
            wins += 1
        elif delta < -TIE_BAND:
            verdict = "LOSE %.1f" % delta
            losses += 1
        else:
            verdict = "tie"
            ties += 1

        print("%-11s | %5.1f / %6.4f / %3d | %5.1f / %6.4f / %3d | %5.1f / %6.4f / %3d | %s"
              % (name,
                 sc_smart, m_smart.stretch_l2, m_smart.island_count,
                 sc_unwrap, m_unwrap.stretch_l2, m_unwrap.island_count,
                 best.score, best.metrics.stretch_l2, best.metrics.island_count,
                 verdict))
        rows.append((name, t_gen, t_smart, t_unwrap,
                     len(report.candidates), len(report.failures)))
        # Kept from this same run rather than regenerated: a second generate()
        # would double the runtime and, worse, could disagree with the table
        # above if anything in the engine were ever non-deterministic.
        recipes.append((name, best, [round(c.score, 1) for c in report.candidates]))

    print("-" * 108)
    print("RECORD: %d wins, %d ties, %d losses" % (wins, ties, losses))
    print()

    print("%-11s | %8s | %8s | %8s | %6s | %s"
          % ("mesh", "gen NxN", "smart", "unwrap", "cands", "failures"))
    for name, t_gen, t_smart, t_unwrap, n_cands, n_fail in rows:
        print("%-11s | %7.2fs | %7.3fs | %7.3fs | %4d/%-2d | %d"
              % (name, t_gen, t_smart, t_unwrap, n_cands, args.count, n_fail))

    if not args.no_recipes:
        # Which recipe won tells you whether the parameter space is doing real
        # work or whether one archetype always wins. If it is always the same
        # one, the search is theatre.
        print()
        print("Winning recipe per mesh:")
        for name, best, spread in recipes:
            print("  %-11s -> %-28s score=%5.1f L2=%.4f isl=%3d flip=%d ovl=%.2e"
                  % (name, best.params.describe(), best.score, best.metrics.stretch_l2,
                     best.metrics.island_count, best.metrics.flipped_tris,
                     best.metrics.overlap_area))
            print("               score spread across %d: %s" % (len(spread), spread))

    print()
    if empty:
        print("FAILED: no candidates at all for: %s" % ", ".join(empty))
        return 1
    print("DONE")
    return 0


#: Scoring weights, set from ``--preset`` in :func:`main`. Module level because
#: :func:`measure_current` is called from the baseline helpers too, and every
#: column must be scored with exactly the same weights.
WEIGHTS = ScoreWeights.preset("balanced")


if __name__ == "__main__":
    sys.exit(main())
