#!/usr/bin/env python3
"""Best-of-N on the organic corpus, before and after -- docs/ORGANIC.md 8.4.

``tools/benchmark_bestofn.py`` answers "is searching better than one pass?" on
seven Blender primitives.  This script answers a different and harder question
on eight organic meshes:

    Did the organic-mesh work actually make anything better?

That question cannot be answered by a table of current numbers, however good
they look, so **the "before" column is mandatory here**.  It is produced by
running the identical harness against a *pristine checkout of the pre-work
commit* -- a second copy of the ``unwrap_variations`` package, imported from its
own directory in the same interpreter, with ``sys.modules`` purged in between.
Both columns are then scored by **one** evaluator, the current one, so the
comparison measures generation and nothing else.

Two guards make that trustworthy, and both are printed:

* The resolved ``__file__`` and a SHA-256 of ``core/engine.py`` are shown for
  each root, so a reader can see the two columns really did run different code.
  Two identical hashes means the before column is not a before column.
* ``core/metrics.py`` and ``core/evaluator.py`` are hashed in both roots and
  compared.  ORGANIC.md rule 1 freezes the scorer for this work; if it moved
  anyway, every before/after margin is meaningless and the script says so in
  capitals rather than quietly publishing it.

Everything else follows ``tools/benchmark_bestofn.py``, including the parts that
are not optional.  Blender's UV operators are **not deterministic**
(``docs/BENCHMARKS.md`` section 3: five runs of ``smart_project`` on an
identical sphere give five distinct UV buffers, varying by about 3%), so every
column is measured ``--repeats`` times and reported as a median with its
observed range, and a verdict is awarded only when the median margin clears
**both** the full combined spread **and** an absolute floor of 1.0 point.

Run::

    blender -b --factory-startup --python tools/benchmark_organic.py -- \\
        --before-ref e0360e5 --repeats 5

or, with the pristine tree already extracted somewhere::

    blender -b --factory-startup --python tools/benchmark_organic.py -- \\
        --before-root /path/to/pristine --repeats 5

Exit code is 0 when every mesh produced at least one candidate, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import os
import statistics
import subprocess
import sys
import tarfile
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import bpy
except ImportError:  # pragma: no cover - the whole script needs Blender.
    print("ERROR: this script must be run inside Blender:")
    print("  blender -b --factory-startup --python tools/benchmark_organic.py")
    raise SystemExit(1) from None

#: Absolute floor a median margin must clear before it is called a win or a
#: loss.  Without it a mesh whose runs happened to be perfectly stable would
#: have any margin at all declared a victory, including 0.05.
MIN_MARGIN = 1.0

#: Blender's own default Smart UV Project angle limit, in radians (66 degrees).
SMART_ANGLE_LIMIT_RAD = 1.15192

#: Sharpness used to pick seam edges for the unwrap baseline, in radians
#: (30 degrees) -- the value Blender's own "Select Sharp Edges" defaults to.
#: On a smooth organic mesh it selects nothing whatsoever, which is the point:
#: ``docs/BENCHMARKS.md`` section 4.2 caveat 3 records that failure mode, and
#: half this corpus is built to provoke it.
SHARP_ANGLE_RAD = 0.523599

#: Files whose contents must match between the two roots for a before/after
#: margin to mean anything.  ORGANIC.md rule 1 freezes the first two.
SCORER_FILES = ("unwrap_variations/core/metrics.py", "unwrap_variations/core/evaluator.py")


def say(message: str = "") -> None:
    """Print one line and flush it.

    Blender block-buffers Python stdout when it is redirected to a file, so an
    unflushed long run looks identical to a hung one for its entire duration.
    A harness that takes an hour has to show progress while it takes it.
    """
    print(message)
    sys.stdout.flush()


def _load(module_name: str, filename: str):
    """Import a sibling ``tools/`` script by path (``tools/`` is not a package)."""
    key = "unwrap_variations_%s" % module_name
    if key in sys.modules:
        return sys.modules[key]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


corpus = _load("organic_corpus", "organic_corpus.py")


# ---------------------------------------------------------------------------
# Two package roots in one interpreter
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    """Return the first 12 hex digits of a file's SHA-256, or ``"missing"``.

    Line endings are normalised to LF before hashing.  That is not cosmetic:
    ``git archive`` on Windows honours ``core.autocrlf`` and writes CRLF, while
    the working tree here is LF, so the raw hashes of a byte-identical
    ``core/metrics.py`` differ (26148 bytes against 26737, 589 CRLF pairs).
    Without this the scorer guard fires on every single run, and a warning that
    always fires is a warning nobody reads.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return "missing"
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()[:12]


class Runtime:
    """One importable copy of ``unwrap_variations``, activated on demand.

    Two versions of a package cannot coexist under one module name, so this
    swaps them: every ``unwrap_variations*`` entry is dropped from
    ``sys.modules`` and the chosen root is put at the front of ``sys.path``
    before re-importing.  That is heavy-handed, and it is the only way to run
    both the old and the new engine in the same Blender process against the
    same mesh in the same session -- which is what makes the two columns
    comparable at all.

    The alternative, two Blender processes with a merge step afterwards, hides
    exactly the kind of environmental difference this is meant to expose.
    """

    def __init__(self, root: str, label: str) -> None:
        """Record a package root and the label it appears under in the table."""
        self.root = os.path.abspath(root)
        self.label = label
        self.engine = None
        self.bridge = None
        self.seams = None
        self.evaluate = None
        self.score = None
        self.weights_of = None

    @property
    def engine_hash(self) -> str:
        """Return the short hash of this root's ``core/engine.py``."""
        return sha256_file(os.path.join(self.root, "unwrap_variations", "core", "engine.py"))

    def scorer_hashes(self) -> tuple[str, ...]:
        """Return the short hashes of the files that decide what a score means."""
        return tuple(sha256_file(os.path.join(self.root, rel)) for rel in SCORER_FILES)

    def activate(self) -> "Runtime":
        """Purge and re-import ``unwrap_variations`` from this root."""
        for name in [
            n for n in sys.modules
            if n == "unwrap_variations" or n.startswith("unwrap_variations.")
        ]:
            del sys.modules[name]
        for path in (_REPO_ROOT, self.root):
            while path in sys.path:
                sys.path.remove(path)
        sys.path.insert(0, self.root)

        self.engine = importlib.import_module("unwrap_variations.core.engine")
        self.bridge = importlib.import_module("unwrap_variations.core.bridge")
        self.seams = importlib.import_module("unwrap_variations.backends.seams")
        evaluator = importlib.import_module("unwrap_variations.core.evaluator")
        metrics = importlib.import_module("unwrap_variations.core.metrics")
        self.evaluate = evaluator.evaluate
        self.score = metrics.score
        self.weights_of = metrics.ScoreWeights.preset
        return self

    def package_file(self) -> str:
        """Return the resolved ``__file__`` of the imported package."""
        module = sys.modules.get("unwrap_variations")
        return getattr(module, "__file__", "?") if module else "?"


def extract_ref(repo: str, ref: str, dest: str) -> str:
    """Extract a git ref into ``dest`` with ``git archive``, and return the path.

    ``git archive`` piped through Python's own ``tarfile`` rather than
    ``git worktree`` or a shell ``tar``: it needs no external tar binary, leaves
    no worktree registered in the repository, and produces a tree that is
    exactly the commit with no working-copy contamination -- which is the whole
    requirement for a "before" measurement.
    """
    out = os.path.join(dest, "before_%s" % ref)
    marker = os.path.join(out, "unwrap_variations", "core", "engine.py")
    if os.path.isfile(marker):
        return out
    os.makedirs(out, exist_ok=True)
    blob = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", repo, "archive", ref],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        # `filter` landed in 3.12 and becomes mandatory in 3.14; Blender 4.5.4
        # ships 3.11, so it is passed only when the interpreter understands it.
        if sys.version_info >= (3, 12):
            tar.extractall(out, filter="data")  # noqa: S202 - our own repository
        else:
            tar.extractall(out)  # noqa: S202 - our own repository
    return out


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def fresh(mesh):
    """Return a brand-new object for ``mesh`` in an empty factory scene.

    Every measurement restarts from ``read_factory_settings`` so that no seam,
    UV map or selection left behind by the previous run can leak into the next
    one.  That leakage is not hypothetical: seams in particular survive an
    object rebuild if the scene is reused, and a repeated benchmark rebuilds
    the same object dozens of times.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = corpus.to_object(mesh)
    if not len(obj.data.uv_layers):
        obj.data.uv_layers.new(name="UVMap")
    return obj


def measure_uvs(rt: Runtime, obj, uvs, weights) -> float:
    """Score a UV array against the *current* evaluator, whoever produced it.

    Both the before and the after column come through here, so the number in
    each cell is the same scorer applied to two different layouts.  Seams are
    derived from the layout rather than read from the mesh flags, because
    ``smart_project`` invents its own charts without marking anything and
    asking the mesh would report a seam length of zero for a heavily cut
    result.
    """
    sample = rt.bridge.extract_sample(obj)
    topo = rt.seams.build_edge_topology(sample)
    layout = sample.with_uvs(uvs) if uvs is not None else sample
    seams = rt.engine.derive_uv_seams(sample, topo, layout.uvs)
    metrics = rt.evaluate(layout.with_seams(seams), compute_overlap=True)
    return float(rt.score(metrics, weights))


def baseline_smart(rt: Runtime, mesh, weights) -> float:
    """Score Blender's own Smart UV Project at its stock defaults."""
    obj = fresh(mesh)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=SMART_ANGLE_LIMIT_RAD, island_margin=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    return measure_uvs(rt, obj, None, weights)


def baseline_unwrap(rt: Runtime, mesh, weights) -> tuple[float, bool]:
    """Score Blender's angle-based Unwrap over sharp-edge seams.

    The deselect in the middle is load-bearing: ``mesh.edges_select_sharp``
    *extends* the selection rather than replacing it, so without it ``mark_seam``
    turns every edge into a seam.

    Returns:
        ``(score, is_stock)``.  ``is_stock`` is true when the unwrap changed
        nothing measurable, i.e. the cell is really the mesh's pre-existing UV
        map.  On a smooth closed organic mesh a 30-degree threshold selects no
        edges at all, ``uv.unwrap`` gets no cuts, fails to solve, and still
        reports ``FINISHED`` -- so the flag is not a nicety, it is the
        difference between a baseline and a fiction.
    """
    obj = fresh(mesh)
    before = measure_uvs(rt, obj, None, weights) if len(obj.data.uv_layers) else None
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
    value = measure_uvs(rt, obj, None, weights)
    return value, before is not None and abs(value - before) < 1e-6


def baseline_xatlas(rt: Runtime, mesh, weights) -> tuple[float | None, str]:
    """Score a single xatlas pass, when xatlas is reachable at all.

    Returns ``(None, reason)`` when it is not.  ``docs/ARCHITECTURE.md`` section
    0 records that xatlas is absent from Blender's interpreter, so this column
    is empty unless the user has pointed the CLI backend at an external binary
    or installed the module into Blender's Python.  It is reported as absent
    rather than silently dropped, because an empty column and a zero are very
    different claims.
    """
    try:
        xatlas_mod = importlib.import_module("unwrap_variations.backends.xatlas_cli")
        params_mod = importlib.import_module("unwrap_variations.core.params")
        base_mod = importlib.import_module("unwrap_variations.backends.base")
    except ImportError as exc:
        return None, "import failed: %s" % exc
    backend = xatlas_mod.XatlasBackend()
    usable, reason = backend.is_available()
    if not usable:
        return None, reason
    obj = fresh(mesh)
    sample = rt.bridge.extract_sample(obj)
    request = base_mod.UnwrapRequest(
        obj_name=obj.name, params=params_mod.UnwrapParams(backend="xatlas"), sample=sample
    )
    result = backend.unwrap(request)
    if not result.ok or result.uvs is None:
        return None, "xatlas failed: %s" % (result.error or "no UVs")
    return measure_uvs(rt, obj, result.uvs, weights), reason


def best_of_n(rt: Runtime, scorer: Runtime, mesh, preset: str, *, count: int, seed: int,
              timeout: float = 0.0):
    """Generate with ``rt`` and score the winner with ``scorer``.

    Splitting the two is what keeps the before and after columns comparable:
    the old engine and the new engine each produce their own candidates, and
    one evaluator then measures both.  The winner is picked by the *common*
    scorer, not by whichever ranking the generating engine happened to use, so
    a change in ranking cannot masquerade as a change in quality.

    ``timeout`` is enforced through the engine's own ``should_cancel`` probe
    rather than by killing anything.  A benchmark whose job is to catch a
    regression must survive the regression: a pathological generation that
    would otherwise hang the whole run for hours is stopped, reported as a
    timeout, and the remaining meshes still get measured.

    Returns:
        ``(score, filled, seconds, timed_out)``.  ``score`` is 0.0 when
        generation produced nothing at all.
    """
    obj = fresh(mesh)
    deadline = time.perf_counter() + timeout if timeout > 0.0 else None
    started = time.perf_counter()
    report = rt.engine.generate(
        obj,
        count=count,
        seed=seed,
        weights=rt.weights_of(preset),
        store=False,
        compute_overlap=True,
        should_cancel=(lambda: time.perf_counter() > deadline) if deadline else None,
    )
    seconds = time.perf_counter() - started
    timed_out = bool(getattr(report, "cancelled", False)) or (
        deadline is not None and time.perf_counter() > deadline
    )
    if not report.candidates:
        return 0.0, 0, seconds, timed_out

    if scorer is rt:
        best = max(float(c.score) for c in report.candidates)
        return best, len(report.candidates), seconds, timed_out

    uv_list = [np.array(c.uvs, copy=True) for c in report.candidates]
    scorer.activate()
    obj2 = fresh(mesh)
    # The measuring weights come from the SCORER's own module, never from the
    # generating one: a ScoreWeights instance built by the old package is a
    # different class object, and passing it across would quietly make the two
    # columns depend on which root defined the dataclass.
    common = scorer.weights_of(preset)
    scored = [measure_uvs(scorer, obj2, uvs, common) for uvs in uv_list]
    return max(scored), len(uv_list), seconds, timed_out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def band(values) -> str:
    """Format a run of samples as ``median [min-max]``."""
    if not values:
        return "%20s" % "n/a"
    return "%5.1f [%5.1f-%5.1f]" % (statistics.median(values), min(values), max(values))


def verdict(ours, baseline) -> tuple[str, float, float]:
    """Decide win / tie / loss for one mesh from repeated samples.

    Returns ``(label, margin, noise)``.  ``noise`` is the combined observed
    spread of both series.  A verdict is awarded only when the median margin
    clears **both** that full spread and :data:`MIN_MARGIN`, so a difference
    that could plausibly be a different roll of Blender's non-deterministic
    packer is reported as a tie.  This is the same rule
    ``tools/benchmark_bestofn.py`` uses, deliberately: two benchmarks in one
    repository with two different definitions of a win is how a project ends up
    quoting whichever number flatters it.
    """
    if not ours or not baseline:
        return "n/a", 0.0, 0.0
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
        prog="benchmark_organic.py",
        description="Best-of-N on the organic corpus, measured before and after.",
    )
    parser.add_argument("--count", type=int, default=10, help="Candidates per mesh.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Samples per column (default 5).  Blender's UV ops are "
             "non-deterministic; 1 disables the statistics and must not be published.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Determinism seed.")
    parser.add_argument("--preset", default="balanced", help="ScoreWeights preset, all columns.")
    parser.add_argument("--meshes", default="", help="Comma-separated corpus subset.")
    parser.add_argument(
        "--before-root", default="", help="Directory holding the pristine pre-work package."
    )
    parser.add_argument(
        "--before-ref",
        default="",
        help="Git ref to extract into --scratch as the before column (e.g. e0360e5).",
    )
    parser.add_argument(
        "--scratch",
        default=os.path.join(os.path.expanduser("~"), ".uvvar_before"),
        help="Where --before-ref is extracted to.",
    )
    parser.add_argument(
        "--no-xatlas", action="store_true", help="Skip the xatlas column entirely."
    )
    parser.add_argument(
        "--gen-timeout",
        type=float,
        default=300.0,
        help="Seconds before one generation is cancelled through the engine's own "
             "should_cancel probe (0 disables).  A benchmark that has to catch a "
             "regression must survive it.",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Run the benchmark and print both tables.  Returns a process exit code."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    repeats = max(1, args.repeats)

    after = Runtime(_REPO_ROOT, "after").activate()
    weights = after.weights_of(args.preset)

    before = None
    before_note = "not requested"
    root = args.before_root
    if not root and args.before_ref:
        try:
            root = extract_ref(_REPO_ROOT, args.before_ref, args.scratch)
        except (OSError, subprocess.CalledProcessError) as exc:
            root = ""
            before_note = "extraction of %s failed: %s" % (args.before_ref, exc)
    if root:
        if os.path.isfile(os.path.join(root, "unwrap_variations", "core", "engine.py")):
            before = Runtime(root, "before")
            before_note = "pristine tree at %s" % root
        else:
            before_note = "no unwrap_variations package under %s" % root

    say("ORGANIC CORPUS BENCHMARK -- Blender %s, numpy %s"
          % (bpy.app.version_string, np.__version__))
    say("preset %r | seed %d | %d candidates | median of %d runs [min-max]"
          % (args.preset, args.seed, args.count, repeats))
    say("A verdict requires the median margin to exceed the FULL combined spread")
    say("and to exceed %.1f point; anything else is a tie." % MIN_MARGIN)
    if repeats < 2:
        say("WARNING: --repeats 1 produces a sample, not a measurement. Do not publish it.")
    say()
    say("BEFORE column: %s" % before_note)
    say("  after  root %s  engine.py %s" % (after.root, after.engine_hash))
    if before is not None:
        say("  before root %s  engine.py %s" % (before.root, before.engine_hash))
        say("  before package resolved to %s" % before.package_file())
        if before.engine_hash == after.engine_hash:
            say("  WARNING: identical engine.py in both roots -- the before column is")
            say("           measuring the same code and cannot show an improvement.")
        same_scorer = before.scorer_hashes() == after.scorer_hashes()
        say("  scorer (metrics.py, evaluator.py): before %s  after %s  -> %s"
              % (before.scorer_hashes(), after.scorer_hashes(),
                 "IDENTICAL" if same_scorer else "DIFFERENT"))
        if not same_scorer:
            say("  WARNING: THE SCORER MOVED between the two roots.  ORGANIC.md rule 1")
            say("           freezes core/metrics.py for this work.  Both columns are")
            say("           still scored by the CURRENT evaluator, so the table below")
            say("           remains internally consistent, but any claim that the")
            say("           before numbers match a previously published table is void.")
    say()

    wanted = [n.strip() for n in args.meshes.split(",") if n.strip()] or list(corpus.NAMES)
    unknown = [n for n in wanted if n not in corpus.NAMES]
    if unknown:
        say("ERROR: unknown mesh(es): %s" % ", ".join(unknown))
        return 1

    rows = []
    empty = []
    for name in wanted:
        mesh = corpus.build(name)
        smart_runs, unwrap_runs, before_runs, after_runs = [], [], [], []
        xatlas_runs = []
        stock_flags, counts, seconds = [], [], []
        timeouts = {"before": 0, "after": 0}
        xatlas_reason = "skipped"

        for _ in range(repeats):
            after.activate()
            smart_runs.append(baseline_smart(after, mesh, weights))
            uw_score, uw_stock = baseline_unwrap(after, mesh, weights)
            unwrap_runs.append(uw_score)
            stock_flags.append(uw_stock)
            if not args.no_xatlas:
                value, xatlas_reason = baseline_xatlas(after, mesh, weights)
                if value is not None:
                    xatlas_runs.append(value)

            if before is not None:
                before.activate()
                value, _filled, _secs, timed = best_of_n(
                    before, after, mesh, args.preset, count=args.count, seed=args.seed,
                    timeout=args.gen_timeout,
                )
                before_runs.append(value)
                timeouts["before"] += int(timed)

            after.activate()
            value, filled, secs, timed = best_of_n(
                after, after, mesh, args.preset, count=args.count, seed=args.seed,
                timeout=args.gen_timeout,
            )
            after_runs.append(value)
            counts.append(filled)
            seconds.append(secs)
            timeouts["after"] += int(timed)

        if not max(counts):
            say("%-17s GENERATION PRODUCED NOTHING" % name)
            empty.append(name)
            continue

        rows.append(
            {
                "mesh": name,
                "tris": mesh.n_tris,
                "smart": smart_runs,
                "unwrap": unwrap_runs,
                "xatlas": xatlas_runs,
                "before": before_runs,
                "after": after_runs,
                "stock": bool(stock_flags) and all(stock_flags),
                "filled": min(counts),
                "seconds": statistics.median(seconds),
                "timeouts": dict(timeouts),
                "xatlas_reason": xatlas_reason,
            }
        )
        note = ""
        if timeouts["after"] or timeouts["before"]:
            note = "  TIMED OUT: before %d/%d, after %d/%d repeats" % (
                timeouts["before"], repeats, timeouts["after"], repeats)
        say("%-17s measured (%d tris, %.1fs median generation)%s"
              % (name, mesh.n_tris, statistics.median(seconds), note))

    say()
    _print_baseline_table(rows, args)
    say()
    _print_before_after_table(rows, before is not None)
    say()
    _print_timings(rows)

    if empty:
        say()
        say("FAILED: no candidates at all for: %s" % ", ".join(empty))
        return 1
    say()
    say("DONE")
    return 0


def _print_baseline_table(rows, args) -> None:
    """Print the after-versus-single-pass table, in the published format."""
    width = 118
    say("=" * width)
    say("TABLE 1 -- best-of-%d against a single pass, on the organic corpus" % args.count)
    say("=" * width)
    say("%-17s %-7s %-20s %-22s %-20s %-20s %s"
        % ("mesh", "tris", "smart_project", "unwrap+sharp", "xatlas",
           "best-of-%d (after)" % args.count, "verdict"))
    say("-" * width)
    record = {"WIN": 0, "tie": 0, "LOSS": 0, "n/a": 0}
    for row in rows:
        baseline = [
            max(a, b) for a, b in zip(row["smart"], row["unwrap"], strict=True)
        ]
        label, margin, noise = verdict(row["after"], baseline)
        record[label] += 1
        say("%-17s %-7d %-20s %-22s %-20s %-20s %-4s %+6.1f (noise %.1f) %d/%d"
              % (row["mesh"], row["tris"], band(row["smart"]),
                 band(row["unwrap"]) + (" *" if row["stock"] else ""),
                 band(row["xatlas"]), band(row["after"]),
                 label, margin, noise, row["filled"], args.count))
    say("-" * width)
    say("RECORD: %d wins, %d ties, %d losses"
          % (record["WIN"], record["tie"], record["LOSS"]))
    if any(row["stock"] for row in rows):
        say("* the unwrap+sharp baseline did nothing measurable on this mesh: a")
        say("  30-degree sharpness threshold selected no edges, uv.unwrap got no cuts,")
        say("  failed to solve, and still reported FINISHED. What that cell contains is")
        say("  the mesh's pre-existing UV map, not a single-pass unwrap.")
    missing = [row["mesh"] for row in rows if not row["xatlas"]]
    if missing:
        say("xatlas column empty on %d mesh(es): %s"
              % (len(missing), rows[0]["xatlas_reason"] if rows else "?"))


def _print_before_after_table(rows, have_before: bool) -> None:
    """Print the before-versus-after table, the one that answers the question."""
    width = 96
    say("=" * width)
    say("TABLE 2 -- did the organic-mesh work help?  before against after")
    say("=" * width)
    if not have_before:
        say("NO BEFORE COLUMN WAS MEASURED.")
        say("Pass --before-ref e0360e5 (or --before-root PATH). Without it this")
        say("script cannot show that any of the organic work improved anything, and")
        say("a table of current numbers on its own is not evidence that it did.")
        return
    say("%-17s %-20s %-20s %s" % ("mesh", "before", "after", "verdict"))
    say("-" * width)
    record = {"WIN": 0, "tie": 0, "LOSS": 0, "n/a": 0}
    for row in rows:
        label, margin, noise = verdict(row["after"], row["before"])
        record[label] += 1
        say("%-17s %-20s %-20s %-4s %+6.1f (noise %.1f)"
              % (row["mesh"], band(row["before"]), band(row["after"]), label, margin, noise))
    say("-" * width)
    say("RECORD vs BEFORE: %d improved, %d unchanged, %d regressed"
          % (record["WIN"], record["tie"], record["LOSS"]))
    if record["LOSS"]:
        say("A REGRESSION IS A BUG, not a trade-off: ORGANIC.md rule 2 says no stage")
        say("may make a candidate worse.")


def _print_timings(rows) -> None:
    """Print per-mesh generation cost next to mesh size."""
    say("=" * 60)
    say("TIMINGS -- median seconds for one best-of-N generation")
    say("=" * 60)
    say("%-17s %-9s %s" % ("mesh", "tris", "seconds"))
    for row in rows:
        say("%-17s %-9d %.2f" % (row["mesh"], row["tris"], row["seconds"]))


if __name__ == "__main__":
    sys.exit(main())
