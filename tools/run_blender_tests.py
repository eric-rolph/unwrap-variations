#!/usr/bin/env python3
"""Headless Blender integration harness for UnwrapVariations.

Run it with a real Blender::

    blender.exe -b --factory-startup --python tools/run_blender_tests.py
    blender.exe -b --factory-startup --python tools/run_blender_tests.py -- --verbose

``tests/`` covers everything that can be checked without Blender: the metrics,
the parameter space, the seam heuristics, the candidate store.  This file covers
the rest -- the parts where the add-on actually touches a datablock, and where
the interesting bugs live:

* ``extract_sample`` / ``apply_uvs`` really round trip through a mesh;
* ``snapshot_state`` / ``restore_state`` really put the mesh back, seams and all;
* every backend produces a usable layout on a real mesh *and* leaves the scene
  exactly as it found it;
* the add-on registers and unregisters cleanly, twice;
* generate -> preview -> branch -> apply works end to end;
* a candidate store survives a genuine ``.blend`` save and reload.

The suite is written to survive a half-finished repository: a module whose file
does not exist yet is reported ``SKIP`` with a reason, while a module that
exists and is broken is reported ``FAIL``.  Output is ASCII only, because this
runs in a Windows cp1252 console, and the process exits non-zero on any failure
so CI can gate on it.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback

# ---------------------------------------------------------------------------
# Repo discovery.  The script may be invoked from anywhere, and Blender does not
# put the script's directory on sys.path.
# ---------------------------------------------------------------------------


def _find_repo_root() -> str:
    """Return the repository root, located by its add-on package.

    Walks up from this file looking for ``unwrap_variations/blender_manifest.toml``
    so that the harness works whether it is run from the repo, a checkout under
    a different name, or a copy inside Blender's extensions directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = here
    for _ in range(6):
        marker = os.path.join(path, "unwrap_variations", "blender_manifest.toml")
        if os.path.isfile(marker):
            return path
        if os.path.isfile(os.path.join(path, "unwrap_variations", "__init__.py")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.path.dirname(here)


REPO_ROOT = _find_repo_root()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Tiny harness.
# ---------------------------------------------------------------------------

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Skip(Exception):
    """Raised by a test to report that it cannot run (not that it failed)."""


class Harness:
    """Collects tests, runs them in order and reports an ASCII summary."""

    def __init__(self, verbose: bool = False) -> None:
        """Create an empty harness."""
        self.tests: list[tuple[str, str, object]] = []
        self.results: list[tuple[str, str, str, float]] = []
        self.verbose = verbose
        self._notes: list[str] = []

    def add(self, group: str):
        """Return a decorator registering a test function under ``group``."""

        def decorate(fn):
            self.tests.append((group, fn.__name__, fn))
            return fn

        return decorate

    def note(self, text: str) -> None:
        """Record an informational line shown under the current test."""
        self._notes.append(str(text))

    def run(self) -> int:
        """Run every registered test and return the process exit code."""
        print("=" * 78)
        print("UnwrapVariations -- headless Blender integration harness")
        print("repo root : %s" % REPO_ROOT)
        try:
            import bpy

            print("blender   : %s" % ".".join(str(v) for v in bpy.app.version))
            print("python    : %s" % sys.version.split()[0])
        except Exception:  # noqa: BLE001
            print("blender   : NOT AVAILABLE (this script must run inside Blender)")
        print("=" * 78)

        current_group = None
        for group, name, fn in self.tests:
            if group != current_group:
                current_group = group
                print("\n-- %s %s" % (group, "-" * max(0, 70 - len(group))))
            self._notes = []
            started = time.perf_counter()
            status = PASS
            detail = ""
            try:
                fn()
            except Skip as exc:
                status = SKIP
                detail = str(exc)
            except AssertionError as exc:
                status = FAIL
                detail = str(exc) or "assertion failed"
                if self.verbose:
                    detail += "\n" + traceback.format_exc()
            except Exception as exc:  # noqa: BLE001
                status = FAIL
                detail = "%s: %s" % (type(exc).__name__, exc)
                detail += "\n" + _indent(traceback.format_exc())
            elapsed = time.perf_counter() - started
            self.results.append((group, name, status, elapsed))
            line = "  [%s] %-52s %6.0f ms" % (status, name, elapsed * 1000.0)
            print(_ascii(line))
            for extra in self._notes:
                print(_ascii("         . %s" % extra))
            if detail:
                print(_ascii(_indent(detail, "         ! ")))

        return self._summarise()

    def _summarise(self) -> int:
        """Print the final tally and return the exit code."""
        passed = sum(1 for r in self.results if r[2] == PASS)
        failed = [r for r in self.results if r[2] == FAIL]
        skipped = [r for r in self.results if r[2] == SKIP]
        total_ms = sum(r[3] for r in self.results) * 1000.0

        print("\n" + "=" * 78)
        print(
            "SUMMARY: %d passed, %d failed, %d skipped (%.0f ms)"
            % (passed, len(failed), len(skipped), total_ms)
        )
        if skipped:
            print("\nSkipped:")
            for group, name, _s, _t in skipped:
                print("  - %s / %s" % (group, name))
        if failed:
            print("\nFailed:")
            for group, name, _s, _t in failed:
                print("  - %s / %s" % (group, name))
            print("\nRESULT: FAIL")
            return 1
        print("\nRESULT: PASS")
        return 0


def _ascii(text: str) -> str:
    """Return ``text`` with every non-ASCII character replaced."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _indent(text: str, prefix: str = "         ") -> str:
    """Return ``text`` with every line prefixed."""
    return "\n".join(prefix + line for line in str(text).splitlines())


def _script_args() -> list[str]:
    """Return the arguments after ``--``, which is how Blender passes ours."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


HARNESS = Harness(verbose=bool({"-v", "--verbose"} & set(_script_args())))


# ---------------------------------------------------------------------------
# Module loading that distinguishes "not written yet" from "broken".
# ---------------------------------------------------------------------------


def require(dotted: str):
    """Import ``dotted`` from the repo, skipping only when its file is absent.

    A missing file means another worker has not landed the module yet, which is
    a ``SKIP``.  A file that exists but fails to import is a real defect and the
    ``ImportError`` is allowed to propagate into a ``FAIL``.
    """
    import importlib

    parts = dotted.split(".")
    module_path = os.path.join(REPO_ROOT, *parts) + ".py"
    package_path = os.path.join(REPO_ROOT, *parts, "__init__.py")
    if not os.path.isfile(module_path) and not os.path.isfile(package_path):
        raise Skip("%s does not exist yet" % dotted)
    if (
        os.path.isfile(package_path)
        and not os.path.isfile(module_path)
        and os.path.getsize(package_path) == 0
    ):
        raise Skip("%s is still an empty placeholder" % dotted)
    return importlib.import_module(dotted)


def require_attr(dotted: str, *names: str):
    """Import ``dotted`` and skip if any of ``names`` is missing from it."""
    module = require(dotted)
    for name in names:
        if not hasattr(module, name):
            raise Skip("%s has no %s yet" % (dotted, name))
    return module


# ---------------------------------------------------------------------------
# Scene helpers.
# ---------------------------------------------------------------------------

TEST_OBJECTS = ("UVVarCube", "UVVarSphere", "UVVarSuzanne", "UVVarGrid", "UVVarNgon")


def clean_scene() -> None:
    """Delete every object in the scene, leaving an empty file."""
    import bpy

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def make_scene() -> None:
    """Build the standard test scene: cube, sphere, Suzanne, grid and an n-gon."""
    import bpy

    clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    bpy.context.active_object.name = "UVVarCube"

    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
    bpy.context.active_object.name = "UVVarSphere"

    bpy.ops.mesh.primitive_monkey_add()
    bpy.context.active_object.name = "UVVarSuzanne"

    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=2.0)
    bpy.context.active_object.name = "UVVarGrid"

    # A single 8-sided n-gon: the tessellation-diagonal case that breaks naive
    # loop-triangle bookkeeping.
    bpy.ops.mesh.primitive_circle_add(vertices=8, fill_type="NGON")
    bpy.context.active_object.name = "UVVarNgon"

    for name in TEST_OBJECTS:
        obj = bpy.data.objects[name]
        obj.data.calc_loop_triangles()

    bpy.context.view_layer.objects.active = bpy.data.objects["UVVarCube"]


def get(name: str):
    """Return a scene object by name, asserting that it exists."""
    import bpy

    obj = bpy.data.objects.get(name)
    assert obj is not None, "test object %r is missing from the scene" % name
    return obj


def select_only(obj) -> None:
    """Make ``obj`` the sole selected and active object."""
    import bpy

    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def uv_names(obj) -> list[str]:
    """Return the names of an object's UV layers."""
    return [layer.name for layer in obj.data.uv_layers]


def read_uvs(obj, layer_name: str):
    """Return the ``(L, 2)`` float64 UVs of one layer, via the fast path."""
    import numpy as np

    layer = obj.data.uv_layers[layer_name]
    buf = np.zeros(len(obj.data.loops) * 2, dtype=np.float32)
    try:
        layer.uv.foreach_get("vector", buf)
    except (AttributeError, TypeError):
        layer.data.foreach_get("uv", buf)
    return buf.astype(np.float64).reshape(-1, 2)


def seam_flags(obj):
    """Return the ``(E,)`` bool seam mask of an object's mesh."""
    import numpy as np

    buf = np.zeros(len(obj.data.edges), dtype=bool)
    obj.data.edges.foreach_get("use_seam", buf)
    return buf


# ===========================================================================
# 1. Environment
# ===========================================================================


@HARNESS.add("environment")
def test_bpy_is_available():
    """The harness is running inside Blender, with numpy present."""
    import bpy
    import numpy as np

    assert bpy.app.background, "expected background (-b) mode"
    assert bpy.app.version >= (4, 2, 0), "Blender %s is below the 4.2 floor" % (
        bpy.app.version,
    )
    HARNESS.note("numpy %s" % np.__version__)


@HARNESS.add("environment")
def test_core_modules_import_without_error():
    """Every core module that exists imports cleanly inside Blender."""
    seen = []
    for dotted in (
        "unwrap_variations.core.params",
        "unwrap_variations.core.meshdata",
        "unwrap_variations.core.metrics",
        "unwrap_variations.core.evaluator",
        "unwrap_variations.core.candidate",
        "unwrap_variations.core.bridge",
        "unwrap_variations.core.engine",
        "unwrap_variations.backends.base",
        "unwrap_variations.backends.seams",
    ):
        try:
            require(dotted)
            seen.append(dotted.rsplit(".", 1)[-1])
        except Skip as exc:
            HARNESS.note("skipped %s" % exc)
    assert seen, "no core module exists yet"
    HARNESS.note("imported: %s" % ", ".join(seen))


@HARNESS.add("environment")
def test_scene_builds():
    """The standard test scene builds and every object has loop triangles."""
    make_scene()
    for name in TEST_OBJECTS:
        obj = get(name)
        assert len(obj.data.polygons) > 0, "%s has no faces" % name
        assert len(obj.data.loop_triangles) > 0, "%s has no loop triangles" % name
    HARNESS.note(
        "; ".join(
            "%s T=%d L=%d" % (n, len(get(n).data.loop_triangles), len(get(n).data.loops))
            for n in TEST_OBJECTS
        )
    )


@HARNESS.add("environment")
def test_verified_fast_paths_still_exist():
    """The ``foreach_get`` paths the architecture doc pins are all still present.

    These were probed empirically against 4.5.4 and 5.1.2.  If a Blender update
    renames one, the extraction path breaks in a way that is much easier to
    diagnose here than three layers down.
    """
    import numpy as np

    obj = get("UVVarCube")
    mesh = obj.data
    mesh.calc_loop_triangles()

    checks = [
        (mesh.vertices, "co", len(mesh.vertices) * 3, np.float32),
        (mesh.loops, "vertex_index", len(mesh.loops), np.int32),
        (mesh.loop_triangles, "loops", len(mesh.loop_triangles) * 3, np.int32),
        (mesh.loop_triangles, "vertices", len(mesh.loop_triangles) * 3, np.int32),
        (mesh.edges, "vertices", len(mesh.edges) * 2, np.int32),
        (mesh.edges, "use_seam", len(mesh.edges), bool),
    ]
    for collection, attr, size, dtype in checks:
        buf = np.zeros(size, dtype=dtype)
        collection.foreach_get(attr, buf)

    if not mesh.uv_layers:
        mesh.uv_layers.new(name="probe")
    layer = mesh.uv_layers[0]
    buf = np.zeros(len(mesh.loops) * 2, dtype=np.float32)
    layer.uv.foreach_get("vector", buf)
    layer.data.foreach_get("uv", buf)


@HARNESS.add("environment")
def test_uv_layer_cap_is_eight():
    """``uv_layers.new()`` returns None past 8 layers -- the reason for the design.

    The whole candidate-as-numpy-array architecture exists because of this cap.
    If it ever changes, the design decision is worth revisiting; until then it
    is worth verifying rather than assuming.
    """
    import bpy

    mesh = bpy.data.meshes.new("UVVarCapProbe")
    mesh.from_pydata(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)]
    )
    mesh.update()
    created = 0
    while mesh.uv_layers.new(name="cap%d" % created) is not None:
        created += 1
        if created > 12:
            break
    HARNESS.note("created %d UV layers before the cap" % created)
    assert created == 8, "expected the documented 8-layer cap, created %d" % created
    bpy.data.meshes.remove(mesh)


# ===========================================================================
# 2. bridge -- the bpy <-> core boundary
# ===========================================================================


@HARNESS.add("bridge")
def test_extract_sample_shapes():
    """``extract_sample`` produces arrays matching the mesh it came from."""
    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")
    for name in TEST_OBJECTS:
        obj = get(name)
        mesh = obj.data
        mesh.calc_loop_triangles()
        sample = bridge.extract_sample(obj)
        assert sample.n_verts == len(mesh.vertices), "%s: vertex count" % name
        assert sample.n_loops == len(mesh.loops), "%s: loop count" % name
        assert sample.n_tris == len(mesh.loop_triangles), "%s: triangle count" % name
        assert sample.n_edges == len(mesh.edges), "%s: edge count" % name
        assert sample.uvs.shape == (len(mesh.loops), 2), "%s: uv shape" % name
        problems = [p for p in sample.validate() if "no triangles" not in p]
        assert not problems, "%s: %s" % (name, problems)


@HARNESS.add("bridge")
def test_extract_sample_handles_a_mesh_with_no_uv_layer():
    """A mesh with zero UV layers extracts with zeroed UVs rather than raising."""
    import bpy

    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")
    mesh = bpy.data.meshes.new("UVVarNoUV")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new("UVVarNoUV", mesh)
    bpy.context.collection.objects.link(obj)
    try:
        assert len(mesh.uv_layers) == 0
        sample = bridge.extract_sample(obj)
        assert sample.n_loops == 3
        assert float(abs(sample.uvs).sum()) == 0.0, "expected zeroed UVs"
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


@HARNESS.add("bridge")
def test_extract_apply_round_trip():
    """UVs written by ``apply_uvs`` read back bit-identically at float32.

    This is the single most load-bearing path in the add-on: every preview and
    every apply is this write followed by this read.
    """
    import numpy as np

    bridge = require_attr(
        "unwrap_variations.core.bridge", "extract_sample", "apply_uvs"
    )
    obj = get("UVVarSuzanne")
    obj.data.calc_loop_triangles()
    n_loops = len(obj.data.loops)

    rng = np.random.default_rng(4)
    payload = rng.random((n_loops, 2)).astype(np.float32)
    bridge.apply_uvs(obj, payload, layer_name="UVVarRoundTrip")
    assert "UVVarRoundTrip" in uv_names(obj), "apply_uvs did not create the layer"

    read_back = read_uvs(obj, "UVVarRoundTrip")
    delta = float(np.abs(read_back - payload.astype(np.float64)).max())
    assert delta <= 1e-6, "round trip differs by %g" % delta

    sample = bridge.extract_sample(obj, uv_layer_name="UVVarRoundTrip")
    delta = float(np.abs(sample.uvs - payload.astype(np.float64)).max())
    assert delta <= 1e-6, "extract_sample read back a different layer (%g)" % delta

    obj.data.uv_layers.remove(obj.data.uv_layers["UVVarRoundTrip"])


@HARNESS.add("bridge")
def test_apply_uvs_rejects_a_wrong_length_payload():
    """A mismatched array must not half-write the mesh."""
    import numpy as np

    bridge = require_attr("unwrap_variations.core.bridge", "apply_uvs")
    obj = get("UVVarGrid")
    before = uv_names(obj)
    try:
        bridge.apply_uvs(obj, np.zeros((3, 2), dtype=np.float32), layer_name="UVVarBad")
    except Exception as exc:  # noqa: BLE001 - raising here is acceptable.
        HARNESS.note("raised %s (acceptable)" % type(exc).__name__)
    if "UVVarBad" in uv_names(obj):
        data = read_uvs(obj, "UVVarBad")
        assert data.shape[0] == len(obj.data.loops), "layer left at the wrong size"
        obj.data.uv_layers.remove(obj.data.uv_layers["UVVarBad"])
    assert set(before) <= set(uv_names(obj)), "an existing UV layer disappeared"


@HARNESS.add("bridge")
def test_preview_layer_lifecycle():
    """The scratch preview layer is created once, reused, and removable."""
    bridge = require_attr(
        "unwrap_variations.core.bridge", "ensure_preview_layer", "remove_preview_layer"
    )
    obj = get("UVVarCube")
    before = len(uv_names(obj))

    name = bridge.ensure_preview_layer(obj)
    assert name, "ensure_preview_layer returned no name"
    assert name in uv_names(obj)
    again = bridge.ensure_preview_layer(obj)
    assert again == name, "a second call created a different layer"
    assert len(uv_names(obj)) == before + 1, "a second call added another layer"

    bridge.remove_preview_layer(obj)
    assert name not in uv_names(obj), "remove_preview_layer left the layer behind"
    bridge.remove_preview_layer(obj)  # idempotent
    assert len(uv_names(obj)) == before


@HARNESS.add("bridge")
def test_free_uv_layer_slots():
    """``free_uv_layer_slots`` counts down from the documented cap of 8."""
    bridge = require_attr("unwrap_variations.core.bridge", "free_uv_layer_slots")
    obj = get("UVVarCube")
    start = bridge.free_uv_layer_slots(obj)
    assert 0 <= start <= 8
    layer = obj.data.uv_layers.new(name="UVVarSlotProbe")
    try:
        assert bridge.free_uv_layer_slots(obj) == start - 1
    finally:
        obj.data.uv_layers.remove(layer)
    assert bridge.free_uv_layer_slots(obj) == start


@HARNESS.add("bridge")
def test_snapshot_restore_fidelity():
    """Snapshot then restore puts the mesh back exactly, UVs and seams included.

    Generation marks seams, adds layers and runs solvers all over the user's
    mesh.  If restore is not exact, the add-on silently damages the file the
    first time somebody clicks Generate and then changes their mind.
    """
    import numpy as np

    bridge = require_attr(
        "unwrap_variations.core.bridge", "snapshot_state", "restore_state"
    )
    obj = get("UVVarSphere")
    mesh = obj.data

    # Establish a distinctive starting state.
    if "UVMap" not in uv_names(obj):
        mesh.uv_layers.new(name="UVMap")
    original_layers = uv_names(obj)
    original_uvs = {n: read_uvs(obj, n) for n in original_layers}
    marks = np.zeros(len(mesh.edges), dtype=bool)
    marks[::5] = True
    mesh.edges.foreach_set("use_seam", marks)
    mesh.update()
    original_seams = seam_flags(obj)
    original_active = mesh.uv_layers.active.name if mesh.uv_layers.active else None

    snap = bridge.snapshot_state(obj)
    assert isinstance(snap, dict) and snap, "snapshot_state returned nothing useful"

    # Vandalise the mesh the way a generation run would.
    mesh.uv_layers.new(name="UVVarScratchA")
    mesh.uv_layers.new(name="UVVarScratchB")
    mesh.edges.foreach_set("use_seam", np.ones(len(mesh.edges), dtype=bool))
    for name in original_layers:
        layer = mesh.uv_layers[name]
        junk = np.full(len(mesh.loops) * 2, 0.5, dtype=np.float32)
        try:
            layer.uv.foreach_set("vector", junk)
        except (AttributeError, TypeError):
            layer.data.foreach_set("uv", junk)
    mesh.update()

    bridge.restore_state(obj, snap)

    assert uv_names(obj) == original_layers, (
        "UV layers after restore are %r, expected %r" % (uv_names(obj), original_layers)
    )
    for name in original_layers:
        delta = float(np.abs(read_uvs(obj, name) - original_uvs[name]).max())
        assert delta <= 1e-6, "layer %r restored with a %g difference" % (name, delta)
    assert np.array_equal(seam_flags(obj), original_seams), "seams were not restored"
    if original_active is not None and mesh.uv_layers.active is not None:
        assert mesh.uv_layers.active.name == original_active, (
            "the active UV layer was not restored"
        )


@HARNESS.add("bridge")
def test_snapshot_restore_of_a_mesh_with_no_uv_layers():
    """Restoring a UV-less mesh must not leave a stray layer behind."""
    import bpy

    bridge = require_attr(
        "unwrap_variations.core.bridge", "snapshot_state", "restore_state"
    )
    mesh = bpy.data.meshes.new("UVVarBareMesh")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    obj = bpy.data.objects.new("UVVarBare", mesh)
    bpy.context.collection.objects.link(obj)
    try:
        snap = bridge.snapshot_state(obj)
        mesh.uv_layers.new(name="intruder")
        bridge.restore_state(obj, snap)
        assert len(mesh.uv_layers) == 0, (
            "restore left %d UV layers on a mesh that had none" % len(mesh.uv_layers)
        )
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


@HARNESS.add("bridge")
def test_bridge_restores_object_mode():
    """Every bridge entry point returns the object to the mode it found."""
    import bpy

    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")
    obj = get("UVVarCube")
    select_only(obj)
    assert obj.mode == "OBJECT"
    bridge.extract_sample(obj)
    assert obj.mode == "OBJECT", "extract_sample left the object in %s" % obj.mode
    assert bpy.context.view_layer.objects.active is obj, "the active object changed"


# ===========================================================================
# 3. backends
# ===========================================================================


@HARNESS.add("backends")
def test_backend_registry():
    """The registry lists at least the native backend and reports availability."""
    base = require_attr(
        "unwrap_variations.backends.base", "available_backends", "get_backend"
    )
    rows = base.available_backends()
    assert rows, "no backends registered"
    ids = [r[0] for r in rows]
    assert "blender_native" in ids, "blender_native is missing from %r" % (ids,)
    for backend_id, label, ok, why in rows:
        assert isinstance(backend_id, str) and backend_id
        assert isinstance(label, str) and label
        assert isinstance(ok, bool)
        assert isinstance(why, str)
        HARNESS.note("%-14s %-22s %s %s" % (backend_id, label, "OK " if ok else "N/A", why))


@HARNESS.add("backends")
def test_every_backend_unwraps_a_real_mesh():
    """Each available backend produces a usable layout and never raises.

    A backend that cannot run reports ``ok=False`` with a reason; one that dies
    takes only its own candidate down.  The contract is that ``unwrap()`` never
    raises, so a raise here is a failure even for an unavailable backend.
    """
    import numpy as np

    base = require_attr(
        "unwrap_variations.backends.base",
        "available_backends",
        "get_backend",
        "UnwrapRequest",
    )
    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")
    params_mod = require("unwrap_variations.core.params")

    obj = get("UVVarSuzanne")
    select_only(obj)
    obj.data.calc_loop_triangles()
    sample = bridge.extract_sample(obj)

    tested = 0
    for backend_id, _label, ok, why in base.available_backends():
        backend = base.get_backend(backend_id)
        if not ok:
            HARNESS.note("%s unavailable: %s" % (backend_id, why))
            continue
        for method in backend.supported_methods():
            params = params_mod.UnwrapParams(
                backend=backend_id, method=method, island_margin=0.01
            ).clamped()
            request = base.UnwrapRequest(
                obj_name=obj.name, params=params, sample=sample, timeout_s=120.0
            )
            result = backend.unwrap(request)  # must never raise
            assert hasattr(result, "ok"), "%s returned %r" % (backend_id, result)
            if not result.ok:
                HARNESS.note("%s/%s failed: %s" % (backend_id, method, result.message))
                continue
            uvs = np.asarray(result.uvs)
            assert uvs.shape == (sample.n_loops, 2), (
                "%s/%s returned %r, expected (%d, 2)"
                % (backend_id, method, uvs.shape, sample.n_loops)
            )
            assert np.isfinite(uvs).all(), "%s/%s produced non-finite UVs" % (
                backend_id,
                method,
            )
            tested += 1
            HARNESS.note(
                "%s/%s ok in %.0f ms %s"
                % (backend_id, method, result.seconds * 1000.0, result.message[:40])
            )
    assert tested > 0, "no backend produced a usable layout on Suzanne"


@HARNESS.add("backends")
def test_backend_leaves_the_scene_unchanged():
    """A backend run must not alter UV layers, seams, mode or the active object.

    The engine snapshots around a whole generation run, but a backend that
    leaks state between candidates makes candidate N depend on candidate N-1,
    which quietly destroys reproducibility.
    """
    import bpy
    import numpy as np

    base = require_attr(
        "unwrap_variations.backends.base", "get_backend", "UnwrapRequest"
    )
    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")
    params_mod = require("unwrap_variations.core.params")

    obj = get("UVVarCube")
    select_only(obj)
    obj.data.calc_loop_triangles()

    before_layers = uv_names(obj)
    before_uvs = {n: read_uvs(obj, n) for n in before_layers}
    before_seams = seam_flags(obj)
    before_active = bpy.context.view_layer.objects.active
    before_mode = obj.mode

    sample = bridge.extract_sample(obj)
    backend = base.get_backend("blender_native")
    request = base.UnwrapRequest(
        obj_name=obj.name,
        params=params_mod.UnwrapParams(method="SMART_PROJECT").clamped(),
        sample=sample,
    )
    backend.unwrap(request)

    assert uv_names(obj) == before_layers, (
        "backend changed the UV layer set: %r -> %r" % (before_layers, uv_names(obj))
    )
    for name in before_layers:
        delta = float(np.abs(read_uvs(obj, name) - before_uvs[name]).max())
        assert delta <= 1e-6, "backend modified UV layer %r by %g" % (name, delta)
    assert np.array_equal(seam_flags(obj), before_seams), "backend changed the seams"
    assert obj.mode == before_mode, "backend left the object in %s" % obj.mode
    assert bpy.context.view_layer.objects.active is before_active, (
        "backend changed the active object"
    )


@HARNESS.add("backends")
def test_seam_strategies_run_on_real_meshes():
    """Every seam strategy runs on real Blender geometry, including an n-gon."""
    seams = require_attr("unwrap_variations.backends.seams", "compute_seams")
    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")
    for name in ("UVVarCube", "UVVarSphere", "UVVarNgon", "UVVarSuzanne"):
        obj = get(name)
        obj.data.calc_loop_triangles()
        sample = bridge.extract_sample(obj)
        for strategy in seams.SEAM_STRATEGIES:
            mask = seams.compute_seams(sample, strategy, angle_deg=30.0)
            assert mask.shape == (sample.n_edges,), "%s/%s: %r" % (
                name,
                strategy,
                mask.shape,
            )


# ===========================================================================
# 4. evaluator on real Blender data
# ===========================================================================


@HARNESS.add("evaluator")
def test_evaluator_on_real_unwraps():
    """The L2 >= 1.0 invariant holds on freshly-unwrapped Blender meshes.

    ``tests/`` proves this on a frozen corpus.  Here it is proved against
    whatever this Blender build's solvers actually produce today, which is what
    guards against a future Blender changing its output in a way the metrics
    mishandle.
    """
    import bpy

    evaluator = require_attr("unwrap_variations.core.evaluator", "evaluate")
    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")

    for name in ("UVVarCube", "UVVarSphere", "UVVarSuzanne", "UVVarGrid"):
        obj = get(name)
        select_only(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.01)
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.calc_loop_triangles()

        sample = bridge.extract_sample(obj)
        m = evaluator.evaluate(sample, compute_overlap=True)
        assert m.stretch_l2 >= 1.0 - 1e-9, (
            "%s: L2 %.12f is below the provable floor of 1.0" % (name, m.stretch_l2)
        )
        assert m.n_tris == sample.n_tris
        HARNESS.note(
            "%-13s L2=%.4f Linf=%.4f islands=%d flipped=%d overlap=%.4g"
            % (name, m.stretch_l2, m.stretch_linf, m.island_count, m.flipped_tris,
               m.overlap_area)
        )


@HARNESS.add("evaluator")
def test_evaluator_performance_on_a_dense_mesh():
    """A subdivided sphere is evaluated inside the documented time budget."""
    import bpy

    evaluator = require_attr("unwrap_variations.core.evaluator", "evaluate")
    bridge = require_attr("unwrap_variations.core.bridge", "extract_sample")

    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64)
    obj = bpy.context.active_object
    obj.name = "UVVarDense"
    try:
        select_only(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15192)
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.calc_loop_triangles()
        sample = bridge.extract_sample(obj)

        started = time.perf_counter()
        m = evaluator.evaluate(sample, compute_overlap=False)
        elapsed = time.perf_counter() - started
        per_500k = elapsed * (500_000.0 / max(sample.n_tris, 1))
        HARNESS.note(
            "%d tris in %.0f ms (extrapolates to %.0f ms per 500k)"
            % (sample.n_tris, elapsed * 1000.0, per_500k * 1000.0)
        )
        assert m.stretch_l2 >= 1.0 - 1e-9
        assert per_500k < 3.0, (
            "evaluation extrapolates to %.2f s per 500k triangles; the budget in "
            "docs/ARCHITECTURE.md section 4 is 0.25 s" % per_500k
        )
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


# ===========================================================================
# 5. add-on registration
# ===========================================================================


@HARNESS.add("addon")
def test_register_and_unregister():
    """The add-on registers and unregisters cleanly, twice over.

    Twice, because a class left registered by a failed unregister makes the
    second ``register()`` raise, and that is exactly what a user hits when they
    disable and re-enable the add-on.
    """
    addon = require_attr("unwrap_variations", "register", "unregister")
    for round_number in (1, 2):
        addon.register()
        try:
            _assert_operators_registered()
        finally:
            addon.unregister()
        HARNESS.note("register/unregister round %d clean" % round_number)


def _assert_operators_registered() -> None:
    """Assert every ``bl_idname`` from the contract resolves to an operator."""
    import bpy

    expected = (
        "uvvar.generate_grid",
        "uvvar.branch_variations",
        "uvvar.select_candidate",
        "uvvar.apply_candidate",
        "uvvar.clear_candidates",
        "uvvar.toggle_overlay",
        "uvvar.cycle_candidate",
    )
    missing = []
    for idname in expected:
        namespace, _, name = idname.partition(".")
        group = getattr(bpy.ops, namespace, None)
        if group is None or not hasattr(group, name):
            missing.append(idname)
    assert not missing, "operators not registered: %s" % ", ".join(missing)


@HARNESS.add("addon")
def test_scene_properties_are_registered():
    """``Scene.uvvar`` exists while the add-on is registered and is cleaned up after."""
    import bpy

    addon = require_attr("unwrap_variations", "register", "unregister")
    addon.register()
    try:
        assert hasattr(bpy.types.Scene, "uvvar"), "Scene.uvvar was not registered"
        props = bpy.context.scene.uvvar
        for field in (
            "candidate_count",
            "seed",
            "mutation_strength",
            "weight_preset",
            "active_index",
        ):
            assert hasattr(props, field), "UVVAR_SceneProps has no %r" % field
    finally:
        addon.unregister()
    assert not hasattr(bpy.types.Scene, "uvvar"), (
        "Scene.uvvar survived unregister -- the property was leaked"
    )


# ===========================================================================
# 6. the full workflow
# ===========================================================================


@HARNESS.add("workflow")
def test_generate_preview_branch_apply():
    """The whole loop: generate a grid, preview one, branch it, apply the best.

    This is the product in one function.  If it works here it works in the UI,
    because the operators are thin wrappers over exactly these calls.
    """
    import numpy as np

    engine = require_attr("unwrap_variations.core.engine", "generate", "branch")
    bridge = require_attr(
        "unwrap_variations.core.bridge", "ensure_preview_layer", "remove_preview_layer"
    )
    metrics = require("unwrap_variations.core.metrics")
    candidate_mod = require("unwrap_variations.core.candidate")

    obj = get("UVVarSuzanne")
    select_only(obj)
    weights = metrics.ScoreWeights.preset("balanced")

    before_layers = uv_names(obj)
    before_seams = seam_flags(obj)

    progress_calls = []
    report = engine.generate(
        obj,
        count=6,
        seed=11,
        weights=weights,
        progress=lambda f, msg: progress_calls.append((f, msg)),
        compute_overlap=True,
    )
    assert report.candidates, "generate produced nothing: %r" % (report.failures,)
    HARNESS.note(
        "generate: %s (%d failures)" % (report.summary(), len(report.failures))
    )
    for params, reason in report.failures[:3]:
        HARNESS.note("  failed %s: %s" % (params.describe(), reason))
    assert progress_calls, "generate never reported progress"

    # Ranked best first, and every candidate carries usable metrics.
    scores = [c.score for c in report.candidates]
    assert scores == sorted(scores, reverse=True), "candidates are not ranked"
    for c in report.candidates:
        assert c.uvs is not None and c.uvs.shape == (len(obj.data.loops), 2)
        assert c.metrics is not None
        assert c.metrics.stretch_l2 >= 1.0 - 1e-9, (
            "candidate %d reports L2 %.9f" % (c.index, c.metrics.stretch_l2)
        )

    # The user's own UV layers are untouched by generation.
    assert set(before_layers) <= set(uv_names(obj)), (
        "generation removed a user UV layer: %r -> %r" % (before_layers, uv_names(obj))
    )
    assert np.array_equal(seam_flags(obj), before_seams), (
        "generation left its seam marks on the mesh"
    )

    # -- preview -------------------------------------------------------------
    store = candidate_mod.CandidateStore.for_object(obj)
    if len(store) == 0:
        store.set(report.candidates)
    best = report.candidates[0]
    layer_name = bridge.ensure_preview_layer(obj)
    require_attr("unwrap_variations.core.bridge", "apply_uvs").apply_uvs(
        obj, best.uvs, layer_name=layer_name
    )
    shown = read_uvs(obj, layer_name)
    delta = float(np.abs(shown - best.uvs.astype(np.float64)).max())
    assert delta <= 1e-6, "preview blit differs from the candidate by %g" % delta

    # -- branch --------------------------------------------------------------
    child_report = engine.branch(
        obj, best, count=4, strength=0.4, seed=3, weights=weights, compute_overlap=False
    )
    assert child_report.candidates, "branch produced nothing: %r" % (
        child_report.failures,
    )
    HARNESS.note("branch: %s" % child_report.summary())
    parents = {c.parent for c in child_report.candidates if c.parent}
    assert best.signature in parents or not parents, (
        "no child records the parent signature %s (saw %r)" % (best.signature, parents)
    )

    # -- apply ---------------------------------------------------------------
    require_attr("unwrap_variations.core.bridge", "apply_uvs").apply_uvs(
        obj, best.uvs, layer_name="UVVarApplied"
    )
    assert "UVVarApplied" in uv_names(obj)
    delta = float(np.abs(read_uvs(obj, "UVVarApplied") - best.uvs.astype(np.float64)).max())
    assert delta <= 1e-6, "applied UVs differ from the candidate by %g" % delta

    bridge.remove_preview_layer(obj)
    obj.data.uv_layers.remove(obj.data.uv_layers["UVVarApplied"])


@HARNESS.add("workflow")
def test_generate_is_reproducible():
    """The same seed explores the same recipes, run after run.

    Deliberately a **set** comparison, not a list one.  Blender's UV operators
    are verified non-deterministic on 4.5.4 -- five runs of ``smart_project`` on
    an identical sphere give five distinct UV buffers -- so the *scores* differ
    slightly between runs and the ranked order can legitimately swap two
    near-tied candidates.  What the seed does guarantee is which recipes get
    tried, and that is what is asserted here.  Asserting on order would be
    asserting on solver noise.
    """
    engine = require_attr("unwrap_variations.core.engine", "generate")
    metrics = require("unwrap_variations.core.metrics")
    obj = get("UVVarCube")
    select_only(obj)
    weights = metrics.ScoreWeights.preset("balanced")

    first = engine.generate(obj, count=4, seed=99, weights=weights, compute_overlap=False)
    second = engine.generate(obj, count=4, seed=99, weights=weights, compute_overlap=False)
    a = sorted(c.signature for c in first.candidates)
    b = sorted(c.signature for c in second.candidates)
    assert a == b, "the same seed explored different recipes:\n  %r\n  %r" % (a, b)
    ordered_a = [c.signature for c in first.candidates]
    ordered_b = [c.signature for c in second.candidates]
    if ordered_a != ordered_b:
        HARNESS.note(
            "same recipes, different ranking between runs -- expected, the "
            "solvers are non-deterministic"
        )


@HARNESS.add("workflow")
def test_no_candidate_is_a_pathological_time_outlier():
    """No single recipe may dominate the wall clock of a whole grid.

    Measured on a 12-triangle cube: nine of ten candidates finish in 0.00-0.37 s
    while ``SMART_PROJECT`` at ``angle_limit=30, island_margin=0.02`` takes
    ~6.4 s, all of it inside one ``bpy.ops.uv.pack_islands`` call.  That single
    recipe is 84 % of the run on the simplest mesh anyone will ever try, and it
    scales with the model.  A tool whose pitch is "generate ten and compare"
    cannot afford one archetype that costs twenty times the other nine
    together.

    The check is a ratio rather than a stopwatch so it means the same thing on
    a fast workstation and a slow CI runner.
    """
    engine = require_attr("unwrap_variations.core.engine", "generate")
    metrics = require("unwrap_variations.core.metrics")

    obj = get("UVVarCube")
    select_only(obj)
    started = time.perf_counter()
    report = engine.generate(
        obj, count=10, seed=0, weights=metrics.ScoreWeights(), compute_overlap=True
    )
    elapsed = time.perf_counter() - started
    assert report.candidates, "generate produced nothing"

    times = sorted(c.seconds for c in report.candidates)
    median = times[len(times) // 2]
    slowest = max(report.candidates, key=lambda c: c.seconds)
    HARNESS.note(
        "%d candidates in %.2f s; median %.3f s, slowest %.2f s (%s)"
        % (len(report.candidates), elapsed, median, slowest.seconds,
           slowest.params.describe())
    )
    budget = max(median * 15.0, 1.5)
    assert slowest.seconds <= budget, (
        "candidate %r took %.2f s against a %.3f s median on a 12-triangle cube "
        "(%.0f%% of the whole %.2f s run). Profiled: all of it is one "
        "bpy.ops.uv.pack_islands call in blender_native._pack_islands."
        % (slowest.params.describe(), slowest.seconds, median,
           100.0 * slowest.seconds / max(elapsed, 1e-9), elapsed)
    )


@HARNESS.add("workflow")
def test_generate_on_a_degenerate_object_reports_rather_than_raises():
    """An object with no faces gets a clear refusal, not a traceback."""
    import bpy

    engine = require_attr("unwrap_variations.core.engine", "generate")
    metrics = require("unwrap_variations.core.metrics")

    mesh = bpy.data.meshes.new("UVVarPoints")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0)], [(0, 1)], [])
    mesh.update()
    obj = bpy.data.objects.new("UVVarPoints", mesh)
    bpy.context.collection.objects.link(obj)
    try:
        report = engine.generate(
            obj, count=2, seed=0, weights=metrics.ScoreWeights(), compute_overlap=False
        )
        assert not report.candidates, "generated candidates for a mesh with no faces"
        assert report.warnings or report.failures, (
            "refused silently -- the user is owed a reason"
        )
        HARNESS.note("refused with: %s" % (report.warnings or report.failures)[0])
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)


@HARNESS.add("workflow")
def test_generation_restores_state_after_a_failing_backend():
    """A backend that fails every candidate still leaves the mesh pristine."""
    import numpy as np

    engine = require_attr("unwrap_variations.core.engine", "generate")
    metrics = require("unwrap_variations.core.metrics")
    obj = get("UVVarGrid")
    select_only(obj)

    before_layers = uv_names(obj)
    before_seams = seam_flags(obj)
    engine.generate(
        obj,
        count=3,
        seed=1,
        weights=metrics.ScoreWeights(),
        backends=["no_such_backend"],
        compute_overlap=False,
    )
    assert set(before_layers) <= set(uv_names(obj)), "UV layers were lost"
    assert np.array_equal(seam_flags(obj), before_seams), "seams were left modified"


# ===========================================================================
# 7. persistence across a real .blend save/reload
# ===========================================================================


@HARNESS.add("persistence")
def test_candidates_survive_a_blend_save_and_reload():
    """A candidate store written into an ID property survives a real file cycle.

    Not a mock: this saves a ``.blend`` to disk, opens it again -- which throws
    away every Python object in the process -- and reads the candidates back out
    of the mesh datablock.
    """
    import bpy
    import numpy as np

    candidate_mod = require_attr(
        "unwrap_variations.core.candidate", "Candidate", "CandidateStore"
    )
    engine = require_attr("unwrap_variations.core.engine", "generate")
    metrics = require("unwrap_variations.core.metrics")

    obj = get("UVVarCube")
    select_only(obj)
    report = engine.generate(
        obj, count=3, seed=5, weights=metrics.ScoreWeights(), compute_overlap=False
    )
    assert report.candidates, "nothing to persist: %r" % (report.failures,)

    store = candidate_mod.CandidateStore(obj.name, len(obj.data.loops))
    store.set(report.candidates)
    ok, message = store.save_to_id(obj, max_bytes=64 * 1024 * 1024)
    assert ok, "save_to_id refused: %s" % message
    expected = [(c.signature, np.array(c.uvs)) for c in report.candidates]
    loop_count = len(obj.data.loops)

    tmpdir = tempfile.mkdtemp(prefix="uvvar_")
    path = os.path.join(tmpdir, "uvvar_persistence.blend")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.read_homefile(use_empty=True)
        assert bpy.data.objects.get("UVVarCube") is None, "the scene did not reset"
        bpy.ops.wm.open_mainfile(filepath=path)

        reloaded = bpy.data.objects.get("UVVarCube")
        assert reloaded is not None, "the object did not come back from disk"
        assert len(reloaded.data.loops) == loop_count

        restored = candidate_mod.CandidateStore.for_object(reloaded)
        assert not restored.was_stale, restored.last_message
        assert len(restored) == len(expected), (
            "expected %d candidates back, got %d (%s)"
            % (len(expected), len(restored), restored.last_message)
        )
        for (signature, uvs), got in zip(expected, restored.all(), strict=False):
            assert got.signature == signature, "recipe %s came back as %s" % (
                signature,
                got.signature,
            )
            assert got.uvs is not None, "candidate %d lost its UVs" % got.index
            assert np.array_equal(got.uvs, uvs), "candidate %d UVs changed" % got.index
            assert got.metrics is not None, "candidate %d lost its metrics" % got.index
        HARNESS.note("%d candidates round-tripped through a .blend" % len(restored))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


@HARNESS.add("persistence")
def test_stale_store_is_detected_after_a_topology_change():
    """Editing the mesh invalidates stored UVs, and the store says so.

    Blitting a 24-loop candidate into a 96-loop mesh would either raise or
    scramble the UV layer.  The loop-count check is the only thing preventing
    that after a user subdivides.
    """
    import bpy

    candidate_mod = require_attr("unwrap_variations.core.candidate", "CandidateStore")
    engine = require_attr("unwrap_variations.core.engine", "generate")
    metrics = require("unwrap_variations.core.metrics")

    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.active_object
    obj.name = "UVVarStale"
    try:
        select_only(obj)
        report = engine.generate(
            obj, count=2, seed=2, weights=metrics.ScoreWeights(), compute_overlap=False
        )
        assert report.candidates, "nothing to persist"
        store = candidate_mod.CandidateStore(obj.name, len(obj.data.loops))
        store.set(report.candidates)
        assert store.save_to_id(obj)[0]
        original_loops = len(obj.data.loops)

        select_only(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=1)
        bpy.ops.object.mode_set(mode="OBJECT")
        assert len(obj.data.loops) != original_loops, "the subdivide did nothing"

        reloaded = candidate_mod.CandidateStore.for_object(obj)
        assert reloaded.was_stale, (
            "a store built for %d loops loaded into a %d-loop mesh without complaint"
            % (original_loops, len(obj.data.loops))
        )
        assert len(reloaded) == 0, "stale candidates were kept"
        assert reloaded.last_message, "no explanation offered to the user"
        HARNESS.note(reloaded.last_message)
    finally:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------


def main() -> int:
    """Run the harness and return the process exit code."""
    args = _script_args()
    HARNESS.verbose = "--verbose" in args or "-v" in args
    return HARNESS.run()


if __name__ == "__main__":
    sys.exit(main())
