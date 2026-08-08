# Developing UnwrapVariations locally

Every command below was run and verified against **Blender 4.5.4 LTS** and
**Blender 5.1.2** on Windows. Paths use Windows conventions; the macOS/Linux
equivalents are noted where they differ.

---

## 1. Get the source

```bash
git clone https://github.com/eric-rolph/unwrap-variations.git
cd unwrap-variations
```

The add-on package is the inner `unwrap_variations/` directory -- the one
containing `blender_manifest.toml`. The repository root is *not* the add-on.

---

## 2. Run the pure-Python tests first (no Blender needed)

The core is deliberately `bpy`-free, so most of the logic is testable in plain
CPython. This is the fastest feedback loop by a wide margin -- use it.

```bash
python -m pip install pytest numpy ruff
python -m pytest tests/ -v
ruff check .
```

---

## 3. Live-editing inside Blender (the good workflow)

Do **not** copy files into Blender's extension directory each time you edit --
you will lose changes and confuse yourself. Link the source directory in place
so Blender loads the files you are actually editing.

Blender's per-user extension directory is:

| Version | Path |
| --- | --- |
| 4.5 | `%APPDATA%\Blender Foundation\Blender\4.5\extensions\user_default` |
| 5.1 | `%APPDATA%\Blender Foundation\Blender\5.1\extensions\user_default` |

(macOS: `~/Library/Application Support/Blender/<ver>/extensions/user_default`,
Linux: `~/.config/blender/<ver>/extensions/user_default`.)

### Windows -- directory junction (no admin rights required)

```bat
mklink /J "%APPDATA%\Blender Foundation\Blender\4.5\extensions\user_default\unwrap_variations" "C:\path\to\unwrap-variations\unwrap_variations"
```

### macOS / Linux -- symlink

```bash
ln -s "$PWD/unwrap_variations" ~/.config/blender/4.5/extensions/user_default/unwrap_variations
```

Then launch Blender and enable it in **Edit > Preferences > Add-ons**, searching
for "Unwrap Variations". As an extension its module name is
`bl_ext.user_default.unwrap_variations`.

After editing source, either restart Blender or toggle the add-on off and on --
`__init__.py` implements the `importlib.reload` pattern so a toggle re-imports
the submodules rather than serving stale bytecode.

> If you are on the Microsoft Store build of Blender, note that its config lives
> in **both** the real `%APPDATA%\Blender Foundation` tree and the MSIX
> `LocalCache\Roaming` tree. The real Roaming path is the one that loads.
> The Store build's `blender.exe` is not directly executable due to its ACL --
> use the `blender-launcher.exe` alias in
> `%LOCALAPPDATA%\Microsoft\WindowsApps`, which **detaches and gives you no
> stdout**. For scripted work, prefer a portable Blender build.

---

## 4. Build a distributable extension .zip

Blender builds and validates extensions itself; there is no setuptools step.

```bash
blender --command extension validate unwrap_variations
blender --command extension build --source-dir unwrap_variations --output-dir dist
```

`validate` is strict and worth running before every release -- it rejects
manifest problems that would otherwise only surface at install time (a `tagline`
over 64 characters or ending in a period is the classic one).

Or use the helper, which does both and names the archive from the manifest:

```bash
python tools/build_extension.py
```

Install the built archive headlessly to check it really works:

```bash
blender --command extension install-file --repo user_default --enable dist/unwrap_variations-<version>.zip
```

---

## 5. Headless integration tests

The pure tests cannot exercise `bpy`. This harness does, and it is what CI runs:

```bash
blender -b --factory-startup --python tools/run_blender_tests.py
```

It exits non-zero on failure. Always pass `--factory-startup` so your personal
add-ons and preferences cannot influence the result -- without it, another
installed add-on can make a passing test lie.

Test against both supported versions before opening a pull request:

```bash
"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" -b --factory-startup --python tools/run_blender_tests.py
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b --factory-startup --python tools/run_blender_tests.py
```

---

## 6. Debugging tips specific to this add-on

- **Print to a file, not stdout, when using a detached launcher.** Store builds
  and `blender-launcher.exe` swallow stdout. Have your `--python` script write
  its own log file and poll it.
- **`bpy.ops.uv.*` operators need the right context.** They act on the *selected*
  geometry in **Edit** mode. Headless, that means an explicit
  `bpy.ops.object.mode_set(mode='EDIT')` plus
  `bpy.ops.mesh.select_all(action='SELECT')`. Missing either silently produces
  an empty or unchanged UV layout rather than an error.
- **`uv.pack_islands` behaves differently across versions.** Verified: it
  returns `CANCELLED` in 4.5 headless but `FINISHED` in 5.1. Anything relying on
  packing must tolerate the cancellation. All such differences belong in
  `utils/compat.py`, not scattered through the codebase.
- **A mesh cannot hold more than 8 UV layers.** `uv_layers.new()` returns `None`
  past that -- it does not raise. This is exactly why candidates are stored as
  numpy arrays and blitted into a single scratch layer instead of becoming
  layers of their own.
- **`smart_project`'s `angle_limit` is in radians**, while this add-on stores
  angles in degrees. The conversion happens once, in
  `backends/blender_native.py`, and must not be duplicated elsewhere.
- **Console encoding on Windows is cp1252.** Non-ASCII in a print statement
  raises `UnicodeEncodeError` and can abort a headless run. Keep source and
  output ASCII-only.

---

## 7. Project layout

```
unwrap_variations/
  __init__.py          add-on registration, bl_info, reload support
  properties.py        PropertyGroups mirrored into the UI
  preferences.py       AddonPreferences
  core/
    meshdata.py        bpy-free mesh interchange (UVMeshSample)
    params.py          parameter space, grid sampling, mutation
    metrics.py         UVMetrics, ScoreWeights, scoring, Pareto front
    evaluator.py       vectorised distortion measurement
    candidate.py       Candidate + CandidateStore persistence
    bridge.py          the ONLY bpy <-> core boundary
    engine.py          generation and branching coordinator
  backends/
    base.py            UnwrapBackend ABC and registry
    blender_native.py  bpy.ops.uv.* driver
    seams.py           seam heuristics incl. the dual-graph solver
    xatlas_cli.py      optional external backend
  operators/           the bpy.ops.uvvar.* operators
  ui/                  N-panel, UV editor panel, GPU stretch overlay
  utils/               logging and version-compat shims
```

`docs/ARCHITECTURE.md` is the binding module contract and records the probed
Blender API ground truth. Read it before changing any interface.
