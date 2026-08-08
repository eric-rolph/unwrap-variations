#!/usr/bin/env python3
"""Build distributable archives for the UnwrapVariations Blender add-on.

Two archive shapes exist and they are not interchangeable:

* **Extension** (Blender 4.2+) -- ``blender_manifest.toml`` and ``__init__.py``
  sit at the *root* of the archive. Installed with
  ``bpy.ops.extensions.package_install_files()`` or drag-and-drop.
* **Legacy add-on** (Blender 4.x without the extensions system) -- a single
  top-level ``unwrap_variations/`` directory holds the package. Installed with
  ``bpy.ops.preferences.addon_install()``.

The extension archive is built by Blender itself when a Blender executable can
be found (``blender --command extension build``), because Blender's builder is
the authority on the archive layout. A pure-Python fallback reproduces the same
layout so the build works on machines with no Blender installed, e.g. a CI
linter job.

Both archives are staged from a temporary directory so that the repository-root
``LICENSE`` can be folded into the package: GPL distribution requires the
license to travel with the code, and Blender's builder cannot reach outside the
source directory.

Usage::

    python tools/build_extension.py
    python tools/build_extension.py --blender "C:/Program Files/Blender/blender.exe"
    python tools/build_extension.py --output-dir dist --no-legacy
    python tools/build_extension.py --python-only      # skip Blender entirely

Exit code is 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    print("ERROR: Python 3.11 or newer is required (tomllib is missing).")
    raise SystemExit(1) from None

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "unwrap_variations"
PACKAGE_DIR = REPO_ROOT / PACKAGE_NAME
MANIFEST_NAME = "blender_manifest.toml"
MANIFEST_PATH = PACKAGE_DIR / MANIFEST_NAME
LICENSE_PATH = REPO_ROOT / "LICENSE"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist"

# Always dropped, whatever the manifest says. Keeps the fallback path honest
# even if `[build]` is ever removed from the manifest.
ALWAYS_EXCLUDE_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", ".ruff_cache"})
ALWAYS_EXCLUDE_GLOBS = ("*.pyc", "*.pyo", "*.pyd", ".DS_Store", "Thumbs.db")

# Candidate Blender executables probed when neither --blender nor $BLENDER is set.
BLENDER_PROBE_GLOBS = (
    "C:/Program Files/Blender Foundation/Blender */blender.exe",
    "C:/Users/*/Applications/Blender/*/blender.exe",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/Applications/Blender.app/Contents/MacOS/Blender",
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def read_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Parse ``blender_manifest.toml`` and sanity-check the fields we rely on."""
    if not path.is_file():
        raise SystemExit(f"ERROR: manifest not found: {path}")
    with path.open("rb") as handle:
        manifest = tomllib.load(handle)

    for key in ("id", "version", "schema_version", "type"):
        if not manifest.get(key):
            raise SystemExit(f'ERROR: manifest is missing "{key}"')
    if manifest["id"] != PACKAGE_NAME:
        # The installed extension directory is named after `id`; a mismatch
        # would break every `from unwrap_variations...` import at runtime.
        raise SystemExit(
            'ERROR: manifest id "{}" must match the package directory "{}"'.format(
                manifest["id"], PACKAGE_NAME
            )
        )
    return manifest


def archive_stem(manifest: dict) -> str:
    """Return ``{id}-{version}``, matching Blender's own default archive name."""
    return "{}-{}".format(manifest["id"], manifest["version"])


# ---------------------------------------------------------------------------
# Exclusion matching (a small subset of Blender's gitignore-style matcher)
# ---------------------------------------------------------------------------


def _compile_excludes(manifest: dict) -> list[tuple[bool, re.Pattern[str]]]:
    """Compile ``[build] paths_exclude_pattern`` into (is_dir_only, regex) pairs.

    Blender implements full gitignore semantics; this covers the patterns the
    manifest actually uses (``name/``, ``*.ext``, plain names). Blender's own
    builder is used whenever it is available, so this only has to be right for
    the fallback.
    """
    patterns = manifest.get("build", {}).get("paths_exclude_pattern", [])
    compiled: list[tuple[bool, re.Pattern[str]]] = []
    for pattern in patterns:
        dir_only = pattern.endswith("/")
        body = pattern.rstrip("/").lstrip("/")
        compiled.append((dir_only, re.compile(fnmatch.translate(body))))
    return compiled


def _is_excluded(rel_posix: str, is_dir: bool, compiled) -> bool:
    """Test one path (relative, forward slashes) against the exclusion rules."""
    parts = rel_posix.split("/")
    if is_dir and parts[-1] in ALWAYS_EXCLUDE_DIRS:
        return True
    if any(part in ALWAYS_EXCLUDE_DIRS for part in parts[:-1]):
        return True
    if not is_dir and any(fnmatch.fnmatch(parts[-1], glob) for glob in ALWAYS_EXCLUDE_GLOBS):
        return True
    for dir_only, regex in compiled:
        if dir_only and not is_dir:
            continue
        if any(regex.match(part) for part in parts):
            return True
    return False


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def stage_package(dest: Path, manifest: dict, *, verbose: bool = False) -> list[str]:
    """Copy the add-on package (plus LICENSE) into ``dest``. Returns relative paths."""
    compiled = _compile_excludes(manifest)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for src in sorted(PACKAGE_DIR.rglob("*")):
        rel = src.relative_to(PACKAGE_DIR).as_posix()
        if _is_excluded(rel, src.is_dir(), compiled):
            continue
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied.append(rel)

    if LICENSE_PATH.is_file():
        shutil.copy2(LICENSE_PATH, dest / "LICENSE")
        copied.append("LICENSE")
    else:
        print(f"WARNING: {LICENSE_PATH} not found; the archive will ship without it")

    if MANIFEST_NAME not in copied:
        raise SystemExit(f"ERROR: {MANIFEST_NAME} was excluded from the staged package")
    if "__init__.py" not in copied:
        raise SystemExit("ERROR: __init__.py is missing from the staged package")

    if verbose:
        for rel in sorted(copied):
            print(f"  staged {rel}")
    return sorted(copied)


# ---------------------------------------------------------------------------
# Blender discovery / invocation
# ---------------------------------------------------------------------------


def find_blender(explicit: str | None) -> Path | None:
    """Locate a Blender executable: --blender, then $BLENDER, then PATH, then probes."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("BLENDER"):
        candidates.append(os.environ["BLENDER"])
    found_on_path = shutil.which("blender")
    if found_on_path:
        candidates.append(found_on_path)

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
        if explicit and candidate == explicit:
            raise SystemExit(f"ERROR: --blender path does not exist: {explicit}")

    for pattern in BLENDER_PROBE_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    return None


def run_blender(blender: Path, args: list[str]) -> tuple[int, str]:
    """Run Blender with ``--factory-startup`` and return (returncode, combined output)."""
    command = [str(blender), "--factory-startup", "--command", "extension", *args]
    print("  $ {}".format(" ".join(command)))
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def validate_with_blender(blender: Path, target: Path) -> bool:
    """Run ``extension validate`` against a source directory or an archive."""
    code, output = run_blender(blender, ["validate", str(target)])
    for line in output.splitlines():
        print(f"    {line}")
    return code == 0


# ---------------------------------------------------------------------------
# Archive builders
# ---------------------------------------------------------------------------


def build_extension_zip(
    staged: Path,
    output_dir: Path,
    manifest: dict,
    blender: Path | None,
) -> Path:
    """Build the 4.2+ extension archive (manifest and __init__.py at the root)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{archive_stem(manifest)}.zip"
    if out_path.exists():
        out_path.unlink()

    if blender is not None:
        code, output = run_blender(
            blender,
            [
                "build",
                "--source-dir",
                str(staged),
                "--output-filepath",
                str(out_path),
            ],
        )
        for line in output.splitlines():
            print(f"    {line}")
        if code != 0 or not out_path.is_file():
            raise SystemExit("ERROR: blender failed to build the extension archive")
        return out_path

    print("  (no Blender found; using the pure-Python fallback builder)")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src in sorted(p for p in staged.rglob("*") if p.is_file()):
            zf.write(src, src.relative_to(staged).as_posix())
    return out_path


def build_legacy_zip(staged: Path, output_dir: Path, manifest: dict) -> Path:
    """Build the pre-4.2 add-on archive (a single top-level package directory)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{archive_stem(manifest)}-legacy.zip"
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src in sorted(p for p in staged.rglob("*") if p.is_file()):
            arcname = f"{PACKAGE_NAME}/{src.relative_to(staged).as_posix()}"
            zf.write(src, arcname)
    return out_path


def describe_zip(path: Path) -> None:
    """Print an archive's size and its first few entries."""
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
    print(f"  {path.name} ({path.stat().st_size} bytes, {len(names)} files)")
    for name in names[:12]:
        print(f"    {name}")
    if len(names) > 12:
        print(f"    ... {len(names) - 12} more")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the UnwrapVariations extension archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Path to a Blender executable (default: $BLENDER, then PATH, then known locations).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write archives into (default: dist/).",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Skip the pre-4.2 legacy add-on archive.",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Never invoke Blender; use the pure-Python builder and skip validation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest and exit without building anything.",
    )
    parser.add_argument("--verbose", action="store_true", help="List every staged file.")
    args = parser.parse_args(argv)

    manifest = read_manifest()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    blender = None if args.python_only else find_blender(args.blender)
    print("UnwrapVariations packaging")
    print(f"  package : {PACKAGE_DIR}")
    print("  id      : {}".format(manifest["id"]))
    print("  version : {}".format(manifest["version"]))
    print("  blender : {}".format(blender if blender else "<not found>"))
    print(f"  output  : {output_dir}")
    print()

    if blender is not None:
        print("Validating the manifest in place")
        if not validate_with_blender(blender, PACKAGE_DIR):
            print("FAILED: manifest validation")
            return 1
        print()
    elif args.validate_only:
        print("ERROR: --validate-only needs a Blender executable.")
        return 1

    if args.validate_only:
        print("OK: manifest is valid.")
        return 0

    with tempfile.TemporaryDirectory(prefix="uvvar-build-") as tmp:
        staged = Path(tmp) / PACKAGE_NAME
        print("Staging package")
        files = stage_package(staged, manifest, verbose=args.verbose)
        print(f"  {len(files)} files staged")
        print()

        print("Building extension archive")
        ext_zip = build_extension_zip(staged, output_dir, manifest, blender)
        describe_zip(ext_zip)
        print()

        legacy_zip = None
        if not args.no_legacy:
            print("Building legacy add-on archive")
            legacy_zip = build_legacy_zip(staged, output_dir, manifest)
            describe_zip(legacy_zip)
            print()

    if blender is not None:
        print("Validating the built archive")
        if not validate_with_blender(blender, ext_zip):
            print("FAILED: the built archive did not validate")
            return 1
        print()

    print("OK")
    print(f"  extension : {ext_zip}")
    if legacy_zip is not None:
        print(f"  legacy    : {legacy_zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
