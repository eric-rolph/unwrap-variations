#!/usr/bin/env python3
"""Packaging entry point for UnwrapVariations -- a shim, not a setuptools build.

UnwrapVariations is a **Blender add-on**, not a Python distribution. It is never
installed with ``pip``: it runs inside Blender's own interpreter and is shipped
as an extension archive described by ``unwrap_variations/blender_manifest.toml``.

Canonical packaging commands::

    python tools/build_extension.py
    blender --command extension build --source-dir unwrap_variations --output-dir dist
    blender --command extension validate unwrap_variations

This file exists because "setup.py / extension manifest" is named in the project
spec, and because ``python setup.py build`` is muscle memory. It simply forwards
to ``tools/build_extension.py`` and refuses -- loudly and with an explanation --
the setuptools commands that would otherwise fail in a confusing way.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD_SCRIPT = REPO_ROOT / "tools" / "build_extension.py"

# Commands that mean "build the add-on archive"; everything after them is passed
# through to tools/build_extension.py untouched.
BUILD_ALIASES = frozenset({"build", "build_extension", "bdist", "dist", "zip"})

# Commands that only make sense for a pip-installable distribution.
SETUPTOOLS_COMMANDS = frozenset(
    {
        "bdist_egg",
        "bdist_wheel",
        "develop",
        "easy_install",
        "egg_info",
        "install",
        "install_egg_info",
        "sdist",
        "test",
    }
)

USAGE = """\
UnwrapVariations is a Blender add-on and is not pip-installable.

Usage:
  python setup.py                 Build the extension archive into dist/
  python setup.py build [ARGS]    Same, forwarding ARGS to tools/build_extension.py
  python setup.py --help          Show this message

Equivalent, and preferred:
  python tools/build_extension.py [ARGS]

Install the built archive in Blender 4.2+ with Edit > Preferences > Add-ons >
Install from Disk, or headlessly with:
  blender -b --factory-startup --python-expr "import bpy; \\
      bpy.ops.extensions.package_install_files(filepath='dist/<archive>.zip', \\
      repo='user_default', enable_on_install=True)"
"""


def main(argv: list[str]) -> int:
    """Dispatch ``setup.py`` arguments to the real builder."""
    if argv and argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    if argv and argv[0] in SETUPTOOLS_COMMANDS:
        print(USAGE, file=sys.stderr)
        print(
            f'ERROR: "{argv[0]}" is a Python-packaging command and does not apply here.\n'
            "       There is no wheel, no sdist and no site-packages install:\n"
            "       Blender loads the add-on from its own extensions directory.",
            file=sys.stderr,
        )
        return 2

    forwarded = argv[1:] if (argv and argv[0] in BUILD_ALIASES) else argv
    if not BUILD_SCRIPT.is_file():
        print(f"ERROR: missing {BUILD_SCRIPT}", file=sys.stderr)
        return 1

    sys.argv = [str(BUILD_SCRIPT), *forwarded]
    # `build_extension.py` ends in `sys.exit(main())`, so the SystemExit it
    # raises carries the real exit code straight through this shim.
    runpy.run_path(str(BUILD_SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
