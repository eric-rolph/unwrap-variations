"""Blender-free unit tests for the UnwrapVariations core.

The whole suite runs under plain ``pytest`` with nothing but the standard
library, numpy and pytest installed -- no Blender, no ``bpy``.  That is a design
constraint of the add-on (see ``docs/ARCHITECTURE.md`` section 1.2), not a
convenience: the modules that decide *what* an unwrap is worth are pure numpy,
so they can be checked against an independent oracle in milliseconds instead of
minutes.

Anything that genuinely needs a Blender datablock -- extract/apply, snapshot and
restore, operator registration -- lives in ``tools/run_blender_tests.py`` and is
run headlessly against real Blender builds.
"""
