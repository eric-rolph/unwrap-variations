"""Optional xatlas backend: in-process Python module, or an external binary.

`xatlas <https://github.com/jpcy/xatlas>`_ does its own chart segmentation and
packing, which often beats Blender's packer on dense scanned or CAD meshes.  It
is **not** shipped inside Blender (see ARCHITECTURE section 0), so this backend
is genuinely optional and reports precisely why it cannot run.

Two paths are supported, tried in this order:

1. **In-process module** -- ``import xatlas``.  Fast, exact, and the parameter
   set maps onto xatlas's own ``ChartOptions`` / ``PackOptions`` so the search
   actually explores something.  Usually unavailable *inside* Blender, because
   Blender ships its own Python without the wheel; the reason string says so.
2. **External binary** -- a command line tool driven over temporary Wavefront
   ``.obj`` files.  The binary comes from the ``UVVAR_XATLAS_BIN`` environment
   variable, from the add-on preference of the same name, or from ``PATH``.

The delicate part is path 2: the returned UVs must land on the *right loops*.
The exported OBJ carries one vertex per mesh vertex and one face per triangle in
:attr:`UVMeshSample.tri_verts` order, but a re-exporting tool is free to split
vertices at chart boundaries, renumber them and reorder faces.  So the importer
never trusts ordering blindly -- see :func:`_map_faces` -- it verifies a
face-for-face correspondence geometrically and falls back to centroid matching
through a spatial hash when the tool has shuffled things.  Corner order within
each face is recovered by picking the cheapest of the six corner permutations.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from types import ModuleType
from typing import Any, ClassVar, Iterable

import numpy as np

from ..core.meshdata import UVMeshSample
from ..core.params import METHODS, UnwrapParams
from .base import UnwrapBackend, UnwrapRequest, UnwrapResult, _Notes, register_backend

__all__ = ["XatlasBackend", "ENV_BIN", "ENV_ARGS", "PREF_ATTR"]

#: Environment variable naming the external xatlas executable.
ENV_BIN = "UVVAR_XATLAS_BIN"

#: Optional environment variable overriding the command line template.  Use
#: ``{input}`` and ``{output}`` placeholders; both are absolute paths inside a
#: private temporary directory that is also the child process's working
#: directory.  Defaults to ``"{input}" "{output}"``.
ENV_ARGS = "UVVAR_XATLAS_ARGS"

#: Name of the add-on preference holding the binary path, if the preferences
#: module defines one.  Looked up defensively; its absence is not an error.
PREF_ATTR = "xatlas_bin"

#: Executable names searched on ``PATH`` when nothing else is configured.
_BINARY_CANDIDATES = ("xatlas", "xatlas_cmd", "xatlas-cmd", "xatlasw")

#: Relative tolerance (against the bounding-box diagonal) for deciding that an
#: imported face corresponds to an exported one.  Generous, because OBJ writers
#: routinely emit only six significant digits.
_MATCH_TOL = 1.0e-4

#: The six ways three output corners can map onto three input corners.
_PERMUTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
)


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def _probe_module() -> tuple[ModuleType | None, str]:
    """Return ``(module, reason)`` for the in-process ``xatlas`` package."""
    try:
        import xatlas  # noqa: PLC0415 - optional dependency, probed on demand
    except BaseException as exc:  # noqa: BLE001
        return None, "xatlas Python module not importable (%s)" % type(exc).__name__
    if not hasattr(xatlas, "parametrize"):
        return None, "xatlas module present but has no parametrize()"
    return xatlas, "xatlas module %s" % getattr(xatlas, "__version__", "?")


def _preference_binary() -> str:
    """Return the binary path from the add-on preferences, or an empty string.

    Deliberately tolerant: the preferences class belongs to another module and
    may not define :data:`PREF_ATTR` -- or exist at all -- and the add-on is
    registered under different keys depending on whether it was installed as a
    legacy add-on (``unwrap_variations``) or as an extension
    (``bl_ext.<repo>.unwrap_variations``).
    """
    try:
        import bpy  # noqa: PLC0415
    except BaseException:  # noqa: BLE001
        return ""
    # ``__package__`` is "unwrap_variations.backends" or
    # "bl_ext.user_default.unwrap_variations.backends"; strip the trailing
    # ".backends" to get the add-on's registration key.
    package = (__package__ or "").rsplit(".", 1)[0]
    leaf = package.rsplit(".", 1)[-1]
    try:
        addons = bpy.context.preferences.addons
        entry = addons.get(package)
        if entry is None:
            for key in addons:
                if key == leaf or key.rsplit(".", 1)[-1] == leaf:
                    entry = addons[key]
                    break
        prefs = getattr(entry, "preferences", None)
        return str(getattr(prefs, PREF_ATTR, "") or "").strip()
    except BaseException:  # noqa: BLE001
        return ""


def _resolve_binary() -> tuple[str, str]:
    """Return ``(path, reason)`` for the external xatlas executable.

    ``path`` is empty when nothing usable was found; ``reason`` then explains
    every place that was searched, which is what the user needs in order to fix
    it.  A configured-but-missing path is reported as an error rather than
    silently falling through to ``PATH``, because a typo in a preference should
    be visible.
    """
    configured = os.environ.get(ENV_BIN, "").strip().strip('"')
    source = ENV_BIN
    if not configured:
        configured = _preference_binary()
        source = "add-on preference '%s'" % PREF_ATTR
    if configured:
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(configured)))
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded, "configured via %s" % source
        found = shutil.which(configured)
        if found:
            return found, "resolved %r from %s on PATH" % (configured, source)
        return "", "%s points at %r, which is not an executable file" % (source, configured)

    for name in _BINARY_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found, "found %r on PATH" % name
    return "", "no xatlas binary: set %s or the '%s' add-on preference (searched PATH for %s)" % (
        ENV_BIN,
        PREF_ATTR,
        ", ".join(_BINARY_CANDIDATES),
    )


# ---------------------------------------------------------------------------
# Wavefront OBJ round-trip
# ---------------------------------------------------------------------------


def _write_obj(path: str, sample: UVMeshSample) -> None:
    """Write ``sample`` as a triangle-only OBJ.

    One ``v`` line per mesh vertex and one ``f`` line per triangle, in
    :attr:`UVMeshSample.tri_verts` order, at nine significant digits so the
    positions survive the text round trip well inside :data:`_MATCH_TOL`.
    """
    verts = np.asarray(sample.verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(sample.tri_verts, dtype=np.int64).reshape(-1, 3) + 1
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("# UnwrapVariations xatlas export\n")
        handle.write("o %s\n" % (_ascii(sample.name) or "mesh"))
        np.savetxt(handle, verts, fmt="v %.9g %.9g %.9g")
        np.savetxt(handle, tris, fmt="f %d %d %d")


def _ascii(text: str) -> str:
    """Return ``text`` with every non-ASCII, non-printable character dropped."""
    return "".join(c for c in str(text) if 32 <= ord(c) < 127 and c not in " \t")


def _parse_index(token: str, count: int) -> int:
    """Convert one OBJ index token to a 0-based index, honouring negatives."""
    value = int(token)
    return value - 1 if value > 0 else count + value


def _read_obj(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse an OBJ into ``(positions, uvs, face_pos, face_uv)``.

    Returns:
        ``positions`` ``(N, 3)`` float64, ``uvs`` ``(M, 2)`` float64,
        ``face_pos`` ``(F, 3)`` int64 position indices and ``face_uv``
        ``(F, 3)`` int64 texture indices (``-1`` where a corner carried none).
        Polygons with more than three corners are fan-triangulated so that a
        tool which re-quadded the mesh still round-trips.
    """
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    face_pos: list[tuple[int, int, int]] = []
    face_uv: list[tuple[int, int, int]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt" and len(parts) >= 3:
                texcoords.append((float(parts[1]), float(parts[2])))
            elif tag == "f" and len(parts) >= 4:
                corner_pos: list[int] = []
                corner_uv: list[int] = []
                for token in parts[1:]:
                    fields = token.split("/")
                    corner_pos.append(_parse_index(fields[0], len(positions)))
                    if len(fields) > 1 and fields[1]:
                        corner_uv.append(_parse_index(fields[1], len(texcoords)))
                    else:
                        corner_uv.append(-1)
                for i in range(1, len(corner_pos) - 1):
                    face_pos.append((corner_pos[0], corner_pos[i], corner_pos[i + 1]))
                    face_uv.append((corner_uv[0], corner_uv[i], corner_uv[i + 1]))

    return (
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(texcoords, dtype=np.float64).reshape(-1, 2),
        np.asarray(face_pos, dtype=np.int64).reshape(-1, 3),
        np.asarray(face_uv, dtype=np.int64).reshape(-1, 3),
    )


def _spatial_match(source: np.ndarray, target: np.ndarray, cell: float) -> np.ndarray:
    """Match every row of ``target`` to its nearest unused row of ``source``.

    A uniform grid keeps this O(N); the true partner is always in the same or an
    adjacent cell because the positional error is text-rounding noise, orders of
    magnitude below ``cell``.

    Args:
        source: ``(N, 3)`` reference points (the exported triangle centroids).
        target: ``(M, 3)`` query points (the imported face centroids).
        cell: Grid spacing.

    Returns:
        ``(M,)`` int64 indices into ``source``, ``-1`` where nothing matched.
    """
    table: dict[tuple[int, int, int], list[int]] = {}
    keys = np.floor(source / cell).astype(np.int64)
    for index in range(keys.shape[0]):
        table.setdefault((int(keys[index, 0]), int(keys[index, 1]), int(keys[index, 2])),
                         []).append(index)

    out_keys = np.floor(target / cell).astype(np.int64)
    mapping = np.full(target.shape[0], -1, dtype=np.int64)
    used = np.zeros(source.shape[0], dtype=bool)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for face in range(target.shape[0]):
        kx, ky, kz = int(out_keys[face, 0]), int(out_keys[face, 1]), int(out_keys[face, 2])
        best, best_distance = -1, float("inf")
        point = target[face]
        for dx, dy, dz in offsets:
            for index in table.get((kx + dx, ky + dy, kz + dz), ()):
                if used[index]:
                    continue
                delta = source[index] - point
                distance = float(delta.dot(delta))
                if distance < best_distance:
                    best_distance, best = distance, index
        if best >= 0:
            mapping[face] = best
            used[best] = True
    return mapping


def _corner_permutation(in_corners: np.ndarray, out_corners: np.ndarray) -> np.ndarray:
    """Return, per face, the permutation taking output corners to input corners.

    Args:
        in_corners: ``(F, 3, 3)`` exported triangle corner positions.
        out_corners: ``(F, 3, 3)`` imported face corner positions, already
            aligned face-for-face with ``in_corners``.

    Returns:
        ``(F, 3)`` int array; entry ``[f, j]`` is the *output* corner that
        matches input corner ``j`` of face ``f``.
    """
    n_faces = in_corners.shape[0]
    if n_faces == 0:
        return np.zeros((0, 3), dtype=np.int64)
    costs = np.empty((n_faces, len(_PERMUTATIONS)), dtype=np.float64)
    for slot, perm in enumerate(_PERMUTATIONS):
        diff = out_corners[:, perm, :] - in_corners
        costs[:, slot] = np.einsum("fij,fij->f", diff, diff)
    best = np.argmin(costs, axis=1)
    table = np.asarray(_PERMUTATIONS, dtype=np.int64)
    return table[best]


def _map_faces(
    sample: UVMeshSample,
    positions: np.ndarray,
    face_pos: np.ndarray,
    notes: _Notes,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Correspond imported faces to the sample's triangles.

    Returns:
        ``(face_for_tri, corner_for_tri)`` where ``face_for_tri[t]`` is the
        imported face carrying triangle ``t`` (``-1`` when unmatched) and
        ``corner_for_tri[t, j]`` is the imported corner carrying corner ``j``.
        ``None`` when the import is unusable.
    """
    n_tris = sample.n_tris
    if n_tris == 0 or face_pos.shape[0] == 0 or positions.shape[0] == 0:
        notes.add("imported OBJ has no usable faces")
        return None
    if int(face_pos.max()) >= positions.shape[0] or int(face_pos.min()) < 0:
        notes.add("imported OBJ has out-of-range face indices")
        return None

    in_corners = sample.verts[sample.tri_verts]              # (T, 3, 3)
    out_corners = positions[face_pos]                        # (F, 3, 3)
    in_centroids = in_corners.mean(axis=1)
    out_centroids = out_corners.mean(axis=1)

    extent = sample.verts.max(axis=0) - sample.verts.min(axis=0)
    diagonal = float(np.linalg.norm(extent)) or 1.0
    tolerance = diagonal * _MATCH_TOL

    face_for_tri = np.full(n_tris, -1, dtype=np.int64)
    identity = False
    if face_pos.shape[0] == n_tris:
        drift = float(np.abs(out_centroids - in_centroids).max())
        if drift <= tolerance:
            face_for_tri = np.arange(n_tris, dtype=np.int64)
            identity = True

    if not identity:
        cell = max(diagonal / 64.0, 1e-9)
        tri_for_face = _spatial_match(in_centroids, out_centroids, cell)
        matched = tri_for_face >= 0
        face_for_tri[tri_for_face[matched]] = np.nonzero(matched)[0]
        missing = int((face_for_tri < 0).sum())
        notes.add("faces were reordered by the tool; matched %d/%d by centroid",
                  n_tris - missing, n_tris)
        if missing == n_tris:
            notes.add("no imported face could be matched to the mesh")
            return None

    valid = face_for_tri >= 0
    corner_for_tri = np.zeros((n_tris, 3), dtype=np.int64)
    if valid.any():
        corner_for_tri[valid] = _corner_permutation(
            in_corners[valid], out_corners[face_for_tri[valid]]
        )
    return face_for_tri, corner_for_tri


def _uvs_from_obj(sample: UVMeshSample, path: str, notes: _Notes) -> np.ndarray | None:
    """Read an OBJ written by an external tool and project its UVs onto loops."""
    positions, texcoords, face_pos, face_uv = _read_obj(path)
    if texcoords.shape[0] == 0:
        notes.add("the tool's output OBJ contains no 'vt' texture coordinates")
        return None
    mapping = _map_faces(sample, positions, face_pos, notes)
    if mapping is None:
        return None
    face_for_tri, corner_for_tri = mapping

    loop_uvs = np.zeros((sample.n_loops, 2), dtype=np.float32)
    written = np.zeros(sample.n_loops, dtype=bool)
    conflicts = 0
    unresolved = 0

    tri_loops = np.asarray(sample.tri_loops, dtype=np.int64)
    for tri in range(sample.n_tris):
        face = int(face_for_tri[tri])
        if face < 0:
            continue
        for corner in range(3):
            uv_index = int(face_uv[face, corner_for_tri[tri, corner]])
            if uv_index < 0 or uv_index >= texcoords.shape[0]:
                unresolved += 1
                continue
            loop = int(tri_loops[tri, corner])
            if loop < 0 or loop >= sample.n_loops:
                continue
            value = texcoords[uv_index]
            if written[loop] and not np.allclose(loop_uvs[loop], value, atol=1e-6):
                conflicts += 1
            loop_uvs[loop] = value
            written[loop] = True

    if unresolved:
        notes.add("%d corners had no texture index", unresolved)
    if conflicts:
        notes.add("%d loops were assigned conflicting UVs (charts split a shared corner)",
                  conflicts)
    if not written.any():
        # Every corner failed to resolve.  Returning a zeroed array here would
        # look like a valid (if terrible) unwrap; it is really a failed import.
        notes.add("no loop received a UV from the tool's output")
        return None
    unwritten = int((~written).sum())
    if unwritten:
        notes.add("%d loops received no UV and were zeroed", unwritten)
    return loop_uvs


# ---------------------------------------------------------------------------
# Parameter mapping onto xatlas's own knobs
# ---------------------------------------------------------------------------


def _lerp(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    """Map ``value`` from ``[low, high]`` onto ``[out_low, out_high]``, clamped."""
    if high <= low:
        return out_low
    t = (float(value) - low) / (high - low)
    t = min(max(t, 0.0), 1.0)
    return out_low + t * (out_high - out_low)


def _chart_pack_options(xatlas: ModuleType, params: UnwrapParams) -> tuple[Any, Any, str]:
    """Translate :class:`UnwrapParams` into xatlas ``ChartOptions``/``PackOptions``.

    xatlas has no notion of Blender's solvers or seam strategies, so the mapping
    is a deliberate reinterpretation rather than an equivalence:

    * the angle that drives chart splitting (``angle_limit_deg`` for Smart
      Project recipes, ``seam_angle_deg`` otherwise) becomes
      ``normal_deviation_weight`` -- a larger tolerated angle means normals
      should matter *less*;
    * ``curvature_threshold`` becomes ``max_cost``, the chart growth budget:
      a higher threshold yields fewer, larger charts;
    * ``iterations`` becomes ``max_iterations`` of chart refinement;
    * ``island_margin`` becomes ``padding``, in texels of the chosen resolution;
    * ``pack_rotate`` / ``rotate_method`` / ``pack_shape_method`` drive
      ``rotate_charts``, ``rotate_charts_to_axis``, ``bruteForce`` and
      ``blockAlign``.

    Returns:
        ``(chart_options, pack_options, description)``.
    """
    chart = xatlas.ChartOptions()
    pack = xatlas.PackOptions()

    if params.method == "SMART_PROJECT":
        split_angle = float(params.angle_limit_deg)
        angle_low, angle_high = 1.0, 89.0
    else:
        split_angle = float(params.seam_angle_deg)
        angle_low, angle_high = 5.0, 120.0

    chart.normal_deviation_weight = _lerp(split_angle, angle_low, angle_high, 8.0, 0.25)
    chart.max_cost = _lerp(float(params.curvature_threshold), 0.02, 0.95, 1.2, 8.0)
    chart.max_iterations = int(min(max(int(params.iterations), 1), 8))
    chart.use_input_mesh_uvs = False

    resolution = 1024
    extra = params.extra if isinstance(params.extra, dict) else {}
    try:
        resolution = int(extra.get("xatlas_resolution", resolution))
    except (TypeError, ValueError):
        pass
    resolution = int(min(max(resolution, 64), 8192))
    pack.resolution = resolution
    pack.padding = int(min(max(round(float(params.island_margin) * resolution), 0), 64))
    pack.rotate_charts = bool(params.pack_rotate)
    pack.rotate_charts_to_axis = str(params.rotate_method).startswith("AXIS_ALIGNED")
    pack.bruteForce = params.pack_shape_method == "CONCAVE"
    pack.blockAlign = params.pack_shape_method == "AABB"
    pack.create_image = False

    description = (
        "normal_deviation_weight=%.2f max_cost=%.2f max_iterations=%d "
        "padding=%d resolution=%d rotate=%s"
        % (chart.normal_deviation_weight, chart.max_cost, chart.max_iterations,
           pack.padding, pack.resolution, pack.rotate_charts)
    )
    return chart, pack, description


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


@register_backend
class XatlasBackend(UnwrapBackend):
    """Unwraps with xatlas, in-process when possible and out-of-process otherwise."""

    id: ClassVar[str] = "xatlas"
    label: ClassVar[str] = "xatlas (optional)"
    requires_bpy: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return ``(usable, reason)``, naming the path that would be taken.

        The reason is written for a user staring at a greyed-out button: it says
        which of the two paths is live, or -- when neither is -- exactly which
        environment variable or preference to set.
        """
        module, module_reason = _probe_module()
        if module is not None:
            return True, "in-process %s" % module_reason
        binary, binary_reason = _resolve_binary()
        if binary:
            return True, "external binary (%s): %s" % (binary_reason, binary)
        return False, "%s; %s" % (module_reason, binary_reason)

    def supported_methods(self) -> tuple[str, ...]:
        """Return every method, since xatlas reinterprets rather than rejects them.

        xatlas always uses its own segmentation, so ``method`` selects a
        *flavour* of chart options (see :func:`_chart_pack_options`) instead of a
        solver.  Refusing the projection methods outright would leave most of a
        generated grid unusable on this backend for no benefit.
        """
        return tuple(METHODS)

    def unwrap(self, req: UnwrapRequest) -> UnwrapResult:
        """Generate one UV layout with xatlas.

        Args:
            req: Object name, recipe and mesh snapshot.  Only ``sample`` and
                ``params`` are read -- this backend never touches the scene, so
                the "leave it as you found it" rule is satisfied trivially.

        Returns:
            An :class:`UnwrapResult`; never raises.
        """
        start = time.perf_counter()
        notes = _Notes()
        sample = req.sample

        if sample.n_tris == 0 or sample.n_loops == 0:
            return UnwrapResult.failure(
                "mesh sample has no triangles", backend=self.id,
                seconds=time.perf_counter() - start)

        module, module_reason = _probe_module()
        try:
            if module is not None:
                uvs = self._unwrap_in_process(module, req, notes)
            else:
                notes.add(module_reason)
                uvs = self._unwrap_subprocess(req, notes)
        except BaseException as exc:  # noqa: BLE001 - containment, per the contract
            return UnwrapResult.failure(
                "%s: %s" % (type(exc).__name__, exc), backend=self.id,
                seconds=time.perf_counter() - start)

        seconds = time.perf_counter() - start
        if uvs is None:
            return UnwrapResult.failure(
                notes.text() or "xatlas produced no UVs", backend=self.id, seconds=seconds)
        return UnwrapResult.success(
            uvs, backend=self.id, seconds=seconds, message=notes.text())

    # -- path 1: in-process ---------------------------------------------------

    def _unwrap_in_process(
        self, xatlas: ModuleType, req: UnwrapRequest, notes: _Notes
    ) -> np.ndarray | None:
        """Run xatlas through its Python bindings.

        The bindings preserve both face order and corner order -- ``vmapping``
        composed with the output indices reproduces the input triangles exactly
        -- so the loop mapping is direct.  That invariant is checked rather than
        assumed; if a future version breaks it, the geometric matcher takes over.
        """
        sample = req.sample
        positions = np.ascontiguousarray(sample.verts, dtype=np.float32)
        indices = np.ascontiguousarray(sample.tri_verts, dtype=np.uint32)

        atlas = getattr(xatlas, "Atlas", None)
        vmapping = out_indices = out_uvs = None
        if atlas is not None and hasattr(xatlas, "ChartOptions"):
            try:
                chart, pack, description = _chart_pack_options(xatlas, req.params)
                instance = atlas()
                instance.add_mesh(positions, indices)
                instance.generate(chart_options=chart, pack_options=pack, verbose=False)
                vmapping, out_indices, out_uvs = instance.get_mesh(0)
                notes.add("xatlas atlas (%s)", description)
            except BaseException as exc:  # noqa: BLE001 - fall back to the simple call
                notes.add("xatlas option path failed (%s: %s); used defaults",
                          type(exc).__name__, exc)
                vmapping = None
        if vmapping is None:
            vmapping, out_indices, out_uvs = xatlas.parametrize(positions, indices)
            notes.add("xatlas parametrize() defaults")

        out_uvs = np.asarray(out_uvs, dtype=np.float64).reshape(-1, 2)
        out_indices = np.asarray(out_indices, dtype=np.int64).reshape(-1, 3)
        vmapping = np.asarray(vmapping, dtype=np.int64).reshape(-1)
        if out_indices.shape[0] == 0 or out_uvs.shape[0] == 0:
            notes.add("xatlas returned an empty atlas")
            return None

        # xatlas normalises to 0..1 in recent builds and to texel space in older
        # ones; normalise defensively so downstream metrics see the unit square.
        span = float(np.abs(out_uvs).max()) if out_uvs.size else 0.0
        if span > 2.0:
            out_uvs = out_uvs / span
            notes.add("UVs were in texel space; normalised by %.1f", span)

        loop_uvs = np.zeros((sample.n_loops, 2), dtype=np.float32)
        tri_loops = np.asarray(sample.tri_loops, dtype=np.int64)
        n_shared = min(out_indices.shape[0], sample.n_tris)
        if out_indices.shape[0] != sample.n_tris:
            notes.add("xatlas returned %d faces for %d triangles",
                      out_indices.shape[0], sample.n_tris)

        origin = vmapping[out_indices[:n_shared]]
        expected = np.asarray(sample.tri_verts, dtype=np.int64)[:n_shared]
        if not np.array_equal(origin, expected):
            if np.array_equal(np.sort(origin, axis=1), np.sort(expected, axis=1)):
                notes.add("xatlas rotated face corners; realigned per corner")
                order = np.argsort(origin, axis=1)
                rank = np.argsort(np.argsort(expected, axis=1), axis=1)
                picked = np.take_along_axis(order, rank, axis=1)
                out_indices = np.take_along_axis(out_indices[:n_shared], picked, axis=1)
            else:
                notes.add("xatlas reordered faces; cannot map UVs back to loops safely")
                return None

        flat_loops = tri_loops[:n_shared].reshape(-1)
        flat_uv = out_uvs[out_indices[:n_shared].reshape(-1)]
        valid = (flat_loops >= 0) & (flat_loops < sample.n_loops)
        loop_uvs[flat_loops[valid]] = flat_uv[valid].astype(np.float32)
        return loop_uvs

    # -- path 2: external binary ---------------------------------------------

    def _unwrap_subprocess(self, req: UnwrapRequest, notes: _Notes) -> np.ndarray | None:
        """Drive an external xatlas executable over temporary OBJ files.

        The child runs in a private temp directory with a wall-clock timeout, so
        a hung or chatty tool can neither block Blender's main thread forever nor
        litter the user's disk.  The directory is removed in a ``finally``.
        """
        binary, reason = _resolve_binary()
        if not binary:
            notes.add(reason)
            return None

        workdir = tempfile.mkdtemp(prefix="uvvar_xatlas_",
                                   dir=os.environ.get("UVVAR_TMPDIR") or None)
        input_path = os.path.join(workdir, "input.obj")
        output_path = os.path.join(workdir, "output.obj")
        try:
            _write_obj(input_path, req.sample)
            template = os.environ.get(ENV_ARGS) or '"{input}" "{output}"'
            command = [binary] + _split_args(template, input_path, output_path)

            timeout = max(1.0, float(req.timeout_s))
            try:
                completed = subprocess.run(  # noqa: S603 - path is user-configured on purpose
                    command,
                    cwd=workdir,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                notes.add("xatlas binary timed out after %.1fs", timeout)
                return None
            except OSError as exc:
                notes.add("could not run %r (%s)", binary, exc)
                return None

            if completed.returncode != 0:
                notes.add("xatlas exited %d: %s", completed.returncode,
                          _tail(completed.stderr or completed.stdout))
                # Some tools return non-zero yet still write usable output, so
                # fall through and look for it before giving up.

            produced = output_path if os.path.isfile(output_path) else _find_output(
                workdir, input_path
            )
            if produced is None:
                notes.add("xatlas produced no OBJ output (%s)",
                          _tail(completed.stdout or completed.stderr))
                return None
            if produced != output_path:
                notes.add("used the tool's own output file %r", os.path.basename(produced))

            notes.add("xatlas binary (%s)", reason)
            return _uvs_from_obj(req.sample, produced, notes)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _split_args(template: str, input_path: str, output_path: str) -> list[str]:
    """Expand a command-line template into argv, honouring simple quoting.

    The template is tokenised **before** the placeholders are substituted, so a
    temp path containing spaces cannot split itself across two arguments.

    Quoting on Windows needs care.  ``shlex.split(posix=True)`` eats backslashes
    -- it would turn ``C:\\Users\\x`` into ``C:Usersx`` -- while
    ``posix=False`` preserves backslashes but *keeps the quote characters
    inside the tokens*, which would hand the child process an argument that
    literally begins with a double quote.  So: split non-posix, then strip one
    matched pair of surrounding quotes per token.
    """
    import shlex  # noqa: PLC0415 - only needed on the subprocess path

    try:
        tokens = shlex.split(template, posix=False)
    except ValueError:
        tokens = ["{input}", "{output}"]

    argv: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        argv.append(token.replace("{input}", input_path).replace("{output}", output_path))
    return argv


def _find_output(workdir: str, input_path: str) -> str | None:
    """Return the newest OBJ in ``workdir`` other than the input, if any.

    The reference ``xatlas_cmd`` example writes a fixed ``example_output.obj``
    into its working directory and ignores any output argument, so scanning is
    what makes real tools work rather than only well-behaved hypothetical ones.
    """
    best: tuple[float, str] | None = None
    try:
        entries: Iterable[str] = os.listdir(workdir)
    except OSError:
        return None
    for entry in entries:
        path = os.path.join(workdir, entry)
        if path == input_path or not entry.lower().endswith(".obj"):
            continue
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            continue
        if best is None or stamp > best[0]:
            best = (stamp, path)
    return best[1] if best else None


def _tail(data: bytes | str | None, limit: int = 200) -> str:
    """Return the last ``limit`` printable ASCII characters of process output."""
    if not data:
        return "<no output>"
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    text = " ".join(text.split())
    text = "".join(c if 32 <= ord(c) < 127 else "?" for c in text)
    return text[-limit:] if len(text) > limit else text
