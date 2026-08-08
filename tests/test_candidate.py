"""Tests for ``unwrap_variations.core.candidate`` -- storage and persistence.

``CandidateStore`` touches a Blender datablock but only by duck typing, so the
whole of it is testable here with a stub object.  Two things are worth breaking
a build over.

**The stale-store check.**  A persisted candidate is a flat array of UVs indexed
by loop.  If the user subdivides the mesh between saving and loading, every one
of those arrays now means something different, and blitting one into the mesh
writes garbage.  The store records the loop count it was built against and must
refuse to load when it no longer matches.

**Contained failure.**  A corrupt, truncated or foreign ID property must produce
a clean "no candidates" plus an explanation, never a traceback out of a file
load.  The user's ``.blend`` opening is not negotiable.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from tests.conftest import import_or_skip
from unwrap_variations.core.params import UnwrapParams, sample_grid

candidate_mod = import_or_skip("unwrap_variations.core.candidate")

Candidate = candidate_mod.Candidate
CandidateStore = candidate_mod.CandidateStore
STORE_MAGIC = candidate_mod.STORE_MAGIC
STORE_VERSION = candidate_mod.STORE_VERSION

metrics_mod = import_or_skip("unwrap_variations.core.metrics")
UVMetrics = metrics_mod.UVMetrics
ScoreWeights = metrics_mod.ScoreWeights


# ---------------------------------------------------------------------------
# A duck-typed stand-in for a Blender object.  Everything the store touches is
# `obj.name`, `obj.data`, `len(mesh.loops)` and mesh ID-property access.
# ---------------------------------------------------------------------------


class FakeMesh(dict):
    """A dict that also has a ``loops`` sequence, like ``bpy.types.Mesh``."""

    def __init__(self, loop_count: int) -> None:
        """Create a mesh stub reporting ``loop_count`` loops."""
        super().__init__()
        self.loops = [None] * int(loop_count)


class FakeObject:
    """An object stub wrapping a :class:`FakeMesh`."""

    def __init__(self, name: str, loop_count: int) -> None:
        """Create an object named ``name`` whose mesh has ``loop_count`` loops."""
        self.name = name
        self.data = FakeMesh(loop_count)


def make_candidate(index: int = 0, *, loops: int = 12, ok: bool = True, score: float = 50.0):
    """Build one populated candidate with deterministic UVs."""
    uvs = np.arange(loops * 2, dtype=np.float32).reshape(loops, 2) * 0.01
    return Candidate(
        index=index,
        params=sample_grid(index + 1, seed=1)[index],
        uvs=uvs if ok else None,
        metrics=UVMetrics(n_tris=100, stretch_l2=1.0 + index * 0.01, packing_efficiency=0.7),
        score=score,
        ok=ok,
        message="" if ok else "backend exploded",
        generation=0,
    )


# ---------------------------------------------------------------------------
# Candidate.
# ---------------------------------------------------------------------------


def test_uvs_are_normalised_to_contiguous_float32():
    """UVs are coerced once, at construction, so preview is a straight memcpy.

    A float64 or strided array would make ``foreach_set`` fall back to a
    per-element conversion loop and destroy the instant-preview promise.
    """
    raw = np.arange(24, dtype=np.float64).reshape(12, 2)[::2]  # strided, float64
    c = Candidate(index=0, params=UnwrapParams(), uvs=raw, metrics=None)
    assert c.uvs.dtype == np.float32
    assert c.uvs.flags["C_CONTIGUOUS"]
    assert c.uvs.shape == (6, 2)


def test_flat_uv_buffer_is_reshaped():
    """A flat ``(2L,)`` buffer -- what ``foreach_get`` yields -- becomes ``(L, 2)``."""
    c = Candidate(index=0, params=UnwrapParams(), uvs=np.zeros(20, dtype=np.float32), metrics=None)
    assert c.uvs.shape == (10, 2)


def test_none_uvs_stay_none():
    """A failed candidate keeps ``None`` rather than an empty array."""
    c = Candidate(index=0, params=UnwrapParams(), uvs=None, metrics=None, ok=False)
    assert c.uvs is None
    assert c.n_loops == 0


def test_malformed_uvs_become_a_failed_candidate():
    """Garbage in the UV slot degrades to ``None``; it never raises."""
    c = Candidate(index=0, params=UnwrapParams(), uvs="not an array", metrics=None)
    assert c.uvs is None


def test_signature_and_label():
    """Identity and the UI row label both come off the params."""
    c = make_candidate(3)
    assert c.signature == c.params.signature()
    assert len(c.signature) == 12
    label = c.label()
    label.encode("ascii")
    assert label.startswith("03")


def test_signature_never_raises():
    """Identity is read on every redraw, so it must not be able to throw."""

    class Broken:
        def signature(self):
            raise RuntimeError("boom")

        def to_dict(self):
            return {}

        def describe(self):
            return "broken"

    c = Candidate(index=0, params=Broken(), uvs=None, metrics=None)
    assert isinstance(c.signature, str)
    assert len(c.signature) == 12


def test_header_dict_round_trip():
    """A candidate survives ``header_dict`` -> JSON -> ``from_header_dict``."""
    c = make_candidate(2)
    data = json.loads(json.dumps(c.header_dict()))
    again = Candidate.from_header_dict(data, c.uvs)
    assert again.index == c.index
    assert again.params == c.params
    assert again.score == pytest.approx(c.score)
    assert again.ok == c.ok
    assert again.generation == c.generation
    assert again.metrics == c.metrics
    np.testing.assert_array_equal(again.uvs, c.uvs)


def test_header_dict_excludes_the_uv_payload():
    """The UVs stay out of the JSON header -- as text they inflate roughly 6x."""
    data = make_candidate(0, loops=1000).header_dict()
    assert "uvs" not in data
    assert len(json.dumps(data)) < 4000


def test_from_header_dict_tolerates_a_future_blend():
    """Unknown keys and missing keys are both survivable."""
    c = Candidate.from_header_dict({"future": 1, "index": 4})
    assert c.index == 4
    assert c.uvs is None
    assert Candidate.from_header_dict({}).index == 0


# ---------------------------------------------------------------------------
# CandidateStore -- container behaviour.
# ---------------------------------------------------------------------------


def test_set_add_get_all_clear():
    """The basic container operations, including index renumbering."""
    store = CandidateStore("Cube", 12)
    assert len(store) == 0
    assert not store
    assert store.get(0) is None

    store.set([make_candidate(9), make_candidate(9)])
    assert len(store) == 2
    assert [c.index for c in store.all()] == [0, 1]

    store.add(make_candidate(99))
    assert store.get(2).index == 2
    assert len(store) == 3
    assert bool(store)

    store.clear()
    assert len(store) == 0
    assert store.all() == []


def test_get_out_of_range_returns_none():
    """A stale active index must not raise out of a draw callback."""
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    for index in (-1, 1, 999, None, "x", 1.5):
        assert store.get(index) in (None, store.all()[0])
    assert store.get(5) is None
    assert store.get(None) is None


def test_iteration_is_in_index_order():
    """Iterating the store yields candidates in index order."""
    store = CandidateStore("Cube", 12)
    store.extend([make_candidate(i) for i in range(4)])
    assert [c.index for c in store] == [0, 1, 2, 3]


def test_ranked_sorts_best_first_and_rescores():
    """Ranking re-scores from the stored metrics under the current weights.

    The user can change the weight preset at any time, and the list has to
    follow immediately rather than showing the score from whatever preset
    happened to be active during generation.
    """
    store = CandidateStore("Cube", 12)
    good = make_candidate(0)
    good.metrics = UVMetrics(n_tris=100, stretch_l2=1.01, packing_efficiency=0.9)
    good.score = 0.0  # deliberately stale
    bad = make_candidate(1)
    bad.metrics = UVMetrics(n_tris=100, stretch_l2=3.0, packing_efficiency=0.2, flipped_tris=50)
    bad.score = 99.0  # deliberately stale and wrong
    store.set([bad, good])

    order = store.ranked(ScoreWeights())
    assert order[0] is good, "ranking trusted the stale cached score"
    assert order[0].score > order[1].score
    assert store.best(ScoreWeights()) is good


def test_ranked_puts_failures_last():
    """A failed candidate is kept for the record but never displaces a real one."""
    store = CandidateStore("Cube", 12)
    failed = make_candidate(0, ok=False)
    failed.metrics = None
    real = make_candidate(1)
    store.set([failed, real])
    order = store.ranked(ScoreWeights())
    assert order[0] is real
    assert order[-1] is failed


def test_ranked_is_stable_for_ties():
    """Equally good candidates keep their original order."""
    store = CandidateStore("Cube", 12)
    cands = [make_candidate(i) for i in range(4)]
    for c in cands:
        c.metrics = UVMetrics(n_tris=100, stretch_l2=1.1, packing_efficiency=0.5)
    store.set(cands)
    assert [c.index for c in store.ranked(ScoreWeights())] == [0, 1, 2, 3]


def test_best_of_an_empty_store_is_none():
    """No candidates, no winner -- and no exception."""
    assert CandidateStore("Cube", 12).best(ScoreWeights()) is None


def test_payload_bytes_counts_only_real_uvs():
    """The size estimate skips failed candidates, which carry no array."""
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0, loops=100))
    store.add(make_candidate(1, ok=False))
    assert store.payload_bytes() == 100 * 2 * 4


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip():
    """A full store survives the trip into an ID property and back out."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.extend([make_candidate(i) for i in range(3)])
    ok, message = store.save_to_id(obj, max_bytes=1 << 20)
    assert ok, message
    assert CandidateStore.ID_KEY in obj.data

    restored = CandidateStore("Cube", 12)
    assert restored.load_from_id(obj)
    assert len(restored) == 3
    for before, after in zip(store.all(), restored.all()):
        assert after.params == before.params
        assert after.metrics == before.metrics
        assert after.score == pytest.approx(before.score)
        np.testing.assert_array_equal(after.uvs, before.uvs)


def test_for_object_loads_in_one_step():
    """``for_object`` builds the store and populates it from the datablock."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    assert store.save_to_id(obj)[0]

    loaded = CandidateStore.for_object(obj)
    assert loaded.obj_name == "Cube"
    assert loaded.loop_count == 12
    assert len(loaded) == 1


def test_blob_format_is_magic_version_and_zlib():
    """The persisted layout is exactly what the contract documents.

    The magic and version sit *outside* the compressed stream so a foreign or
    corrupt property is rejected before zlib ever sees it.
    """
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    store.save_to_id(obj)
    blob = bytes(obj.data[CandidateStore.ID_KEY])

    assert blob[: len(STORE_MAGIC)] == STORE_MAGIC
    (version,) = struct.unpack_from("<I", blob, len(STORE_MAGIC))
    assert version == STORE_VERSION
    inner = zlib.decompress(blob[len(STORE_MAGIC) + 4 :])
    (header_len,) = struct.unpack_from("<I", inner, 0)
    header = json.loads(inner[4 : 4 + header_len].decode("utf-8"))
    assert header["loop_count"] == 12
    assert header["n_candidates"] == 1
    # The float32 payload follows the header, uncompressed inside the stream.
    assert len(inner) - 4 - header_len == 12 * 2 * 4


# -- store invalidation ------------------------------------------------------


def test_stale_store_is_detected_by_loop_count():
    """A topology change invalidates every stored UV array.

    This is the check that stops the add-on writing 12 loops' worth of UVs into
    a mesh that now has 48 -- either an exception or, far worse, a silently
    scrambled UV layer.
    """
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.extend([make_candidate(i) for i in range(2)])
    assert store.save_to_id(obj)[0]

    # The user subdivides: the loop count changes underneath the store.
    obj.data.loops = [None] * 48

    reloaded = CandidateStore("Cube", 48)
    assert reloaded.load_from_id(obj) is False
    assert reloaded.was_stale is True
    assert len(reloaded) == 0
    assert "12" in reloaded.last_message and "48" in reloaded.last_message


def test_stale_store_is_discarded_from_the_mesh():
    """By default a stale blob is deleted so it neither returns nor bloats the file."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    store.save_to_id(obj)
    obj.data.loops = [None] * 13

    CandidateStore("Cube", 13).load_from_id(obj)
    assert CandidateStore.ID_KEY not in obj.data


def test_stale_store_can_be_inspected_without_discarding():
    """``discard_stale=False`` is the read-only path a ``poll()`` can use."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    store.save_to_id(obj)
    obj.data.loops = [None] * 13

    CandidateStore("Cube", 13).load_from_id(obj, discard_stale=False)
    assert CandidateStore.ID_KEY in obj.data


def test_stored_loop_count_is_cheap_and_correct():
    """A panel can ask "is this stale?" without decompressing megabytes of UVs."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    store.save_to_id(obj)
    assert CandidateStore.stored_loop_count(obj) == 12
    assert CandidateStore.stored_loop_count(FakeObject("Empty", 5)) is None
    assert CandidateStore.stored_loop_count(object()) is None


# -- budget ------------------------------------------------------------------


def test_over_budget_stores_metadata_only():
    """An oversized exploration persists its metadata, never silently bloats the file."""
    obj = FakeObject("Cube", 5000)
    store = CandidateStore("Cube", 5000)
    store.extend([make_candidate(i, loops=5000) for i in range(4)])
    ok, message = store.save_to_id(obj, max_bytes=1024)
    assert ok is False
    assert "budget" in message.lower()

    restored = CandidateStore("Cube", 5000)
    assert restored.load_from_id(obj)
    assert len(restored) == 4
    assert all(c.uvs is None for c in restored.all()), (
        "the UV payload was supposed to be dropped when over budget"
    )
    # The recipes and scores survive, so the user can regenerate.
    assert all(c.params is not None for c in restored.all())


def test_absurdly_small_budget_persists_nothing():
    """When even the metadata does not fit, nothing is written and it is said so."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.extend([make_candidate(i) for i in range(3)])
    ok, message = store.save_to_id(obj, max_bytes=8)
    assert ok is False
    assert message
    assert CandidateStore.ID_KEY not in obj.data


def test_save_with_no_mesh_reports_rather_than_raises():
    """An object with no mesh data is a clear message, not a traceback."""

    class NoData:
        name = "Empty"
        data = None

    ok, message = CandidateStore("Empty", 0).save_to_id(NoData())
    assert ok is False
    assert "mesh" in message.lower()


# -- corrupt input -----------------------------------------------------------


def test_missing_blob_is_not_an_error():
    """An object that was never explored simply has no candidates."""
    store = CandidateStore("Cube", 12)
    assert store.load_from_id(FakeObject("Cube", 12)) is False
    assert store.was_stale is False


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"short",
        b"WRONGMAGIC" + b"\x00" * 20,
        STORE_MAGIC + struct.pack("<I", STORE_VERSION) + b"not zlib at all",
        STORE_MAGIC + struct.pack("<I", STORE_VERSION) + zlib.compress(b"\x00"),
        STORE_MAGIC + struct.pack("<I", STORE_VERSION + 99) + zlib.compress(b"x" * 40),
    ],
)
def test_corrupt_blob_is_rejected_cleanly(blob):
    """Truncated, foreign, non-zlib and future-version blobs all fail politely.

    A ``.blend`` failing to open because an add-on choked on its own ID property
    is the worst possible outcome, so every one of these paths returns ``False``
    with an explanation.
    """
    obj = FakeObject("Cube", 12)
    obj.data[CandidateStore.ID_KEY] = blob
    store = CandidateStore("Cube", 12)
    assert store.load_from_id(obj) is False
    assert len(store) == 0
    assert isinstance(store.last_message, str)


def test_non_bytes_property_is_rejected():
    """Someone else's ID property under the same key is ignored, not decoded."""
    obj = FakeObject("Cube", 12)
    obj.data[CandidateStore.ID_KEY] = {"not": "our blob"}
    store = CandidateStore("Cube", 12)
    assert store.load_from_id(obj) is False
    assert store.last_message


def test_clear_id_removes_the_property():
    """Clearing drops the persisted blob entirely."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    store.save_to_id(obj)
    store.clear_id(obj)
    assert CandidateStore.ID_KEY not in obj.data
    store.clear_id(obj)  # idempotent


def test_failed_candidates_survive_persistence():
    """A backend failure is part of the record: the user should see what was tried."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    store.add(make_candidate(0))
    store.add(make_candidate(1, ok=False))
    assert store.save_to_id(obj)[0]

    restored = CandidateStore("Cube", 12)
    restored.load_from_id(obj)
    assert len(restored) == 2
    assert restored.get(1).ok is False
    assert restored.get(1).uvs is None
    assert "exploded" in restored.get(1).message


def test_empty_store_round_trips():
    """Persisting nothing and loading it back yields nothing, cleanly."""
    obj = FakeObject("Cube", 12)
    store = CandidateStore("Cube", 12)
    ok, _ = store.save_to_id(obj)
    assert ok
    restored = CandidateStore("Cube", 12)
    assert restored.load_from_id(obj)
    assert len(restored) == 0


def test_preview_layer_and_id_key_are_contract_constants():
    """Names other modules hard-code against must not drift."""
    assert CandidateStore.PREVIEW_LAYER == "UVVar_Preview"
    assert CandidateStore.ID_KEY == "uvvar_store_v1"
