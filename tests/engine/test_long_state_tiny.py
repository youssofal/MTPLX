"""Long-context state on the native engine: state banks larger than the
executor's copy limit, through a replay site, the way the verify step writes
them (a few rows a round, in place).

At long context the engine left every round of a request for two reasons
that had nothing to do with the program's arithmetic (12 November 2026, the
real pack): a bank arriving in a buffer with less of the allocator's room
than the one the program was recorded in (``input_buffer_small``: 98 of 154
rounds of a cold 130K prompt, 80 of 81 of its follow-up turn), and the
first-use check needing a private copy of every bank the program writes
(``verify_memory``: every round above about 115K tokens in the founder's Pi
session). These tests reproduce both on a synthetic step with a 9 MiB bank.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.engine import runtime
from mtplx.engine.ir import ArrayIR, ProgramIR
from mtplx.engine.site import ReplaySite, compare_outputs

pytestmark = pytest.mark.skipif(
    not runtime.available(), reason=f"native engine unavailable: {runtime.unavailable()}"
)

ROWS, WIDTH, STEP = 4608, 1024, 4  # a 9 MiB float16 bank, four rows a round
BANK_BYTES = ROWS * WIDTH * 2


@pytest.fixture(autouse=True)
def _switch(monkeypatch):
    for name in (
        "MTPLX_NATIVE_ENGINE", "MTPLX_NATIVE_ENGINE_DUMP", "MTPLX_NATIVE_ENGINE_REPLAY",
        "MTPLX_NATIVE_ENGINE_PROFILE", "MTPLX_NATIVE_ENGINE_MIXED",
        "MTPLX_NATIVE_ENGINE_PROGRAMS", "MTPLX_NATIVE_ENGINE_PROGRAM_MB",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_WARM_CALLS", "1")
    yield
    mx.clear_cache()


def _step(bank, rows, at):
    """A verify step's shape: write this round's rows at the bank's offset, read the bank.

    The read of the whole bank is a sum along each row: a wrong write
    anywhere changes the output, and, like the verify step's reads, its
    working set does not grow with the bank (a sum down the columns stages a
    float32 partial of half a bank on this geometry, which would hide what
    the check itself holds)."""

    bank = mx.slice_update(bank, rows, at, axes=(0,))
    window = mx.slice(bank, at, axes=(0,), slice_size=(STEP, WIDTH))
    return [bank, (window.astype(mx.float32) * 2.0).sum(axis=0), bank.sum(axis=1)]


def _rows(n: int):
    return mx.random.normal((STEP, WIDTH), key=mx.random.key(1000 + n)).astype(mx.float16)


class _Conversation:
    """One conversation's state, held as the product holds it: by a cache
    entry, handed to the site with the call's own list, and replaced by the
    lazy outputs before MLX evaluates them (so MLX writes the bank in place)."""

    def __init__(self, site, engine, *, room: int, rows: int = ROWS):
        self.site = site
        self.engine = engine
        self.bank_bytes = rows * WIDTH * 2
        bank = mx.random.normal((rows, WIDTH), key=mx.random.key(7)).astype(mx.float16)
        mx.eval(bank)
        self.cache = {
            "bank": engine.clone_layout(bank, min_length=self.bank_bytes + room) if room else bank
        }
        self.reference = mx.array(bank)  # what MLX alone computes, for comparison
        mx.eval(self.reference)
        self.offset = 16
        self.kinds: list[str] = []
        # MLX's peak over its active memory, round by round, from just before
        # the site sees the round to its outputs evaluated: what serving the
        # round held, without the test's own reference computation.
        self.peaks: list[int] = []

    def bank_room(self) -> int:
        ref = self.engine.describe([self.cache["bank"]])[0]
        return int(ref.buffer_length) - int(ref.nbytes)

    def restore_exact(self) -> None:
        """What a restore by copy leaves: the bank in a buffer of its own size."""

        mx.clear_cache()
        exact = self.cache["bank"] + 0
        mx.eval(exact)
        self.cache["bank"] = exact
        assert self.bank_room() == 0, self.bank_room()

    def round(self, n: int, *, admit=None, windows=None, declared_offset: int = 0) -> None:
        rows = _rows(n)
        at = mx.array(self.offset, dtype=mx.int32)
        mx.eval(rows, at)
        expected = _step(self.reference, rows, at)
        mx.eval(*expected)
        self.reference = expected[0]
        inputs = [self.cache["bank"], rows, at]
        if declared_offset:
            # A window named wrong: the offset input of the declaration says
            # rows other than the ones the call writes.
            inputs.append(mx.array(self.offset + declared_offset, dtype=mx.int32))
            mx.eval(inputs[-1])
        mx.synchronize()
        active = mx.get_active_memory()
        mx.reset_peak_memory()
        try:
            ticket = self.site.begin("bank", inputs, admit=admit, extra_refs=1, windows=windows)
            del inputs
            if ticket.native:
                outputs = ticket.outputs
                self.cache["bank"] = outputs[0]
            else:
                outputs = _step(self.cache["bank"], rows, at)
                self.cache["bank"] = outputs[0]
                mx.async_eval(*outputs)
            ticket.finish(outputs)
            mx.eval(*outputs)
            mx.synchronize()
        finally:
            self.peaks.append(mx.get_peak_memory() - active)
        assert compare_outputs(list(outputs), list(expected)) is None, n
        self.kinds.append(ticket.kind)
        self.offset += STEP
        del outputs, expected


def _fallbacks(site) -> dict:
    return dict(site.counters["fallback"])


def test_a_bank_restored_into_a_buffer_of_its_own_size_stays_on_the_engine():
    engine = runtime.engine()
    site = ReplaySite("long.bank", engine=engine, log=lambda line: None)
    talk = _Conversation(site, engine, room=16384)
    assert talk.bank_room() == 16384
    for n in range(8):
        talk.round(n)
    assert site.counters["programs_live"] == 1, site.counters
    assert site.counters["inputs_relaxed"] == 1, site.counters
    native_before = site.counters["native"]
    talk.restore_exact()
    for n in range(8, 14):
        talk.round(n)
    assert site.counters["native"] == native_before + 6, _fallbacks(site)
    assert "input_buffer_small" not in _fallbacks(site)
    # In place: no copy of the bank was made to run it.
    assert int(site.counters["inputs_grown"]) == 0


def test_without_the_room_check_a_restored_bank_leaves_the_engine(monkeypatch):
    """The same conversation with the relaxation switched off: what the
    engine did before (every round after the restore on MLX's path)."""

    engine = runtime.engine()
    monkeypatch.setattr(type(engine), "relax_input", lambda self, program, index: False)
    site = ReplaySite("long.bank.strict", engine=engine, log=lambda line: None)
    talk = _Conversation(site, engine, room=16384)
    for n in range(8):
        talk.round(n)
    assert site.counters["programs_live"] == 1, site.counters
    talk.restore_exact()
    native_before = site.counters["native"]
    for n in range(8, 14):
        talk.round(n)
    assert site.counters["native"] == native_before
    assert _fallbacks(site).get("input_buffer_small") == 6, _fallbacks(site)


# The bank is written at rows [at, at + STEP) of axis 0, ``at`` being input 2.
WINDOWS = [(0, 0, 2, STEP)]


def _no_room_for_a_bank(need: int) -> bool:
    """The memory line at long context: a copy of a bank never fits."""

    return need < (1 << 20)


def _check_extra(talk) -> int:
    """What the checked round held beyond a round MLX served alone."""

    checked = talk.kinds.index("verify")
    plain = [peak for peak, kind in zip(talk.peaks, talk.kinds) if kind == "fallback"]
    assert plain, talk.kinds
    return talk.peaks[checked] - max(plain)


def _checked_conversation(rows: int, *, admit, windows):
    engine = runtime.engine()
    site = ReplaySite(f"long.check.{rows}", engine=engine, log=lambda line: None)
    talk = _Conversation(site, engine, room=0, rows=rows)
    for n in range(10):
        talk.round(n, admit=admit, windows=windows)
    return site, talk


def test_at_long_context_the_first_use_check_needs_no_copy_of_the_state():
    """The founder's session above about 115K tokens: the check's private copy
    of the banks never fit, so every new program stayed off the engine
    (verify_memory). Checked on the banks themselves, the program goes live
    and every round after gives MLX's bits; what the check holds beyond the
    round itself (digests, the chunks of the round's rows) does not grow
    with the bank: the same at 9 MiB and at 36 MiB, far under one bank."""

    extras = {}
    for rows in (ROWS, 4 * ROWS):
        site, talk = _checked_conversation(rows, admit=_no_room_for_a_bank, windows=WINDOWS)
        assert site.counters["programs_live"] == 1, site.counters
        assert site.counters["checks_in_place"] == 1, site.counters
        assert "verify_memory" not in _fallbacks(site), _fallbacks(site)
        assert site.counters["native"] >= 5, site.counters
        extras[rows] = _check_extra(talk)
        assert extras[rows] < (1 << 20), (rows, extras[rows], talk.peaks, talk.kinds)
    # Four times the bank: the check's extra memory moves by its digests
    # (8 bytes a 32 KiB chunk), not by the 27 MiB a copy would add.
    assert abs(extras[4 * ROWS] - extras[ROWS]) < (256 << 10), extras


def test_the_copying_check_holds_a_whole_bank():
    """The control for the test above: with the copy allowed (no windows, no
    memory line), the check's private copy is visible in the same
    measurement as a whole bank beyond the round."""

    site, talk = _checked_conversation(ROWS, admit=None, windows=None)
    assert site.counters["programs_live"] == 1, site.counters
    assert not site.counters.get("checks_in_place"), site.counters
    assert _check_extra(talk) >= BANK_BYTES, (talk.peaks, talk.kinds)


def test_without_windows_the_check_waits_for_memory_that_never_comes():
    """The same conversation without the declaration: what the engine did
    before (the check declined on every call)."""

    engine = runtime.engine()
    site = ReplaySite("long.check.copy", engine=engine, log=lambda line: None)
    talk = _Conversation(site, engine, room=0)
    for n in range(10):
        talk.round(n, admit=_no_room_for_a_bank)
    assert site.counters["programs_live"] == 0, site.counters
    assert site.counters["native"] == 0
    assert _fallbacks(site).get("verify_memory", 0) >= 5, _fallbacks(site)


def test_a_replay_that_writes_outside_its_named_rows_ends_the_request():
    """A check run on the live banks cannot restore what a wrong replay wrote:
    a write outside the named rows refuses the key and ends the request, as a
    replay lost on the GPU does (the session's state is dropped upstream)."""

    engine = runtime.engine()
    site = ReplaySite("long.check.outside", engine=engine, log=lambda line: None)
    talk = _Conversation(site, engine, room=0)
    wrong = [(0, 0, 3, STEP)]  # rows named by input 3, which says other rows
    lost = None
    for n in range(10):
        try:
            talk.round(n, admit=_no_room_for_a_bank, windows=wrong, declared_offset=64)
        except runtime.ReplayLost as exc:
            lost = exc
            break
    assert lost is not None and "outside" in str(lost), lost
    assert site.counters["programs_refused"].get("wrote_outside_window") == 1, site.counters
    assert site.counters["native"] == 0


def test_first_use_preserves_inputs_recycled_for_another_layers_output(monkeypatch):
    engine = runtime.engine()
    site = ReplaySite("bank.relocated", engine=engine, log=lambda line: None)
    banks = [mx.contiguous(mx.full((1, 4, 299, 256), i, dtype=mx.bfloat16)) for i in range(3)]
    inputs = [*banks, mx.array(44)]
    mx.eval(*inputs)
    arrays = [ArrayIR(i, 0, (1, 4, 299, 256), "bfloat16", (306176, 76544, 256, 1),
                      nbytes=bank.nbytes, flags=2) for i, bank in enumerate(banks)]
    # Output zero (the first bank's new state) used the second input's storage,
    # and vice versa. All three bank arrays have indistinguishable layouts.
    ir = ProgramIR("swap", [], [], inputs=[*arrays, ArrayIR(3, 0, (), "int32")],
                   outputs=[arrays[1], arrays[0], arrays[2]],
                   output_alias={0: 1, 1: 0, 2: 2})
    entry = SimpleNamespace(ir=ir, clobbered=(0, 1, 2))
    windows = [(bank, 2, 3, 4, bank) for bank in range(3)]
    plan = site._in_place_plan(engine, entry, inputs, windows)
    assert plan.relocated and set(plan.banks) == {2}
    assert plan.banks[2].output == 2
    assert site.counters["relocated_bank_checks"] == 2
    checked = site._replay_inputs(engine, entry, inputs, admit=None, keep=tuple(plan.banks))
    assert checked[0] is not banks[0] and checked[1] is not banks[1]
    assert checked[2] is banks[2]

    # At a tight memory seat the compiled round serves with every input intact.
    charges = []

    def decline(need):
        charges.append(need)
        return False

    monkeypatch.setattr("mtplx.engine.site._room_for_relocated_banks", decline)
    ticket = site._begin_check(engine, "swap", entry, inputs, None, "verify", windows=windows)
    assert ticket.reason == "verify_memory" and charges == [banks[0].nbytes + banks[1].nbytes]
