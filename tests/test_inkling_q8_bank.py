"""8-bit global KV for Inkling (``--kv-quant q8``): packed rows, the bank cache, and the switch."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache  # noqa: E402

from mtplx.models.inkling import cache as C  # noqa: E402


def _rows(n, seed=0, heads=8, dim=128):
    mx.random.seed(seed)
    return (mx.random.normal((1, heads, n, dim)) * 3).astype(mx.bfloat16)


def _reference(x):
    return mx.dequantize(*mx.quantize(x, group_size=64, bits=8), group_size=64, bits=8)


def test_packed_rows_are_mx_quantize_bit_for_bit():
    x = _rows(37)
    p = C.pack_rows(x)
    assert p.dtype == mx.uint8 and p.shape == (1, 8, 37, 136)
    assert C.packed_width(128) == 136 and p.nbytes / x.nbytes == 136 / 256
    assert bool(mx.array_equal(C.unpack_rows(p), _reference(x)))


def test_unpack_reads_the_byte_order_it_claims():
    # Negative control: the same bytes in the wrong order within each 4-byte word must not
    # give the same rows, or the test above could not catch a layout mistake.
    x = _rows(9, seed=1)
    p = C.pack_rows(x)
    vals = p[..., :128].reshape(1, 8, 9, 32, 4)[..., ::-1].reshape(1, 8, 9, 128)
    swapped = mx.concatenate([vals, p[..., 128:]], axis=-1)
    assert not bool(mx.array_equal(C.unpack_rows(swapped), _reference(x)))


def test_pack_refuses_other_dtypes():
    with pytest.raises(ValueError):
        C.pack_rows(_rows(2).astype(mx.float16))


def test_bank_cache_appends_like_kvcache_and_returns_unpacked_rows():
    bank, stock = C.Q8BankCache(), KVCache()
    pieces = [_rows(n, seed=n) for n in (700, 1, 1, 300)]
    for x in pieces:
        k, v = bank.update_and_fetch(x, -x)
        sk, _ = stock.update_and_fetch(x, -x)
    assert bank.offset == stock.offset == 1002
    assert bank.keys.dtype == mx.uint8 and bank.keys.shape[2] == stock.keys.shape[2]
    want = mx.concatenate([_reference(x) for x in pieces], axis=2)
    assert k.dtype == mx.bfloat16 and bool(mx.array_equal(k, want))
    assert bool(mx.array_equal(v, mx.concatenate([_reference(-x) for x in pieces], axis=2)))
    assert bank.trim(2) == 2 and bank.offset == 1000
    keys, values = bank.state
    assert keys.shape == (1, 8, 1000, 136)


def test_size_global_banks_grows_packed_banks_in_their_own_layout():
    from mtplx.models.inkling.fast.state import BANK_STEP

    bank = C.Q8BankCache()
    bank.update_and_fetch(_rows(10), _rows(10, seed=2))
    assert C.size_global_banks([bank], 5000) == 1
    assert bank.keys.dtype == mx.uint8 and bank.keys.shape == (1, 8, 2 * BANK_STEP, 136)
    assert bool(mx.array_equal(C.unpack_rows(bank.keys[:, :, :10]), _reference(_rows(10))))
    C.settle_bank_steps([bank])
    assert bank.step == BANK_STEP


def test_make_cache_switch(monkeypatch):
    from mtplx.models.inkling.model import Model as InklingModel

    fake = SimpleNamespace(
        visit_order=[0, 1, 1],
        layers=[SimpleNamespace(sliding=True), SimpleNamespace(sliding=False)],
        config={"sliding_window_size": 1024},
    )
    for raw, want in (("off", KVCache), ("q8", C.Q8BankCache)):
        monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", raw)
        monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
        cache = InklingModel.make_cache(fake)
        assert [type(c).__name__ for c in cache[0::2]] == ["WindowCache", want.__name__, want.__name__]


def test_footprint_counts_packed_rows_for_global_visits_only(monkeypatch):
    from mtplx.models.inkling.layout import cache_footprint

    import json
    from pathlib import Path

    path = Path.home() / ".mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Bare-Speed/config.json"
    if not path.exists():
        pytest.skip("needs the Inkling 2 Mini config")
    cfg = json.loads(path.read_text())
    cfg = cfg.get("text_config", cfg)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    off = cache_footprint(cfg)
    assert off.full_bytes_per_token == 32768  # 8 global visits x 2 x 8 heads x 128 x 2 bytes
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    q8, q8_draft = cache_footprint(cfg), cache_footprint(cfg, draft=True)
    assert q8.full_bytes_per_token * 256 == off.full_bytes_per_token * 136
    assert q8.sliding_bytes_per_token == off.sliding_bytes_per_token
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    assert q8_draft == cache_footprint(cfg, draft=True)


def test_policy_lets_q8_through_the_server_gate():
    from mtplx.backends.descriptors import INKLING_DESCRIPTOR, kv_quant_policy_for_model

    policy = kv_quant_policy_for_model(model_ref="Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed",
                                       descriptor=INKLING_DESCRIPTOR)
    assert policy.supported and "q8" in policy.modes and "q4" not in policy.modes


def _panel(history, rows, seed):
    mx.random.seed(seed)
    cap = -(-(history + rows) // 4096) * 4096
    k = (mx.random.normal((1, 8, cap, 128)) * 0.7).astype(mx.bfloat16)
    v = mx.random.normal((1, 8, cap, 128)).astype(mx.bfloat16)
    q = mx.random.normal((1, rows, 24, 128)).astype(mx.bfloat16)
    rel = mx.random.normal((1, rows, 24, 16)).astype(mx.bfloat16)
    sink = mx.random.normal((1, rows, 24)).astype(mx.bfloat16)
    gate = mx.random.normal((1, rows, 3072)).astype(mx.bfloat16)
    kr = mx.random.normal((1, 8, rows, 128)).astype(mx.bfloat16)
    vr = mx.random.normal((1, 8, rows, 128)).astype(mx.bfloat16)
    basis = (mx.random.normal((16, 1024)) * 0.3).astype(mx.bfloat16)
    return k, v, (q, rel, sink, gate, kr, vr), basis


def _bits(x):
    return x.view(mx.uint16)


@pytest.mark.parametrize("history,rows", [(300, 1), (1500, 3), (4142, 8), (16384, 4), (2100, 32)])
def test_attend_q8_is_class_c_on_the_unpacked_bank_bit_for_bit(history, rows):
    """So attend_q8 inherits class C's bound against RC6, row stability and capacity independence
    (``test_inkling_fast_attend``), and the only change is the stored values."""
    from mtplx.models.inkling.fast.attend import attend, attend_c, attend_q8

    k, v, panel, basis = _panel(history, rows, seed=history + rows)
    pk, pv = C.pack_rows(k), C.pack_rows(v)
    args = (*panel,)
    got = attend_q8(*args, pk, pv, history, history, history, basis)
    want = attend_c(*args, C.unpack_rows(pk), C.unpack_rows(pv), history, history, history, basis, window=False)
    assert bool(mx.array_equal(_bits(got), _bits(want)))
    routed = attend(*args, pk, pv, history, history, history, basis, window=False)
    assert bool(mx.array_equal(_bits(routed), _bits(got))), "attend() must send packed banks to attend_q8"
    # Negative control: one committed row's scale changed must change the result.
    bad = mx.array(pk)
    bad[0, 0, history // 2, 128:130] = mx.array([0x80, 0x3F], mx.uint8)
    assert not bool(mx.array_equal(_bits(attend_q8(*args, bad, pv, history, history, history, basis)), _bits(got)))


def _fake_model(width=16):
    conv = SimpleNamespace(weight=mx.zeros((4, width)))
    layers = [SimpleNamespace(sliding=True, convs=[conv] * 4), SimpleNamespace(sliding=False, convs=[conv] * 4)]
    return SimpleNamespace(
        visit_order=[0, 1, 1], layers=layers,
        config={"sliding_window_size": 1024, "sconv_kernel_size": 4, "num_key_value_heads": 8, "head_dim": 128},
        embed_tokens=SimpleNamespace(weight=mx.zeros((1, 1), mx.bfloat16)),
    )


def test_fast_path_engages_on_packed_banks(monkeypatch):
    """The served decode path with --kv-quant q8: the fast decoder accepts the packed banks
    (no "layout" fallback to the reference), the decode state keeps them packed through
    prepare and commit, and the global view it hands to attention is the packed bank, which
    attend() serves with attend_q8."""
    from mlx_lm.models.cache import RotatingKVCache

    from mtplx.models.inkling.fast import decoder as D
    from mtplx.models.inkling.fast.attend import attend, attend_q8
    from mtplx.models.inkling.fast.state import BANK_STEP, DecodeState, VisitDelta
    from mtplx.models.inkling.model import Model as InklingModel

    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_INKLING_FAST", raising=False)
    model = _fake_model()
    cache = InklingModel.make_cache(model)
    history = 600
    for visit in range(3):
        rows = _rows(history, seed=visit)
        cache[2 * visit].update_and_fetch(rows, -rows)   # the prompt read, as the reference path writes it
    fast = D.FastDecoder(SimpleNamespace(model=model, trunk_entries=6))
    assert fast.decline(1, cache) is None, "the fast path must not decline packed banks"
    assert fast.decline(4, cache) is None
    # Control: a global visit holding something other than a global bank still declines.
    wrong = list(cache)
    wrong[2] = RotatingKVCache(max_size=1024)
    wrong[2].update_and_fetch(_rows(4), _rows(4))
    assert fast.decline(1, wrong) == "layout"

    state = DecodeState(model, cache)
    state.prepare()
    bank = cache[2]
    assert bank.keys.dtype == mx.uint8 and bank.keys.shape[3] == 136 and bank.keys.shape[2] % BANK_STEP == 0
    kept = 3
    deltas = [VisitDelta(_rows(4, seed=10 + v), _rows(4, seed=20 + v), tuple(mx.zeros((1, 4, 16)) for _ in range(4)))
              for v in range(3)]
    state.commit(deltas, kept)
    assert state.length == history + kept
    assert bank.keys.dtype == mx.uint8, "a commit must keep the bank packed"
    assert bool(mx.array_equal(bank.keys[:, :, history:history + kept], C.pack_rows(deltas[1].keys[:, :, :kept])))
    assert bool(mx.array_equal(C.unpack_rows(bank.keys[:, :, :history]), _reference(_rows(history, seed=1))))

    keys, values, count = state.global_view(1)
    assert keys.dtype == mx.uint8
    _, _, panel, basis = _panel(state.length, 1, seed=3)
    got = attend(*panel, keys, values, count, count, count, basis, window=False)
    want = attend_q8(*panel, keys, values, count, count, count, basis)
    assert bool(mx.array_equal(_bits(got), _bits(want)))


def test_committing_round_stays_off_under_q8(monkeypatch):
    """The one-program committing round writes bf16 rows into the banks; under q8 the program
    only decides and the host commit packs the kept rows."""
    from mtplx.models.inkling.fast.decoder import round_commit_enabled

    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_INKLING_ROUND_COMMIT", "1")
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    assert round_commit_enabled()
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    assert not round_commit_enabled()


def test_fp16_pack_turns_q8_off(monkeypatch, capsys):
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    assert C.settle_kv_quant_for_pack(mx.bfloat16) == "q8" and C.q8_enabled()
    assert C.settle_kv_quant_for_pack(mx.float16) == "off"
    assert not C.q8_enabled(), "the env must read off afterwards, for make_cache, the footprint and the identity"
    assert "needs a bf16 pack" in capsys.readouterr().out
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    assert C.settle_kv_quant_for_pack(mx.float16) == "off"


def test_session_identity_is_salted_only_for_inkling_q8(monkeypatch):
    """Snapshots restore only into a cache of their own layout: q8 and off hash apart for Inkling,
    off hashes exactly as before (existing snapshots still match), other models never change."""
    openai = pytest.importorskip("mtplx.server.openai")
    inkling, qwen = SimpleNamespace(backend_id="inkling2_loop"), SimpleNamespace(backend_id="qwen3_next_mtp")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "off")
    plain = "abc123"
    assert openai._session_identity_hash(inkling, plain) == plain
    assert openai._session_identity_hash(qwen, plain) == plain
    assert openai._session_identity_hash(inkling, None) is None
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")
    salted = openai._session_identity_hash(inkling, plain)
    assert salted != plain and salted == openai._session_identity_hash(inkling, plain)
    assert openai._session_identity_hash(qwen, plain) == plain
