"""Construction and routing gates for exact DeepSeek-V4 M3 Q4 query projections."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase, model_forward_kind
import mtplx.deepseek_v4_attn_proj_wide_m3 as A


class _ArrayMeta:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _Q4Projection:
    def __init__(self, n: int):
        self.weight = _ArrayMeta((n, 128), mx.uint32)
        self.scales = _ArrayMeta((n, 16), mx.bfloat16)
        self.biases = _ArrayMeta((n, 16), mx.bfloat16)
        self.bits = 4
        self.group_size = 64
        self.mode = "affine"
        self.bias = None

    def __call__(self, value):
        return ("stock", value)


class _DenseProjection:
    def __init__(self, n: int, k: int = 1024):
        self.weight = _ArrayMeta((n, k), mx.bfloat16)
        self.bias = None

    def __call__(self, value):
        return ("dense", value)


def _fake_model():
    layers = []
    for layer_id in range(43):
        attn = SimpleNamespace(
            wq_b=_Q4Projection(32768),
            wq_a=object(),
            wkv=object(),
            wo_a=object(),
            wo_b=object(),
            compress_ratio=(0 if layer_id < 2 else (4 if layer_id % 2 == 0 else 128)),
        )
        if layer_id >= 2 and layer_id % 2 == 0:
            attn.indexer = SimpleNamespace(wq_b=_Q4Projection(8192))
        layers.append(SimpleNamespace(attn=attn))
    mtp_attn = SimpleNamespace(
        wq_b=_DenseProjection(32768),
        wq_a=_DenseProjection(1024, 4096),
        wkv=_DenseProjection(512, 4096),
        wo_a=_DenseProjection(8192, 4096),
        wo_b=_DenseProjection(4096, 8192),
        compress_ratio=0,
    )
    return SimpleNamespace(
        model_type="deepseek_v4",
        args=SimpleNamespace(
            hidden_size=4096,
            q_lora_rank=1024,
            num_attention_heads=64,
            head_dim=512,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
            num_hidden_layers=43,
            num_nextn_predict_layers=1,
        ),
        layers=layers,
        mtp_blocks=[SimpleNamespace(attn=mtp_attn)],
    )


def _config():
    ratios = [0, 0] + [4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 43)] + [0]
    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 4096,
        "q_lora_rank": 1024,
        "num_attention_heads": 64,
        "head_dim": 512,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "compress_ratios": ratios,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }


def test_kernel_source_pins_stock_q4_association_and_three_row_weight_reuse(monkeypatch):
    definition = {}
    launch = {}

    def fake_metal_kernel(**kwargs):
        definition.update(kwargs)

        def run(**kwargs):
            launch.update(kwargs)
            return [_ArrayMeta(kwargs["output_shapes"][0], kwargs["output_dtypes"][0])]

        return run

    monkeypatch.setattr(A.mx.fast, "metal_kernel", fake_metal_kernel)
    kernel = A._affine_q4_wide_m3_kernel.__wrapped__(32768)
    monkeypatch.setattr(A, "_affine_q4_wide_m3_kernel", lambda _n: kernel)
    stock = _Q4Projection(32768)
    projection = A._AffineQ4WideM3Projection(stock, n=32768)
    output = projection(_ArrayMeta((1, 3, 1024), mx.bfloat16))

    assert projection.weight is stock.weight
    assert projection.scales is stock.scales
    assert projection.biases is stock.biases
    source = definition["header"] + definition["source"]
    header = definition["header"]
    for declaration in (
        "constant constexpr int M = 3;",
        "constant constexpr int K = 1024;",
        "constant constexpr int VALUES_PER_THREAD = 8;",
        "constant constexpr int BYTES_PER_PACK = 4;",
        "constant constexpr int BLOCK_SIZE = 256;",
        "constant constexpr int NUM_SIMDGROUPS = 2;",
        "constant constexpr int RESULTS_PER_SIMDGROUP = 4;",
        "constant constexpr int ROWS_PER_THREADGROUP = 8;",
    ):
        assert declaration in header
    assert not any(
        line.strip().startswith("constexpr int") for line in header.splitlines()
    )
    assert "constexpr int N = 32768;" in source
    assert "uint packed_weights" in source
    assert source.index("uint packed_weights") < source.index("for (int m = 0; m < M; ++m)")
    assert "result[m][row] += qdot4_exact" in source
    assert "simd_sum(result[m][row])" in source
    assert launch["grid"] == ((32768 // 8) * 64, 1, 1)
    assert launch["threadgroup"] == (64, 1, 1)
    assert launch["output_shapes"] == [(1, 3, 32768)]
    assert launch["output_dtypes"] == [mx.bfloat16]
    assert output.shape == (1, 3, 32768)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("bits", 2, "bits=4"),
        ("group_size", 32, "group_size=64"),
        ("mode", "mxfp4", "mode=affine"),
        ("bias", _ArrayMeta((32768,), mx.bfloat16), "additive bias"),
        ("scales", _ArrayMeta((32768, 16), mx.float16), "scales"),
    ],
)
def test_projection_rejects_noncanonical_storage(field, value, error):
    stock = _Q4Projection(32768)
    setattr(stock, field, value)
    with pytest.raises(ValueError, match=error):
        A._AffineQ4WideM3Projection(stock, n=32768)


def test_installed_route_is_authoritative_target_verify_physical_m3_only():
    calls = []

    def stock(value):
        calls.append(("stock", value.shape))
        return "stock"

    def candidate(value):
        calls.append(("candidate", value.shape))
        return "candidate"

    route = A._InstalledAttnProjectionWideM3Route(stock, candidate)
    m2 = SimpleNamespace(shape=(1, 2, 1024))
    m3 = SimpleNamespace(shape=(1, 3, 1024))
    m4 = SimpleNamespace(shape=(1, 4, 1024))
    with attention_phase("decode_verify"), model_forward_kind("target_verify"):
        assert route(m2) == "stock"
        assert route(m3) == "candidate"
        assert route(m4) == "stock"
    with attention_phase("decode_verify"), model_forward_kind("repair"):
        assert route(m3) == "stock"
    with attention_phase("prefill"), model_forward_kind("other"):
        assert route(m3) == "stock"
    assert calls == [
        ("stock", (1, 2, 1024)),
        ("candidate", (1, 3, 1024)),
        ("stock", (1, 4, 1024)),
        ("stock", (1, 3, 1024)),
        ("stock", (1, 3, 1024)),
    ]


def test_preparation_censuses_exact_body_q4_shapes_and_keeps_mtp_olora_stock(monkeypatch):
    model = _fake_model()
    main_controls = tuple(layer.attn.wq_b for layer in model.layers)
    index_controls = tuple(
        model.layers[layer_id].attn.indexer.wq_b for layer_id in range(2, 43, 2)
    )
    olora_controls = tuple(
        (layer.attn.wo_a, layer.attn.wo_b) for layer in model.layers
    )
    mtp_controls = tuple(
        getattr(model.mtp_blocks[0].attn, name) for name in A._ATTN_PROJECTION_NAMES
    )
    checked = []
    monkeypatch.setattr(A, "_require_metal", lambda: None)
    monkeypatch.setattr(A, "_require_bf16_activation", lambda _model: None)
    monkeypatch.setattr(
        A,
        "_AffineQ4WideM3Projection",
        lambda stock, *, n: (lambda value: ("candidate", n, stock, value)),
    )
    monkeypatch.setattr(A, "_validate_real_weight_sentinels", lambda routes: checked.extend(routes))

    selector = A.prepare_deepseek_v4_attn_proj_wide_m3_routes(model, _config())

    assert len(checked) == 43
    assert selector.report["body_wq_b_prepared"] == 43
    assert selector.report["body_indexer_wq_b_prepared"] == 0
    assert selector.report["body_indexer_wq_b_stock"] == 21
    assert selector.report["total_q4_projections_prepared"] == 43
    assert selector.report["indexer_activation_threshold_rows"] == 512
    assert selector.report["canonical_max_compressed_rows"] == 146
    assert selector.report["mtp_attention_dense_stock"] == 1
    assert selector.report["o_lora_stock"] == 86
    assert tuple(layer.attn.wq_b for layer in model.layers) == main_controls
    assert tuple(
        model.layers[layer_id].attn.indexer.wq_b for layer_id in range(2, 43, 2)
    ) == index_controls
    assert tuple((layer.attn.wo_a, layer.attn.wo_b) for layer in model.layers) == olora_controls
    assert tuple(
        getattr(model.mtp_blocks[0].attn, name) for name in A._ATTN_PROJECTION_NAMES
    ) == mtp_controls

    selector.select_candidate()
    assert all(
        type(layer.attn.wq_b) is A._InstalledAttnProjectionWideM3Route
        for layer in model.layers
    )
    assert tuple(
        model.layers[layer_id].attn.indexer.wq_b for layer_id in range(2, 43, 2)
    ) == index_controls
    assert tuple((layer.attn.wo_a, layer.attn.wo_b) for layer in model.layers) == olora_controls
    assert tuple(
        getattr(model.mtp_blocks[0].attn, name) for name in A._ATTN_PROJECTION_NAMES
    ) == mtp_controls
    selector.select_control()
    assert tuple(layer.attn.wq_b for layer in model.layers) == main_controls
    assert tuple(
        model.layers[layer_id].attn.indexer.wq_b for layer_id in range(2, 43, 2)
    ) == index_controls


def test_selfcheck_failure_publishes_no_partial_route(monkeypatch):
    model = _fake_model()
    main_controls = tuple(layer.attn.wq_b for layer in model.layers)
    index_controls = tuple(
        model.layers[layer_id].attn.indexer.wq_b for layer_id in range(2, 43, 2)
    )
    monkeypatch.setattr(A, "_require_metal", lambda: None)
    monkeypatch.setattr(A, "_require_bf16_activation", lambda _model: None)
    monkeypatch.setattr(
        A, "_AffineQ4WideM3Projection", lambda stock, *, n: ("candidate", stock, n)
    )
    monkeypatch.setattr(
        A,
        "_validate_real_weight_sentinels",
        lambda _routes: (_ for _ in ()).throw(ValueError("layer 22 sentinel")),
    )
    with pytest.raises(ValueError, match="layer 22 sentinel"):
        A.prepare_deepseek_v4_attn_proj_wide_m3_routes(model, _config())
    assert tuple(layer.attn.wq_b for layer in model.layers) == main_controls
    assert tuple(
        model.layers[layer_id].attn.indexer.wq_b for layer_id in range(2, 43, 2)
    ) == index_controls


def test_installer_publishes_only_after_all_validation_and_selfchecks(monkeypatch):
    model = _fake_model()
    events = []

    class _Selector:
        report = {"route": "test"}

        def select_candidate(self):
            events.append("select")

    selector = _Selector()
    monkeypatch.setattr(
        A,
        "prepare_deepseek_v4_attn_proj_wide_m3_routes",
        lambda candidate_model, config: (
            events.append("prepare") or selector
        ),
    )
    report = A.install_deepseek_v4_attn_proj_wide_m3(model, _config())
    assert report == {"route": "test"}
    assert events == ["prepare", "select"]
    assert model._mtplx_dsv4_attn_proj_wide_m3_selector is selector


def test_runtime_opt_in_is_read_at_construction_only(monkeypatch):
    monkeypatch.delenv("MTPLX_DSV4_ATTN_PROJ_WIDE_M3", raising=False)
    assert A.deepseek_v4_attn_proj_wide_m3_enabled() is False
    monkeypatch.setenv("MTPLX_DSV4_ATTN_PROJ_WIDE_M3", "1")
    assert A.deepseek_v4_attn_proj_wide_m3_enabled() is True

    runtime = (Path(A.__file__).parent / "runtime.py").read_text()
    assert "deepseek_v4_attn_proj_wide_m3_enabled" in runtime
    assert "deepseek_v4_attn_proj_wide_m3_report" in runtime
    route_source = Path(A.__file__).read_text()[
        Path(A.__file__).read_text().index("class _InstalledAttnProjectionWideM3Route"):
        Path(A.__file__).read_text().index("class _PreparedProjection")
    ]
    assert "environ" not in route_source
    assert "getattr(" not in route_source
    assert "try:" not in route_source


def test_feature_off_deepseek_model_path_remains_direct():
    source = (Path(A.__file__).parent / "models" / "deepseek_v4.py").read_text()
    attention = source[
        source.index("class DeepseekV4Attention"):source.index("class DeepseekV4MLP")
    ]
    assert "q = self.wq_b(qr)" in attention
    assert "q = self.wq_b(qr).reshape" in attention
    assert "_attn_proj_wide_m3" not in attention
