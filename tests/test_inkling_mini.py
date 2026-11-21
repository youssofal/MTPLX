from mtplx.commands.forge_inkling_quant import matrix_layout
from mtplx.models.inkling.layout import draft_keys
from mtplx.models.inkling.protocol import Channels, prepare_message


class Tokenizer:
    def __init__(self):
        self.ids = {}

    def convert_tokens_to_ids(self, text):
        return self.ids.setdefault(text, len(self.ids) + 300)


def test_native_records_stream_across_chunks_without_leaking_controls():
    tok = Tokenizer()
    channels = Channels(tok)

    def control(name):
        return tok.convert_tokens_to_ids("<|" + name + "|>")

    raw = [
        control("message_model"),
        control("content_thinking"),
        *b"think",
        control("end_message"),
        control("message_model"),
        *b"find",
        control("content_invoke_tool_json"),
        *b'{"name":"find","args":{"q":"x"}}',
        control("end_message"),
        control("message_model"),
        control("content_text"),
        *b"OK",
        control("end_message"),
    ]
    result = "".join(
        channels.feed(raw[i : i + 3], lambda ids: bytes(ids).decode(), lambda: "")
        for i in range(0, len(raw), 3)
    )
    assert (
        result + channels.finish(lambda: "")
        == '<think>think</think><tool_call>{"name":"find","arguments":{"q":"x"}}</tool_call>OK'
    )


def test_native_image_parts_preserve_order_and_escape_user_placeholders():
    images = []
    msg = prepare_message(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|unused_200054|>"},
                {"type": "image_url", "image_url": {"url": "sample"}},
                {"type": "text", "text": "after"},
            ],
        },
        images,
        lambda value: value.encode(),
    )
    assert [item["type"] for item in msg["content"]] == ["text", "image", "text"]
    assert "<|unused_200054|>" not in msg["content"][0]["text"]
    assert images == [b"sample"]


def test_pack_precision_preserves_published_experts_and_loop_weights():
    head = "model.llm.unembed.weight"
    assert matrix_layout(head, "quality") is None
    assert matrix_layout(head, "native-quality") == (8, 32)
    assert matrix_layout(head, "speed") == (4, 64)
    assert matrix_layout("model.llm.layers.7.attn.wq_du.weight", "speed") == (8, 64)
    assert matrix_layout("model.llm.layers.22.mlp.experts.w2_weight", "speed-affine") == (4, 64)
    for recipe in ("speed", "speed-nvfp4w"):
        assert matrix_layout(head, recipe) == (4, 64)
        assert matrix_layout("model.llm.layers.7.attn.wq_du.weight", recipe) == (8, 64)
        assert matrix_layout("model.llm.layers.22.mlp.experts.w2_weight", recipe) == "nvfp4"
    for recipe in ("quality", "native-quality"):
        for layer in range(24):
            for suffix in (
                "mlp.experts.w13_weight",
                "mlp.experts.w2_weight",
                "attn.wq_du.weight",
                "attn.wo_ud.weight",
                "controllers.k.down_weight",
            ):
                assert (
                    matrix_layout(f"model.llm.layers.{layer}.{suffix}", recipe) is None
                )
        assert matrix_layout("model.visual.proj_in.weight", recipe) is None
    assert len(draft_keys(3)) == 54


def test_cache_geometry_charges_every_loop_visit_and_its_convolution_tail():
    from mtplx.models.inkling.layout import cache_footprint
    from mtplx.memory_plan import dense_kv_bytes_per_token_from_config

    text = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2304,
        "sliding_window_size": 1024,
        "sconv_kernel_size": 4,
        "local_layer_ids": [n for n in range(24) if n % 6 != 5],
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
    }
    footprint = cache_footprint(text)
    assert (
        dense_kv_bytes_per_token_from_config(
            {"model_type": "inkling2_loop", "text_config": text}
        )
        == 32768
    )
    assert footprint.bounded_bytes(65536) == 40 * 1024 * 4096 + 48 * 3 * 4 * (
        2048 + 4608
    )
    assert (
        footprint.bounded_bytes(4096, 512) - footprint.bounded_bytes(4096)
        == 40 * 511 * 4096
    )
    draft = cache_footprint(
        {**text, "num_hidden_layers": 3, "local_layer_ids": [0, 2]}, draft=True
    )
    assert draft.full_bytes_per_token == 4096
    assert draft.bounded_bytes(65536) == 2 * 1024 * 4096 + 2304 * 2


def test_convolution_cache_keeps_exact_last_rows_across_chunks():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache
    from mtplx.models.inkling.model import Convolution

    weights = {"conv.weight": mx.ones((8, 1, 4))}
    conv = Convolution(weights, "conv")
    cache = ArraysCache(1)
    x = mx.arange(128).reshape(1, 16, 8).astype(mx.bfloat16)
    conv(x, cache, 0)
    assert mx.array_equal(cache[0], x[:, -3:].astype(mx.float32)).item()
    conv(x[:, :1], cache, 0)
    expected = mx.concatenate([x[:, -2:], x[:, :1]], axis=1).astype(mx.float32)
    assert mx.array_equal(cache[0], expected).item()
