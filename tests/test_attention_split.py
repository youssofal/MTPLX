from __future__ import annotations

from mtplx.attention_split import configure_split_full_attention


class DummyProjection:
    weight = None


class DummyAttention:
    q_proj = DummyProjection()
    q_norm = DummyProjection()

    def __call__(self, x, mask=None, cache=None):
        return x


class DummyLayer:
    is_linear = False

    def __init__(self):
        self.self_attn = DummyAttention()


class DummyInner:
    def __init__(self):
        self.layers = [DummyLayer()]


class DummyModel:
    def __init__(self):
        self.model = DummyInner()


def test_vllm_paged_hook_does_not_enable_split_full_attention(monkeypatch):
    monkeypatch.delenv("MTPLX_SPLIT_FULL_ATTN", raising=False)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")

    model = DummyModel()
    stats = configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    assert stats["enabled"] is True
    assert stats["split_full_attn_enabled"] is False
    assert attn._mtplx_vllm_metal_paged_enabled is True
    assert attn._mtplx_split_full_attention_explicit_enabled is False


def test_explicit_split_full_attention_chunk_one_gets_safe_default(monkeypatch):
    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN", "1")
    monkeypatch.setenv("MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE", "1")

    model = DummyModel()
    stats = configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    assert stats["split_full_attn_enabled"] is True
    assert stats["split_full_attn_chunk_size"] == 2048
    assert stats["split_full_attn_chunk_size_defaulted"] is True
    assert attn._mtplx_split_full_attention_explicit_enabled is True
    assert attn._mtplx_split_full_attention_chunk_size == 2048


def test_position_kwargs_delegate_to_original_attention():
    from mtplx.attention_split import _install_split_attention_hook

    class PositionAttention:
        def __call__(
            self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
        ):
            assert mask == "mask" and cache == "cache"
            assert position_ids == "canvas" and position_embeddings == "rotary"
            return x + 1

    attention = PositionAttention()
    assert _install_split_attention_hook(attention)
    assert (
        attention(
            2,
            mask="mask",
            cache="cache",
            position_ids="canvas",
            position_embeddings="rotary",
        )
        == 3
    )


def test_no_position_kwargs_preserve_original_attention():
    from mtplx.attention_split import _install_split_attention_hook

    class PlainAttention:
        def __call__(self, x, mask=None, cache=None):
            return x

    attention = PlainAttention()
    assert _install_split_attention_hook(attention)
    assert attention(7) == 7
