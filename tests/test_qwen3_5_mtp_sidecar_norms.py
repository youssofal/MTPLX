"""A hand-placed MTP sidecar must not bind silently mis-scaled.

Raw Qwen3.5-family HF checkpoints store MTP RMSNorm gains zero-centred (the
delta convention). mlx-lm's ``qwen3_5`` sanitize restores the +1.0 absolute
convention for TRUNK norms, but it drops every ``mtp.*`` key before that loop
runs, so a head extracted from such a checkpoint never receives the shift.

Forge shifts at build time (#301), so forge-built artifacts are already
absolute. A sidecar that reached the model another way -- copied in by hand,
extracted from a base checkpoint -- did not pass through forge, and before the
load-time shift it bound with no error at all: injection reported success,
``validate_qwen3_5_mtp_support`` passed, and decode ran at ~0% acceptance,
slower than plain AR, with nothing in the logs to say why.

No model loads here: the convention detector and the loader are exercised on
synthetic tensors written to a temporary sidecar.
"""

import mlx.core as mx
import pytest

from mtplx.compressed_tensors import mtp_sidecar_norms_are_delta, shift_delta_mtp_norms
from mtplx.qwen3_5_mtp_patch import _load_mtp_weights

# Fleet means from #301: the low set separates at 0.30-0.39 delta vs 0.87+
# absolute, q/k at 0.73-0.75 delta vs 1.73+ absolute.
DELTA_MEANS = {
    "mtp.layers.0.self_attn.q_norm.weight": 0.79,
    "mtp.layers.0.self_attn.k_norm.weight": 0.78,
    "mtp.layers.0.input_layernorm.weight": 0.04,
    "mtp.layers.0.post_attention_layernorm.weight": 0.21,
    "mtp.pre_fc_norm_hidden.weight": -0.16,
    "mtp.pre_fc_norm_embedding.weight": -0.46,
    "mtp.norm.weight": 1.25,
}


def _write_sidecar(tmp_path, means, *, shift=0.0):
    weights = {k: mx.full((16,), v + shift) for k, v in means.items()}
    weights["mtp.fc.weight"] = mx.zeros((4, 8))
    path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(path), weights)
    return path


class TestConventionDetector:
    def test_delta_gains_are_detected(self):
        weights = {k: mx.full((16,), v) for k, v in DELTA_MEANS.items()}
        assert mtp_sidecar_norms_are_delta(weights) is True

    def test_absolute_gains_are_not_detected(self):
        weights = {k: mx.full((16,), v + 1.0) for k, v in DELTA_MEANS.items()}
        assert mtp_sidecar_norms_are_delta(weights) is False

    def test_shift_is_idempotent(self):
        weights = {k: mx.full((16,), v) for k, v in DELTA_MEANS.items()}
        once = shift_delta_mtp_norms(weights)
        twice = shift_delta_mtp_norms(once)
        for key in DELTA_MEANS:
            assert float(mx.mean(once[key])) == pytest.approx(
                float(mx.mean(twice[key])), abs=1e-6
            ), key


class TestLoaderRestoresTheConvention:
    def test_delta_sidecar_is_shifted_on_load(self, tmp_path):
        path = _write_sidecar(tmp_path, DELTA_MEANS)
        loaded = _load_mtp_weights([path])
        for key, mean in DELTA_MEANS.items():
            got = float(mx.mean(loaded[key[len("mtp.") :]]))
            assert got == pytest.approx(mean + 1.0, abs=1e-3), key

    def test_absolute_sidecar_passes_through_unchanged(self, tmp_path):
        """Forge-built sidecars are already absolute; they must not double-shift."""
        path = _write_sidecar(tmp_path, DELTA_MEANS, shift=1.0)
        loaded = _load_mtp_weights([path])
        for key, mean in DELTA_MEANS.items():
            got = float(mx.mean(loaded[key[len("mtp.") :]]))
            assert got == pytest.approx(mean + 1.0, abs=1e-3), key

    def test_non_norm_tensors_are_untouched(self, tmp_path):
        path = _write_sidecar(tmp_path, DELTA_MEANS)
        loaded = _load_mtp_weights([path])
        assert float(mx.max(mx.abs(loaded["fc.weight"]))) == 0.0

    def test_mtp_prefix_is_stripped(self, tmp_path):
        path = _write_sidecar(tmp_path, DELTA_MEANS)
        loaded = _load_mtp_weights([path])
        assert all(not k.startswith("mtp.") for k in loaded)
        assert "norm.weight" in loaded
