"""MTP (speculative decoding) on Edge0-120B packs (mtplx/edge0/mtp.py and its wiring).

A tiny synthetic Edge0 folder is forged twice: once without a draft head and
once with a synthetic Flash Next head (tests/edge0_synth.write_mtp_head, the
base head's layout at the synthetic geometry) copied in by the forge. The
checks: the runtime attaches and validates the head, draft logits are finite,
greedy speculative decode equals autoregressive decode, --no-mtp loads no
head, the compatibility verdict with and without the head, the backend
descriptor's sampler/depth/reasoning policy, and that the CLI and the server
resolve Edge0 to the 'batched' verify with the recurrent-state snapshot kept
and none of the base family's (qwen4_exp) speed lanes. The real-pack check is
header-only and runs with MTPLX_EDGE0_PACK set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.edge0 import forge  # noqa: E402
from mtplx.edge0 import mtp as edge0_mtp  # noqa: E402
from mtplx.models import edge0_latent  # noqa: E402
from tests import edge0_synth as syn  # noqa: E402

REAL_PACK = os.environ.get("MTPLX_EDGE0_PACK")
PROMPT = [3, 17, 42, 8, 61, 5, 29, 77, 11, 50]
# Speculative lanes Edge0 shares with the base family: stamped in MTP mode only.
EDGE0_MTP_LANES = {"MTPLX_FAMILY_CAPTURE_COMMIT", "MTPLX_QSA_GATHER"}
EDGE0_FRSPEC = {"MTPLX_FRSPEC_DRAFT", "MTPLX_FRSPEC_VOCAB"}
# Keys the base family (qwen4_exp) stamps that must never reach Edge0.
QWEN4_ONLY = {
    "MTPLX_FUSED_HC_V3",
    "MTPLX_FUSED_GDN_INPROJ",
    "MTPLX_FUSED_GATE_UP",
    "MTPLX_FUSED_CONVNORM_VERIFY",
    "MTPLX_QSA_GATHER_MAX_ROWS",
    "MTPLX_QSA_M4_FUSED_KV_GATHER",
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_BATCH_TARGET_ARRAYS",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
    "MTPLX_NGRAM_PREWARM",
    "MTPLX_SESSION_BANK_SHED_BOUNDARIES",
    "MTPLX_SESSION_BANK_PROTECTED_TERMINAL",
}


def _qwen4_only(keys) -> set[str]:
    return {k for k in keys if k in QWEN4_ONLY or k.startswith("MTPLX_QWEN4_")}


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    """(root, plain pack without a head, pack with the synthetic head)."""
    root = tmp_path_factory.mktemp("edge0-mtp")
    syn.make_folder(root / "src", root / "base")
    forge.build_pack(root / "src", root / "plain", tokenizer_from=root / "base")
    cfg = json.loads((root / "plain" / "config.json").read_text())
    syn.write_mtp_head(root / "base", cfg)
    # auto (the default) finds the base folder's mtp.safetensors next to the tokenizer
    forge.build_pack(root / "src", root / "out", tokenizer_from=root / "base")
    return root, root / "plain", root / "out"


@pytest.fixture
def snapshot_kept(monkeypatch):
    monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "0")


@pytest.fixture(scope="module")
def rt(packs):
    from mtplx import runtime

    _root, _plain, out = packs
    return runtime.load(out, mtp=True)


def _inspection(path) -> dict:
    from mtplx.artifacts import inspect_model

    return inspect_model(str(path)).to_dict()


# --------------------------------------------------------------------------- runtime


def test_config_predicate_is_its_own(packs):
    from mtplx.models.qwen4_exp import is_qwen4_exp_mtp_config

    _root, _plain, out = packs
    cfg = json.loads((out / "config.json").read_text())
    assert edge0_mtp.is_edge0_latent_mtp_config(cfg) and edge0_mtp.declares_head(cfg)
    assert not is_qwen4_exp_mtp_config(cfg)  # every qwen4_exp lane stays off Edge0
    assert not edge0_mtp.is_edge0_latent_mtp_config({"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}})


def test_runtime_load_attaches_and_validates_the_head(rt):
    from mtplx.models.qwen4_exp import Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    assert isinstance(rt.model, edge0_latent.Model)
    assert rt.mtp_enabled and validate_mtp_support(rt.model)
    assert isinstance(rt.model.language_model.mtp, Qwen4ExpMTP)
    # the head's own MoE is the base geometry (switch_mlp), not a latent block
    assert "switch_mlp" in rt.model.language_model.mtp.layers[0].mlp


def test_draft_logits_are_finite(rt):
    model = rt.model
    cache = model.make_cache()
    logits, hidden = model(mx.array([PROMPT]), cache=cache, return_hidden=True)
    nxt = int(mx.argmax(logits[0, -1]).item())
    draft = model.mtp_forward(hidden[:, -1:], mx.array([[nxt]]), mtp_cache=model.make_mtp_cache())
    mx.eval(draft)
    assert draft.shape == (1, 1, logits.shape[-1])
    assert bool(mx.isfinite(draft).all())


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_greedy_speculative_equals_autoregressive(rt, snapshot_kept, depth):
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.sampling import SamplerConfig

    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    ar = generate_ar(rt, PROMPT, max_tokens=40, sampler=greedy, stop_token_ids=set())
    sp = generate_mtpk(rt, PROMPT, max_tokens=40, sampler=greedy, stop_token_ids=set(),
                       speculative_depth=depth, verify_strategy="batched")
    assert len(ar.tokens) == 40
    assert sp.tokens == ar.tokens
    assert sp.stats.drafted_tokens > 0  # the head really drafted


def test_serve_draft_lm_head_and_sampled_decode(packs, snapshot_kept):
    """The serve path installs a 4-bit draft copy of lm_head for native-MTP descriptors (Edge0's included)."""
    from mtplx import runtime
    from mtplx.draft_lm_head import _install_draft_lm_head
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.sampling import SamplerConfig

    r = runtime.load(packs[2], mtp=True)
    report = _install_draft_lm_head(r, bits=4, group_size=64, mode="affine")
    assert report["draft_only"]["bits"] == 4
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    ar = generate_ar(r, PROMPT, max_tokens=30, sampler=greedy, stop_token_ids=set())
    sp = generate_mtpk(r, PROMPT, max_tokens=30, sampler=greedy, stop_token_ids=set(), speculative_depth=3,
                       verify_strategy="batched")
    assert sp.tokens == ar.tokens
    sampled = generate_mtpk(r, PROMPT, max_tokens=30, sampler=SamplerConfig(temperature=1.0, top_p=0.95, top_k=20),
                            stop_token_ids=set(), speculative_depth=3, verify_strategy="batched", seed=3)
    assert len(sampled.tokens) == 30 and sampled.stats.drafted_tokens > 0


def test_bfloat16_policy_with_the_head(packs, snapshot_kept, monkeypatch):
    from mtplx import runtime
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    monkeypatch.setenv(edge0_latent.DTYPE_ENV, "bfloat16")
    r = runtime.load(packs[2], mtp=True)
    assert r.mtp_enabled and r.model.activation_dtype == mx.bfloat16
    out = generate_mtpk(r, PROMPT, max_tokens=12, sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
                        stop_token_ids=set(), speculative_depth=3, verify_strategy="batched")
    assert len(out.tokens) == 12


def test_no_mtp_loads_no_head(packs):
    from mtplx import runtime

    r = runtime.load(packs[2], mtp=False)
    assert not r.mtp_enabled and getattr(r.model.language_model, "mtp", None) is None


def test_pack_without_a_head_degrades_to_autoregressive(packs):
    from mtplx import runtime

    r = runtime.load(packs[1], mtp=True)
    assert not r.mtp_enabled and getattr(r.model.language_model, "mtp", None) is None


def test_a_head_that_does_not_fit_fails_the_load_loudly(packs, tmp_path):
    from mtplx import runtime

    _root, _plain, out = packs
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    cfg = json.loads((bad / "config.json").read_text())
    other = json.loads(json.dumps(cfg))
    other["text_config"]["moe_intermediate_size"] = int(other["text_config"]["moe_intermediate_size"]) * 2
    syn.write_mtp_head(bad / "mtp.safetensors", other)
    with pytest.raises(RuntimeError, match="mtp.safetensors does not fit"):
        runtime.load(bad, mtp=True)


# --------------------------------------------------------------------------- registry and descriptor


def test_verdict_with_the_head(packs):
    comp = _inspection(packs[2])["compatibility"]
    assert comp["arch_id"] == "edge0-latent" and comp["can_run"]
    assert comp["mtp_supported"] == "yes" and comp["recommended_backend"] == "edge0_latent"
    assert comp["runtime_compatibility"] == "native-family-gated"
    assert comp["tier"] == "family-compatible-unverified"


def test_verdict_without_the_head_says_why(packs):
    comp = _inspection(packs[1])["compatibility"]
    assert comp["arch_id"] == "edge0-latent" and comp["can_run"] and comp["tier"] == "AR-only"
    assert comp["mtp_supported"] == "no" and comp["runtime_compatibility"] == "native-ar-only-missing-mtp"
    assert "no MTP draft head" in comp["message"] and "--mtp-head base" in comp["message"]


def test_verdict_declared_but_missing_and_not_fitting(packs, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(packs[2], out)
    (out / "mtp.safetensors").unlink()
    comp = _inspection(out)["compatibility"]
    assert comp["runtime_compatibility"] == "native-ar-only-missing-mtp" and comp["mtp_supported"] == "no"
    assert "although config.json declares one" in comp["message"]
    shutil.copyfile(out / "model-latent-000.safetensors", out / "mtp.safetensors")  # not a head
    comp = _inspection(out)["compatibility"]
    assert comp["runtime_compatibility"] == "native-ar-only-mtp-unsupported" and comp["mtp_supported"] == "no"
    assert comp["can_run"] and "not a Flash Next draft head" in comp["message"]


def test_descriptor_policy(packs):
    from mtplx.backends import descriptors as d
    from mtplx.backends.family_settings import family_settings
    from mtplx.backends.registry import SUPPORTED_ARCH_IDS

    assert "edge0-latent" in SUPPORTED_ARCH_IDS
    insp = _inspection(packs[2])
    desc = d.descriptor_from_inspection(insp)
    assert desc.backend_id == "edge0_latent" and desc.uses_draft_lm_head
    assert d.model_family_from_inspection(insp, model_ref=str(packs[2])) == "edge0_latent"
    sem = d.draft_semantics_for_model(str(packs[2]), insp, desc)
    assert (sem.default, sem.maximum) == (3, 3)
    assert d.sampler_defaults_for_model(str(packs[2]), insp, desc).to_dict() == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    codec = d.reasoning_policy_for_model(str(packs[2]), insp, desc)
    assert codec.parser == "qwen3" and codec.effort_levels == ("xhigh", "medium", "low")
    assert family_settings("edge0_latent") is None  # no base-family settings block
    # the daemon re-resolves from --backend-id and the model ref, without an inspection
    runtime_desc = d.descriptor_from_runtime(SimpleNamespace(backend_id=None, model_path=str(packs[2])),
                                             SimpleNamespace(backend_id="edge0_latent", model=str(packs[2])))
    assert runtime_desc.backend_id == "edge0_latent" and runtime_desc.draft_semantics.maximum == 3
    assert runtime_desc.sampler_defaults.temperature == 1.0 and runtime_desc.reasoning_codec.parser == "qwen3"


# --------------------------------------------------------------------------- CLI and server gating


def test_cli_resolves_batched_and_keeps_the_head(packs):
    from mtplx.commands import public

    insp = _inspection(packs[2])
    args = SimpleNamespace(_cli_flags=set(), verify_strategy="capture_commit", generation_mode="mtp", depth=3,
                           load_mtp=True)
    assert public._apply_runtime_compatibility_mode(args, insp, printer=lambda *_: None) is None
    assert args.load_mtp is True
    public._apply_backend_serve_defaults(args, insp)
    assert args.verify_strategy == "batched"
    assert (args.temperature, args.top_p, args.top_k) == (1.0, 0.95, 20)
    # an explicit operator choice still wins
    args = SimpleNamespace(_cli_flags={"verify-strategy"}, verify_strategy="sequential")
    public._apply_backend_serve_defaults(args, insp)
    assert args.verify_strategy == "sequential"


def test_cli_without_the_head_serves_autoregressive(packs):
    from mtplx.commands import public

    insp = _inspection(packs[1])
    args = SimpleNamespace(_cli_flags=set(), generation_mode="mtp", depth=3, load_mtp=True)
    lines = []
    public._apply_runtime_compatibility_mode(args, insp, printer=lines.append)
    assert args.load_mtp is False and args.depth == 0 and lines


def test_server_and_run_keep_the_snapshot_and_no_base_lanes(packs, monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai as srv

    for key in ("MTPLX_NAX_VERIFY", "MTPLX_SKIP_VERIFY_SNAPSHOT", *QWEN4_ONLY, *EDGE0_MTP_LANES, *EDGE0_FRSPEC):
        monkeypatch.delenv(key, raising=False)
    out = str(packs[2])
    args = argparse.Namespace(model=out, generation_mode="mtp", verify_strategy="batched")
    got = srv._server_runtime_env_overrides(args, {})
    assert got["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0" and got["MTPLX_NAX_VERIFY"] == "0"
    assert not _qwen4_only(got), _qwen4_only(got)
    # the speculative lanes Edge0 shares with the base; no draft vocabulary in this pack, so no pruned head
    assert all(got[k] == "1" for k in EDGE0_MTP_LANES) and not EDGE0_FRSPEC & set(got)
    # an operator's export of a lane wins, and a pack loaded without its head gets none
    monkeypatch.setenv("MTPLX_FAMILY_CAPTURE_COMMIT", "0")
    assert "MTPLX_FAMILY_CAPTURE_COMMIT" not in srv._server_runtime_env_overrides(args, {})
    monkeypatch.delenv("MTPLX_FAMILY_CAPTURE_COMMIT")
    no_head = argparse.Namespace(model=out, generation_mode="mtp", verify_strategy="batched", load_mtp=False)
    assert not (EDGE0_MTP_LANES | EDGE0_FRSPEC) & set(srv._server_runtime_env_overrides(no_head, {}))
    # turbo-style export of the skip does not reach the speculative lane
    monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "1")
    assert srv._server_runtime_env_overrides(args, {})["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0"
    # an operator's NAX export wins
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    assert "MTPLX_NAX_VERIFY" not in srv._server_runtime_env_overrides(args, {})
    # autoregressive: neither key is stamped
    ar = srv._server_runtime_env_overrides(argparse.Namespace(model=out, generation_mode="ar", verify_strategy=None), {})
    assert "MTPLX_SKIP_VERIFY_SNAPSHOT" not in ar and "MTPLX_NAX_VERIFY" not in ar and not _qwen4_only(ar)
    assert not (EDGE0_MTP_LANES | EDGE0_FRSPEC) & set(ar)
    # `mtplx run` resolves the same contract in-process
    run = public._in_process_runtime_env_overrides(SimpleNamespace(verify_strategy="batched"), out, generation_mode="mtp")
    assert run["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0" and not _qwen4_only(run)
    # bench/tune runs: the snapshot key only
    from mtplx.profiles import get_profile

    bench = public._runtime_env_with_model_contract_overrides({}, _inspection(out), get_profile("turbo"), model=out)
    assert bench == {"MTPLX_SKIP_VERIFY_SNAPSHOT": "0"}


def test_draft_vocabulary_is_written_and_stamped(packs, tmp_path, monkeypatch):
    """The release's draft vocabulary rides the head: sorted ``draft_vocab.npy``, and the server prunes drafts to it."""
    from mtplx.server import openai as srv

    root = packs[0]
    src = tmp_path / "src"
    shutil.copytree(root / "src", src, symlinks=True)
    vocab = json.loads((packs[2] / "config.json").read_text())["text_config"]["vocab_size"]
    ids = np.array([vocab - 1, 0, 7, 3], dtype="<i4")
    (src / "mtp").mkdir(exist_ok=True)
    (src / "mtp" / "draft_vocab.bin").write_bytes(ids.tobytes())
    out = tmp_path / "out"
    state = forge.build_pack(src, out, tokenizer_from=root / "base")
    assert state["draft_vocab"]["ids"] == 4
    assert np.load(out / "draft_vocab.npy").tolist() == sorted(ids.tolist())
    for key in (*EDGE0_MTP_LANES, *EDGE0_FRSPEC):
        monkeypatch.delenv(key, raising=False)
    args = argparse.Namespace(model=str(out), generation_mode="mtp", verify_strategy="batched")
    got = srv._server_runtime_env_overrides(args, {})
    assert got["MTPLX_FRSPEC_DRAFT"] == "1" and got["MTPLX_FRSPEC_VOCAB"] == str(out / "draft_vocab.npy")
    monkeypatch.setenv("MTPLX_FRSPEC_DRAFT", "0")
    assert not EDGE0_FRSPEC & set(srv._server_runtime_env_overrides(args, {}))
    # without the head the list goes too
    state = forge.build_pack(src, out, tokenizer_from=root / "base", mtp_head="none")
    assert state["draft_vocab"] is None and not (out / "draft_vocab.npy").exists()
    # ids outside the vocabulary, or repeated, are refused
    (src / "mtp" / "draft_vocab.bin").write_bytes(np.array([1, 1], dtype="<i4").tobytes())
    with pytest.raises(forge.ForgeError, match="duplicate"):
        forge.build_pack(src, out, tokenizer_from=root / "base")
    (src / "mtp" / "draft_vocab.bin").write_bytes(np.array([vocab], dtype="<i4").tobytes())
    with pytest.raises(forge.ForgeError, match="outside the vocabulary"):
        forge.build_pack(src, out, tokenizer_from=root / "base")


def test_server_coerces_structure_verify_strategies(packs, capsys):
    from mtplx.server import openai as srv

    for strategy in ("capture_commit", "target_prefix", "graphbank"):
        args = argparse.Namespace(model=str(packs[2]), verify_strategy=strategy)
        srv._coerce_family_verify_strategy(args)
        assert args.verify_strategy == "batched"
    assert "edge0_latent verifies 'batched'" in capsys.readouterr().out
    args = argparse.Namespace(model=str(packs[2]), verify_strategy="sequential")
    srv._coerce_family_verify_strategy(args)
    assert args.verify_strategy == "sequential"
    assert srv._served_model_is_flash_next_shell(args) and not srv._served_model_type_is_qwen4_exp(args)
    assert srv._served_model_family(args) != "qwen4_exp"


# --------------------------------------------------------------------------- the real pack (header reads only)


@pytest.mark.skipif(not REAL_PACK, reason="MTPLX_EDGE0_PACK not set")
def test_real_base_head_fits_the_real_pack():
    base = forge.find_tokenizer_source()
    if base is None or not (base / edge0_mtp.HEAD_FILE).exists():
        pytest.skip("no base Flash Next pack with mtp.safetensors")
    cfg = json.loads((Path(REAL_PACK) / "config.json").read_text())
    report = edge0_mtp.check_head(base / edge0_mtp.HEAD_FILE, cfg)
    assert report["tensors"] == 58 and report["bits"] == [4, 8]
