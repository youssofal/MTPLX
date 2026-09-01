"""MLX-free structural gates for the vendored Steel QSA prefill consumer.

These run anywhere — no MLX, no Metal, no built extension — because the
things they pin are the ones that are expensive to discover late: the tier
order inside ``Attention.__call__``, the producer auto-gate that decides
whether the lane arms at all on M3, the env registration, the vendored
provenance and licenses, and the two renames (Metal library, namespace) that
keep a co-installed oMLX from colliding with this module.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mtplx" / "models" / "qwen4_exp.py"
WRAPPER_PATH = ROOT / "mtplx" / "kernels" / "qsa_prefill_direct.py"
PROFILES_PATH = ROOT / "mtplx" / "profiles.py"
EXT_DIR = ROOT / "native_extensions" / "qsa_kernels"
PKG_DIR = EXT_DIR / "mtplx_qsa_kernels"

MODEL_TEXT = MODEL_PATH.read_text(encoding="utf-8")
MODEL_TREE = ast.parse(MODEL_TEXT, filename=str(MODEL_PATH))
WRAPPER_TEXT = WRAPPER_PATH.read_text(encoding="utf-8")

# The oMLX tree this port was taken from. Recorded here as well as in the
# NOTICE files so a later "which upstream commit is this?" is one grep.
OMLX_REVISION = "dc312e6e905e03d21ef0c4a86289cbfa2cf857cc"


def _class_method_source(class_name: str, method_name: str) -> str:
    cls = next(
        node
        for node in MODEL_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    source = ast.get_source_segment(MODEL_TEXT, method)
    assert source is not None
    return source


def _function_source(name: str) -> str:
    node = next(
        item
        for item in MODEL_TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    source = ast.get_source_segment(MODEL_TEXT, node)
    assert source is not None
    return source


def test_attention_dispatch_order_is_flash_direct_gather_dense():
    """The one ordering the whole port depends on.

    MPP flash keeps first refusal (M4/M5 regression-free), the direct Steel
    kernel takes the seam M3 can actually reach, and both sit above the
    portable gather tier and the dense-mask reconstruction.
    """

    source = _class_method_source("Attention", "__call__")
    flash = source.index("qsa_prefill_flash(")
    direct = source.index("qsa_prefill_direct(")
    gather = source.index("_qsa_prefill_gather_attention(")
    dense = source.index("_qsa_blocks_to_dense_mask(")
    assert flash < direct < gather < dense


def test_direct_branch_is_independent_of_the_mpp_branch():
    """The direct tier must not import or probe the MPP kernel to run.

    If the direct support check were nested inside the MPP branch, M3 — where
    the MPP check is always False — would never reach it.
    """

    source = _class_method_source("Attention", "__call__")
    start = source.index("qsa_prefill_direct_supported(")
    branch = source[source.index("_qsa_prefill_direct_attention_enabled(") : start]
    assert "qsa_prefill_flash_supported" not in branch
    assert "qsa_indexer_select_nax_available" not in branch


def test_direct_branch_passes_logical_cache_views_not_capacity_backing():
    """params.kL is k.shape(2); it must describe the live cache, not spare
    capacity. The MPP branch deliberately takes cache.kv.keys and carries a
    separate total_tokens contract; this ABI has no logical-K parameter."""

    source = _class_method_source("Attention", "__call__")
    call_start = source.index("out = qsa_prefill_direct(")
    call = source[call_start : source.index("def _qsa_gather_call")]
    assert "cache.kv.keys" not in call
    assert "cache.kv.values" not in call
    positional = [line.strip() for line in call.splitlines()]
    assert "k," in positional
    assert "v," in positional


def test_direct_lane_has_its_own_engagement_counter():
    """A/B law: no benchmark number without proof the arm's code ran.

    The counters live in the extracted tier chooser, which is also what
    tests/test_qsa_prefill_direct_routing.py drives with callable fakes.
    """

    source = _function_source("_qsa_prefill_dispatch_tier")
    assert '_qsa_prefill_count("direct_kernel")' in source
    assert "direct_call()" in source


def test_producer_auto_gate_can_arm_without_nax():
    """The blocking wiring bug both reviewers flagged.

    Without this, the direct kernel is dead code with a green import: the
    ("flash_prefill", ...) tuple it consumes is only produced when the auto
    gate passes, and the gate used to be NAX-only.
    """

    node = next(
        item
        for item in MODEL_TREE.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "qsa_prefill_lane_auto_supported"
    )
    # Compare executable statements, not prose: the docstring names both
    # consumers and would decide the ordering assertion below.
    body = "\n".join(
        ast.get_source_segment(MODEL_TEXT, stmt) or ""
        for stmt in node.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    )
    assert "qsa_indexer_select_nax_available()" in body
    assert "qsa_prefill_direct_ready" in body
    # NAX first: an M4/M5 machine must not have its answer changed by whether
    # someone happened to build this extension.
    assert body.index("qsa_indexer_select_nax_available()") < body.index(
        "qsa_prefill_direct_ready"
    )


def test_master_kill_switch_still_outranks_the_direct_lane():
    enabled = _function_source("_qsa_prefill_enabled")
    assert 'raw in {"0", "false", "no", "off"}' in enabled
    assert "qsa_prefill_lane_auto_supported()" in enabled


def test_direct_tier_has_its_own_context_crossover():
    source = _function_source("_qsa_prefill_direct_attention_enabled")
    assert "_qsa_large_prefill_enabled(rows, total_tokens)" in source
    assert (
        "int(total_tokens) - int(rows) >= _qsa_prefill_direct_min_context()" in source
    )
    floor = _function_source("_qsa_prefill_direct_min_context")
    assert 'os.environ.get("MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT") or 32768' in floor


def test_direct_knobs_are_registered_for_operator_overrides():
    profiles = PROFILES_PATH.read_text(encoding="utf-8")
    for key in (
        "MTPLX_QSA_PREFILL_DIRECT",
        "MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_DIRECT_VALIDATE",
    ):
        assert f'"{key}"' in profiles


def test_wrapper_pins_the_measured_production_tiles():
    """oMLX's fast.py defaults are (128, 32); its production glue uses
    (64, 64), which is what was measured on M3 and what MTPLX packages."""

    assert "_KEY_TILE = 64" in WRAPPER_TEXT
    assert "_DIMENSION_TILE = 64" in WRAPPER_TEXT
    assert "key_tile=_KEY_TILE" in WRAPPER_TEXT
    assert "dimension_tile=_DIMENSION_TILE" in WRAPPER_TEXT


def test_wrapper_support_check_mirrors_the_cpp_unsupported_clauses():
    """Python "supported" must mean "will dispatch"; a gap either wastes the
    fast path or turns a static mismatch into a runtime throw."""

    for clause in (
        "queries.ndim != 4",
        "_SUPPORTED_DTYPES",
        "keys.dtype != queries.dtype",
        "(_BATCH, _Q_HEADS, _HEAD_DIM)",
        "(_BATCH, _KV_HEADS, _HEAD_DIM)",
        "_last_dim_contiguous",
        "pos_start_i < 0",
        "key_len != total_tokens_i",
        "int(key_tile) != _KEY_TILE or int(dimension_tile) != _DIMENSION_TILE",
        "int(compress_ratio) != _COMPRESS_RATIO",
        "int(block_topk) != _TOP_K_BLOCKS",
        "scale_f != _EXPECTED_SCALE",
        "total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS",
    ):
        assert clause in WRAPPER_TEXT, clause


def test_wrapper_converts_topk_explicitly_and_contiguously():
    """A bare reshape+cast would hand the kernel a strided view, and a
    negative int32 id would become a huge uint32."""

    assert "mx.contiguous(block_ids.astype(mx.uint32)[None, None])" in WRAPPER_TEXT


def test_wrapper_proves_the_metal_pipeline_on_first_dispatch():
    """abi_probe proves the nanobind casters; only an eval proves the
    metallib and the kernel specialization exist."""

    assert "_PIPELINE_PROVEN" in WRAPPER_TEXT
    assert "mx.eval(out)" in WRAPPER_TEXT
    assert "def qsa_prefill_direct_preflight" in WRAPPER_TEXT


def test_wrapper_never_falls_back_to_dense_after_dispatch():
    """Fail-closed before dispatch, fail-loud after. A silent retry would
    make an A/B arm report the fallback's numbers as the kernel's."""

    tree = ast.parse(WRAPPER_TEXT, filename=str(WRAPPER_PATH))
    direct = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "qsa_prefill_direct"
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(direct))


# --------------------------------------------------------------------------
# Vendored native module
# --------------------------------------------------------------------------


def test_only_the_qwen4_sparse_gqa_primitive_was_vendored():
    """oMLX's csrc carries seven kernel families; MTPLX takes one. Anything
    else is build, symbol, and packaging risk for no benefit here."""

    sources = sorted(p.name for p in EXT_DIR.glob("*.cpp"))
    assert sources == ["bindings.cpp", "qwen4_qsa_sparse_gqa.cpp"]
    metal = sorted(p.name for p in EXT_DIR.glob("*.metal"))
    assert metal == ["qwen4_qsa_sparse_gqa.metal"]
    headers = sorted(p.name for p in (EXT_DIR / "kernels").glob("*.h"))
    assert headers == ["steel_qwen4_qsa_sparse_gqa.h"]


def test_library_and_namespace_were_renamed_away_from_omlx():
    """A co-installed oMLX must not be able to satisfy get_library() for this
    module, and vice versa."""

    raw = (EXT_DIR / "qwen4_qsa_sparse_gqa.cpp").read_text(encoding="utf-8")
    # The provenance comment block legitimately names the old strings; only
    # the code must be free of them.
    cpp = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("//")
    )
    assert 'kMetalLibrary = "mtplx_qsa_kernels"' in cpp
    assert "omlx_glm_kernels" not in cpp
    assert "namespace mtplx::qsa_kernels" in cpp
    assert "namespace omlx" not in cpp
    assert "DEFINE_NAME(MTPLXQwen4QSASparseGQAAttention)" in cpp

    cmake_raw = (EXT_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    cmake = "\n".join(
        line for line in cmake_raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert 'MTPLX_QSA_METAL_LIBRARY "mtplx_qsa_kernels"' in cmake
    assert "omlx" not in cmake.lower()


def test_metal_include_order_is_preserved():
    """Upstream says the order is load-bearing: Steel's attention header
    provides Limits for the specialized kernel."""

    metal = (EXT_DIR / "qwen4_qsa_sparse_gqa.metal").read_text(encoding="utf-8")
    utils = metal.index("mlx/backend/metal/kernels/utils.h")
    steel = metal.index("steel/attn/kernels/steel_attention.h")
    params = metal.index("struct Qwen4QSASparseGQAParams")
    kernel = metal.index("kernels/steel_qwen4_qsa_sparse_gqa.h")
    assert utils < steel < params < kernel
    assert "clang-format off" in metal
    cpp = (EXT_DIR / "qwen4_qsa_sparse_gqa.cpp").read_text(encoding="utf-8")
    for field in (
        "int q_offset",
        "float scale",
        "int64_t Q_strides[3]",
        "int64_t O_strides[3]",
    ):
        assert field in metal
        assert field in cpp


def test_only_the_packaged_specialization_is_instantiated():
    metal = (EXT_DIR / "qwen4_qsa_sparse_gqa.metal").read_text(encoding="utf-8")
    instantiations = [
        line for line in metal.splitlines() if line.startswith("instantiate_")
    ]
    assert instantiations == [
        "instantiate_qwen4_sparse_gqa(float16, half, 64, 64);",
        "instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 64, 64);",
    ]


def test_threadgroup_ceiling_is_asserted_at_compile_time():
    """Issue #400: M2/M3 cap some kernels at 896 threads, not 1024. WM is
    fixed at 2 (64 threads) — a future retune must re-open this deliberately
    rather than produce an unlaunchable pipeline."""

    header = (EXT_DIR / "kernels" / "steel_qwen4_qsa_sparse_gqa.h").read_text(
        encoding="utf-8"
    )
    cpp = (EXT_DIR / "qwen4_qsa_sparse_gqa.cpp").read_text(encoding="utf-8")
    assert "static_assert(WM * 32 <= 896" in header
    assert "static_assert(wm * 32 <= 896" in cpp
    assert "constexpr int wm = 2;" in cpp
    # WM must not become a Python tuning knob: it is part of the compiled
    # kernel name and its MMA layout.
    assert "WM" not in WRAPPER_TEXT
    assert "warp" not in WRAPPER_TEXT.lower()


def test_native_build_pins_the_exact_nanobind_abi_and_mlx_domain():
    pyproject = (EXT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    assert '"nanobind==2.15.0"' in pyproject
    # Exactly one wheel: a range lets a PEP 517 isolated build resolve a
    # different 0.32.x than the serving venv holds, and this extension
    # links MLX's private C++ ABI.
    assert '"mlx==0.32.2"' in pyproject
    cmake = (EXT_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "NB_DOMAIN\n  mlx" in cmake
    # sys.executable must win over any framework Python CMake might find.
    setup = (EXT_DIR / "setup.py").read_text(encoding="utf-8")
    assert "Python_EXECUTABLE={sys.executable}" in setup
    assert "Python3_EXECUTABLE={sys.executable}" in setup


def test_extension_is_optional_and_packages_its_artifacts():
    setup = (EXT_DIR / "setup.py").read_text(encoding="utf-8")
    for artifact in ("*.so", "*.dylib", "*.metallib", "LICENSE.txt", "NOTICE"):
        assert f'"{artifact}"' in setup
    # A missing extension degrades; it never raises at import.
    assert "except Exception as exc:" in WRAPPER_TEXT
    assert "return None, _detach(exc)" in WRAPPER_TEXT


def test_licenses_and_provenance_are_recorded():
    for name in (
        "LICENSE.txt",
        "NOTICE",
        "MLX_LICENSE.txt",
        "MLX_SERVE_LICENSE.txt",
    ):
        assert (PKG_DIR / name).is_file(), name
    notice = (PKG_DIR / "NOTICE").read_text(encoding="utf-8")
    assert OMLX_REVISION in notice
    assert "Apache License" in (PKG_DIR / "LICENSE.txt").read_text(encoding="utf-8")
    # mlx-serve's MIT staging attribution must survive verbatim in the header.
    steel = (EXT_DIR / "kernels" / "steel_qwen4_qsa_sparse_gqa.h").read_text(
        encoding="utf-8"
    )
    assert "mlx-serve's MIT" in steel
    assert "David Dalcu" in steel
    assert "SPDX-License-Identifier: Apache-2.0" in steel

    root_notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert OMLX_REVISION in root_notice
    assert "native_extensions/qsa_kernels" in root_notice


def test_every_vendored_source_carries_the_spdx_tag():
    for path in list(EXT_DIR.glob("*.cpp")) + list(EXT_DIR.glob("*.h")) + list(
        EXT_DIR.glob("*.metal")
    ) + list((EXT_DIR / "kernels").glob("*.h")):
        head = path.read_text(encoding="utf-8")[:200]
        assert "SPDX-License-Identifier: Apache-2.0" in head, path.name


# --------------------------------------------------------------------------
# Contract parity and fail-closed state (Codex CHANGES_REQUIRED follow-up)
# --------------------------------------------------------------------------


def _wrapper_function_source(name: str) -> str:
    tree = ast.parse(WRAPPER_TEXT, filename=str(WRAPPER_PATH))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    source = ast.get_source_segment(WRAPPER_TEXT, node)
    assert source is not None
    return source


def _wrapper_function_body(name: str) -> str:
    """Executable statements only: prose legitimately names what the code
    must not call."""

    tree = ast.parse(WRAPPER_TEXT, filename=str(WRAPPER_PATH))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return "\n".join(
        ast.get_source_segment(WRAPPER_TEXT, stmt) or ""
        for stmt in node.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    )


def test_cpp_unsupported_enforces_the_logical_view_and_scale_contracts():
    """The native boundary must refuse what the Python wrapper refuses.

    This machine has no built ``.so``, so the parity that matters here is
    the source contract: a caller that reaches the symbol directly, or a
    later wrapper edit that drops a clause, must still hit ``std::
    invalid_argument`` instead of a kernel reading the wrong rows.
    """

    cpp = (EXT_DIR / "qwen4_qsa_sparse_gqa.cpp").read_text(encoding="utf-8")
    # Equality, not <=: params.kL IS k.shape(2), so Q is the suffix ending
    # exactly at the logical frontier (Python: key_len != total_tokens_i).
    assert "q_offset + q.shape(2) != k.shape(2)" in cpp
    assert "q_offset + q.shape(2) > k.shape(2)" not in cpp
    # The one production scale. 1/sqrt(256) == 1/16 is exactly representable,
    # so the equality needs no tolerance (Python: scale_f != _EXPECTED_SCALE).
    assert "scale != 0.0625f" in cpp
    # scale has to reach unsupported() to be checked there.
    assert "const array &selected, float scale, int q_offset" in cpp
    message = cpp[cpp.index("std::ostringstream msg;") :]
    assert "q_offset+M==K" in message
    assert "scale==0.0625" in message


def test_tier_chooser_orders_flash_direct_gather_dense():
    """The extracted chooser is what the routing tests drive; its order is
    the port's whole contract."""

    source = _function_source("_qsa_prefill_dispatch_tier")
    order = [
        source.index("flash_supported()"),
        source.index("direct_supported()"),
        source.index("if gather_enabled:"),
        source.index('_qsa_prefill_count("dense_fallback")'),
    ]
    assert order == sorted(order)
    for lane in ("flash_kernel", "direct_kernel", "gather_tier", "dense_fallback"):
        assert f'_qsa_prefill_count("{lane}")' in source


def test_readiness_enforces_the_mlx_build_receipt():
    """A .so built against a different mlx imports, probes, lists every
    symbol, and then mis-reads MLX's private structs."""

    assert "def _build_receipt_mismatch" in WRAPPER_TEXT
    mismatch = _wrapper_function_source("_build_receipt_mismatch")
    assert 'getattr(_EXT, "BUILT_AGAINST_MLX"' in mismatch
    assert 'getattr(mx, "__version__"' in mismatch
    assert "built != runtime" in mismatch
    reason = _wrapper_function_source("_lane_unavailable_reason")
    assert "_build_receipt_mismatch()" in reason
    assert "logger.warning" in reason


def test_a_failed_pipeline_proof_disables_the_lane_process_wide():
    """Symbol presence is not readiness, and a failed proof is terminal:
    otherwise the M3 auto-gate arms a producer for a consumer that cannot
    run, every request re-hits the same wall, and the fallback never
    engages."""

    assert "_PIPELINE_FAILED" in WRAPPER_TEXT
    assert "_PIPELINE_UNPROVEN" in WRAPPER_TEXT
    reason = _wrapper_function_source("_lane_unavailable_reason")
    assert "_PIPELINE_STATE == _PIPELINE_FAILED" in reason
    # The per-call support check answers from the same state, so a failed
    # proof also makes qsa_prefill_direct() raise instead of dispatching.
    assert "_lane_unavailable_reason()" in _wrapper_function_source(
        "qsa_prefill_direct_unsupported_reason"
    )
    ready = _wrapper_function_source("qsa_prefill_direct_ready")
    assert "_lane_eligible()" in ready
    # No recursion: the proof consults the eligibility helper, never ready().
    for name in ("_lane_unavailable_reason", "_prove_pipeline"):
        assert "qsa_prefill_direct_ready" not in _wrapper_function_body(name)
    preflight = _wrapper_function_body("qsa_prefill_direct_preflight")
    assert "_lane_eligible()" in preflight
    assert "qsa_prefill_direct_ready" not in preflight


def test_the_pipeline_proof_is_tracked_per_query_dtype():
    """The Metal kernel name embeds the query dtype, so one bfloat16 proof
    never licenses the float16 specialization."""

    assert "_PIPELINE_PROVEN_DTYPES" in WRAPPER_TEXT
    ready = _wrapper_function_body("qsa_prefill_direct_ready")
    assert "_prove_all_pipelines()" in ready
    prove_all = _wrapper_function_body("_prove_all_pipelines")
    assert "_SUPPORTED_DTYPES" in prove_all
    # The first-dispatch proof takes the dtype it is proving, so a float16
    # first call cannot ride a bfloat16 PROVEN flag.
    assert "def _prove_first_dispatch(out: mx.array, *, dtype: mx.Dtype)" in WRAPPER_TEXT
    assert "_prove_first_dispatch(out, dtype=queries.dtype)" in WRAPPER_TEXT


def test_failed_is_terminal_and_the_first_proof_is_serialized():
    """Check-then-set across an mx.eval: without a lock two callers both see
    UNPROVEN and a late success can overwrite FAILED."""

    assert "import threading" in WRAPPER_TEXT
    assert "_PIPELINE_LOCK = threading.RLock()" in WRAPPER_TEXT
    for name in ("_prove_pipeline", "_prove_first_dispatch"):
        assert "with _PIPELINE_LOCK:" in _wrapper_function_body(name), name
    success = _wrapper_function_body("_record_pipeline_success")
    assert "if _PIPELINE_STATE == _PIPELINE_FAILED:" in success
    assert "return" in success
