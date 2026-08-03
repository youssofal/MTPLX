"""Interactive onboarding flow for ``mtplx start``.

Three screens for first-time users (model -> mode -> interface/client), and a
"same as last time?" prompt for returning users. Choices persist to
``~/.mtplx/quickstart.json`` so the next run can offer the same defaults.

The module gracefully degrades to plain stdlib ``print`` and ``input`` when
``rich`` is not available, so the onboarding works even in minimal venvs.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.constants import (
    EXPECTED_ALL_PREQUANTIZED_MTP_KEYS,
    EXPECTED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_MTP_KEYS,
    EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS,
    expand_mtp_layer_keys,
)
from mtplx.default_models import (
    DefaultModelSelection,
    OPTIMIZED_QUALITY_DESCRIPTION,
    OPTIMIZED_QUALITY_LABEL,
    is_verified_default_model_ref,
    is_optimized_quality_model_ref,
    optimized_quality_model_ref,
    select_default_model,
)
from mtplx.profiles import DEFAULT_HF_MODEL_ID
from mtplx.server_urls import bind_label, is_wildcard_bind, local_url_for_bind

DEFAULT_HF_MODEL = DEFAULT_HF_MODEL_ID
STATE_PATH = Path("~/.mtplx/quickstart.json").expanduser()
OPTIMIZED_QUALITY_MODEL_MARKER = "qwen3.6-27b-mtplx-optimized-quality"
LEGACY_OPTIMIZED_MODEL_NAMES = frozenset(
    {
        "qwen3.6-27b-mtplx-optimized",
        "youssofal--qwen3.6-27b-mtplx-optimized",
    }
)

# Tier ranks for sorting scanned-model lists. Lower = surfaced higher in the
# picker. The intent is "verified first, runnable next, blocked last" so the
# user never has to scroll past unsupported entries to find a launchable one.
_TIER_RANK: dict[str, int] = {
    "verified": 0,
    "arch-compatible": 1,
    "needs-verification": 2,
    "mtp-invalid": 3,
    "mtp-missing": 4,
    "backend-pending": 5,
    "no-mtp": 6,
    "incompatible": 7,
    "unknown": 8,
}
LOCAL_SCAN_TIMEOUT_S = 3.0
LOCAL_SCAN_CLASSIFY_TIMEOUT_S = 4.0
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _pretty_path(value: str | Path | None) -> str:
    """Render a filesystem path with the user's home directory collapsed to ``~``.

    Returns the input unchanged when it isn't path-shaped (HF refs like
    ``namespace/name``) or when it doesn't live under ``$HOME``. The point is
    to stop the UI from screaming ``/Users/<me>/...`` at users who reasonably
    expect a portable product.
    """

    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    # Heuristic: an HF repo ref is exactly ``namespace/name`` with no leading
    # path indicator. Don't try to munge those.
    if not text.startswith(("/", "~", "./", "../")):
        return text
    try:
        path = Path(text).expanduser()
        home = Path.home()
        try:
            rel = path.relative_to(home)
        except ValueError:
            return text
        rel_str = str(rel)
        if rel_str in {"", "."}:
            return "~"
        return f"~/{rel_str}"
    except Exception:
        return text


def _verified_default_selection() -> DefaultModelSelection:
    return select_default_model()


def _verified_default_model() -> str:
    return _verified_default_selection().model


def _verified_default_label() -> str:
    return _verified_default_selection().label


def _optimized_quality_label() -> str:
    return f"{OPTIMIZED_QUALITY_LABEL}  ·  {OPTIMIZED_QUALITY_DESCRIPTION}"


def _model_display(value: str | Path | None) -> str:
    if is_optimized_quality_model_ref(value):
        return OPTIMIZED_QUALITY_LABEL
    return _pretty_path(value)


# ---------- state file ------------------------------------------------------
def _state_path() -> Path:
    env = os.environ.get("MTPLX_QUICKSTART_STATE")
    return Path(env).expanduser() if env else STATE_PATH


def load_state() -> dict | None:
    """Return the last saved quickstart state, or ``None`` if absent / invalid."""

    path = _state_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_state(state: dict) -> None:
    """Persist the chosen configuration. Adds a UTC timestamp."""

    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------- local-folder scanning ------------------------------------------
@dataclass(frozen=True)
class ScannedModel:
    """A model directory found while walking a user-supplied folder.

    ``tier`` is the normalized compatibility verdict, one of
    ``verified`` / ``arch-compatible`` / ``needs-verification`` /
    ``mtp-invalid`` / ``mtp-missing`` / ``backend-pending`` / ``no-mtp`` /
    ``incompatible`` / ``unknown``. The display layer turns it into a coloured
    badge. ``arch-compatible`` means launchable, not merely recognized.
    """

    path: Path
    tier: str
    arch_id: str | None
    architecture: str | None
    error: str | None = None


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file()


# Folder names that are known to be internal/auxiliary and not worth recursing
# into. ``blobs`` and ``refs`` are part of the HuggingFace cache layout where
# only ``snapshots/<commit>/`` contains model files; the rest are common
# project clutter.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".cache",
        "__pycache__",
        "node_modules",
        "blobs",
        "refs",
        "DerivedData",
    }
)


def _scan_for_models(
    root: Path,
    *,
    max_depth: int = 4,
    cap: int = 200,
    timeout_s: float = LOCAL_SCAN_TIMEOUT_S,
) -> list[Path]:
    """Walk ``root`` and return all directories containing ``config.json``.

    The user can point us at three kinds of paths:

    * A single model directory (e.g. ``~/models/Qwen-7B/``) — returns
      ``[root]`` immediately.
    * A flat parent of model directories (e.g. ``~/models/``) — returns each
      child that has a ``config.json``.
    * A nested layout like LM Studio (``<root>/<publisher>/<model>/``) or the
      HuggingFace cache (``<root>/models--*/snapshots/<hash>/``). We descend
      up to ``max_depth`` levels and stop at the first ``config.json`` on each
      branch so we don't double-count.

    ``cap`` bounds the result list so a misdirected scan into ``$HOME`` can't
    produce a 10k-line picker. ``timeout_s`` keeps parent-folder scans from
    making first-run setup feel dead on slow external/cloud-backed folders.
    """

    if _is_model_dir(root):
        return [root]

    results: list[Path] = []
    deadline = time.monotonic() + max(0.1, float(timeout_s)) if timeout_s else None

    def _expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    def _walk(p: Path, depth: int) -> None:
        if depth > max_depth or len(results) >= cap or _expired():
            return
        try:
            entries = sorted(p.iterdir(), key=lambda x: x.name.lower())
        except (PermissionError, OSError):
            return
        for child in entries:
            if len(results) >= cap or _expired():
                return
            if not child.is_dir():
                continue
            name = child.name
            if name in _SKIP_DIRS:
                continue
            # Hidden directories below the root are noise (``.DS_Store``,
            # ``.huggingface``). The root itself is allowed to be hidden — that
            # is how ``~/.lmstudio/models`` works.
            if name.startswith(".") and depth >= 0:
                continue
            if _is_model_dir(child):
                results.append(child)
                continue
            _walk(child, depth + 1)

    _walk(root, 0)
    return results


def _expected_embedded_mtp_keys(config: dict[str, Any]) -> set[str]:
    tcfg = config.get("text_config", config) if isinstance(config, dict) else {}
    n_layers = max(
        int(
            tcfg.get("mtp_num_hidden_layers")
            or tcfg.get("num_nextn_predict_layers")
            or config.get("num_nextn_predict_layers")
            or 0
        ),
        1,
    )
    if _is_qwen_moe_mtp_config(config):
        mtp_quant = config.get("mtplx_mtp_quantization", {})
        prequantized = isinstance(mtp_quant, dict) and bool(mtp_quant.get("prequantized"))
        if prequantized:
            return expand_mtp_layer_keys(EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS, n_layers)
        return expand_mtp_layer_keys(EXPECTED_QWEN_MOE_MTP_KEYS, n_layers)
    mtp_quant = config.get("mtplx_mtp_quantization", {})
    prequantized = isinstance(mtp_quant, dict) and bool(mtp_quant.get("prequantized"))
    quant_policy = str(mtp_quant.get("policy") or "") if isinstance(mtp_quant, dict) else ""
    if prequantized and quant_policy == "all":
        return expand_mtp_layer_keys(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS, n_layers)
    if prequantized:
        return expand_mtp_layer_keys(EXPECTED_PREQUANTIZED_MTP_KEYS, n_layers)
    return expand_mtp_layer_keys(EXPECTED_MTP_KEYS, n_layers)


def _is_qwen_moe_mtp_config(config: dict[str, Any]) -> bool:
    tcfg = config.get("text_config", config) if isinstance(config, dict) else {}
    markers = (
        str(config.get("model_type") or ""),
        str(tcfg.get("model_type") or ""),
        " ".join(str(item) for item in (config.get("architectures") or [])),
        " ".join(str(item) for item in (tcfg.get("architectures") or [])),
    )
    return any("qwen3_5_moe" in marker.lower() or "qwen3_5moe" in marker.lower() for marker in markers)


def _is_mtp_weight_key(key: str) -> bool:
    text = str(key)
    return text.startswith("mtp.") or text.startswith("language_model.mtp.")


def _normalize_mtp_weight_key(key: str) -> str:
    text = str(key)
    if text.startswith("language_model.mtp."):
        return "mtp." + text[len("language_model.mtp.") :]
    return text


def _scan_mtp_sidecar_exists(model_dir: Path, config: dict[str, Any]) -> bool:
    candidates: list[str] = []
    extra = config.get("mlx_lm_extra_tensors", {})
    if isinstance(extra, dict) and extra.get("mtp_file"):
        candidates.append(str(extra["mtp_file"]))
    candidates.extend(("mtp.safetensors", "mtp/weights.safetensors", "model-mtp.safetensors"))
    for rel in candidates:
        try:
            if (model_dir / rel).is_file():
                return True
        except OSError:
            continue
    return False


def _scan_embedded_mtp_keys(model_dir: Path) -> tuple[str, ...]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return ()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict):
        return ()
    return tuple(
        sorted(_normalize_mtp_weight_key(str(key)) for key in weight_map if _is_mtp_weight_key(str(key)))
    )


def _classify_scanned_model(model_dir: Path) -> ScannedModel:
    """Bucket a model directory into our four-tier compatibility model from
    config.json alone — never mmap any safetensors file.

    The scanner runs over arbitrary user folders (LM Studio caches, HF caches,
    half-downloaded directories) and at least one of those models is allowed
    to be partially evicted, broken, or APFS-dataless without crashing the
    whole picker. ``inspect_model`` mmaps ``mtp.safetensors`` to count tensors
    — a SIGBUS on that mmap kills the Python process and there is nothing
    Python-level can catch. So for the picker we synthesize a minimal stub
    inspection and run it through the same ``compatibility_for_inspection``
    verdict logic the rest of the runtime uses. The runtime layer does the
    full mmap when the user actually picks a model and produces a precise
    error there if the artifact is broken.
    """

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return ScannedModel(
            path=model_dir,
            tier="incompatible",
            arch_id=None,
            architecture=None,
            error="missing config.json",
        )

    try:
        raw = config_path.read_text(encoding="utf-8")
        config = json.loads(raw)
    except Exception as exc:
        return ScannedModel(
            path=model_dir,
            tier="unknown",
            arch_id=None,
            architecture=None,
            error=str(exc)[:80],
        )

    tcfg = config.get("text_config", config) if isinstance(config, dict) else {}
    archs = (
        (config.get("architectures") if isinstance(config, dict) else None)
        or tcfg.get("architectures")
        or []
    )
    architecture = archs[0] if archs else None
    model_type = tcfg.get("model_type") or (
        config.get("model_type") if isinstance(config, dict) else None
    )

    def _maybe_int(v: Any) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    mtp_num_hidden_layers = max(
        _maybe_int(tcfg.get("mtp_num_hidden_layers")),
        _maybe_int(tcfg.get("num_nextn_predict_layers")),
        _maybe_int(tcfg.get("num_mtp_modules")),
        _maybe_int(config.get("num_nextn_predict_layers")) if isinstance(config, dict) else 0,
        _maybe_int(config.get("num_mtp_modules")) if isinstance(config, dict) else 0,
    )

    # Runtime contract is a small JSON file — safe to read.
    contract_path = model_dir / "mtplx_runtime.json"
    runtime_contract_data: dict | None = None
    runtime_contract_error: str | None = None
    if contract_path.is_file():
        try:
            runtime_contract_data = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            runtime_contract_error = str(exc)[:80]

    # List ``model*.safetensors`` filenames without reading tensor data, so the
    # non-Qwen verified-runtime gate has the file presence it expects.
    try:
        model_files = tuple(sorted(p.name for p in model_dir.glob("model*.safetensors")))
    except OSError:
        model_files = ()
    sidecar_exists = _scan_mtp_sidecar_exists(model_dir, config if isinstance(config, dict) else {})
    embedded_mtp_keys = _scan_embedded_mtp_keys(model_dir)
    expected_embedded = _expected_embedded_mtp_keys(config if isinstance(config, dict) else {})
    embedded_gate = bool(embedded_mtp_keys) and set(embedded_mtp_keys) == expected_embedded
    mtp_artifact_exists = sidecar_exists or bool(embedded_mtp_keys)
    mtp_tensor_gate = sidecar_exists or embedded_gate

    class _StubMTP:
        # Never mmap safetensors here. Sidecar presence or complete embedded
        # ``mtp.*`` keys are enough for picker-level "can try this" UX; the
        # runtime still performs the full tensor gate before loading.
        exists = mtp_artifact_exists
        passes_tensor_gate = mtp_tensor_gate

    class _StubInspection:
        pass

    stub = _StubInspection()
    stub.model_dir = str(model_dir)
    stub.architecture = architecture
    stub.model_type = model_type
    stub.mtp_num_hidden_layers = mtp_num_hidden_layers
    stub.mtp = _StubMTP()
    stub.model_files = model_files
    stub.runtime_contract_data = runtime_contract_data
    stub.runtime_contract_error = runtime_contract_error
    stub.runtime_contract_path = str(contract_path) if contract_path.is_file() else None

    try:
        from mtplx.backends.registry import compatibility_for_inspection
    except Exception as exc:
        return ScannedModel(
            path=model_dir,
            tier="unknown",
            arch_id=None,
            architecture=architecture,
            error=str(exc)[:80],
        )

    try:
        verdict = compatibility_for_inspection(stub)
    except Exception as exc:
        return ScannedModel(
            path=model_dir,
            tier="unknown",
            arch_id=None,
            architecture=architecture,
            error=str(exc)[:80],
        )

    # Resolve the supported-arch set lazily so missing the import doesn't
    # crash the picker — the unknown bucket is fine in that pathological case.
    try:
        from mtplx.backends.registry import SUPPORTED_ARCH_IDS as _SUPPORTED
    except Exception:
        _SUPPORTED = set()

    raw_tier = verdict.tier
    runtime_status = verdict.runtime_compatibility
    artifact_missing = mtp_num_hidden_layers > 0 and not mtp_artifact_exists
    if raw_tier == "verified":
        tier = "verified"
    elif verdict.can_run or raw_tier == "family-compatible-unverified":
        tier = "arch-compatible"
    elif raw_tier == "architecture-compatible-but-unverified":
        # The picker's safety pass deliberately skips the safetensors mmap
        # that the runtime's verified gate needs for ``qwen3-next-mtp``. To
        # avoid showing the flagship as "Compatible (unverified)" in the
        # picker when it has a blessed ``mtplx_runtime.json`` we trust the
        # contract: contract present + parsed cleanly + arch in our supported
        # set ⇒ label as verified. The runtime layer still runs the full
        # tensor check before any model actually loads.
        if (
            verdict.runtime_contract is not None
            and verdict.arch_id is not None
            and verdict.arch_id in _SUPPORTED
            and runtime_contract_error is None
        ):
            tier = "verified"
        elif verdict.runtime_compatibility == "missing-mtp-weights":
            tier = "mtp-missing"
        elif artifact_missing:
            tier = "mtp-missing"
        elif verdict.runtime_compatibility == "invalid-mtp-tensor-layout":
            tier = "mtp-invalid"
        elif verdict.runtime_compatibility in {"needs-contract", "needs-grafting"}:
            tier = "needs-verification"
        elif verdict.runtime_compatibility == "recognized-backend-pending":
            tier = "backend-pending"
        else:
            tier = "needs-verification"
    elif raw_tier == "no-MTP":
        tier = "no-mtp"
    elif raw_tier == "incompatible-architecture":
        tier = "backend-pending" if runtime_status == "recognized-backend-pending" else "incompatible"
    else:
        tier = "unknown"

    return ScannedModel(
        path=model_dir,
        tier=tier,
        arch_id=verdict.arch_id,
        architecture=architecture,
    )


def _tier_badge(tier: str) -> tuple[str, str]:
    """Return ``(label, rich_style)`` for a compatibility tier."""

    if tier == "verified":
        return ("Verified", "bold green")
    if tier == "arch-compatible":
        return ("Runnable (unverified)", "yellow")
    if tier == "needs-verification":
        return ("Needs MTPLX verification", "yellow")
    if tier == "mtp-invalid":
        return ("MTP weights invalid", "yellow")
    if tier == "mtp-missing":
        return ("MTP weights missing", "yellow")
    if tier == "backend-pending":
        return ("Backend not runnable yet", "dim")
    if tier == "no-mtp":
        return ("No MTP head", "dim")
    if tier == "incompatible":
        return ("Unsupported architecture", "red")
    return ("Unknown", "dim")


def _scanned_model_options(
    models: list[ScannedModel],
    root: Path,
) -> list[tuple[str, str, str]]:
    """Convert scanned-model entries into ``(key, headline, subline)`` tuples
    suitable for ``_choice_panel``.

    Display path is rendered relative to ``root`` so the panel doesn't show
    the whole ``/Users/<me>/...`` prefix on every line.
    """

    options: list[tuple[str, str, str]] = []
    for index, model in enumerate(models, start=1):
        try:
            display_name = str(model.path.relative_to(root))
        except ValueError:
            display_name = _pretty_path(model.path)
        badge, _ = _tier_badge(model.tier)
        arch = model.architecture or model.arch_id or "unknown architecture"
        if model.error:
            subline = f"{badge}  ·  {model.error[:80]}"
        else:
            subline = f"{badge}  ·  {arch}"
        options.append((str(index), display_name, subline))
    return options


# ---------- rich helpers (with stdlib fallback) -----------------------------
def _console() -> Any | None:
    try:
        from rich.console import Console
    except ImportError:
        return None
    try:
        return Console()
    except Exception:
        return None


def _step_panel(
    *,
    step: int,
    total: int,
    title: str,
    options: list[tuple[str, str, str]],
) -> None:
    """Render a numbered onboarding step (Step N of M)."""

    _choice_panel(
        heading=f"Step {step} of {total}  ·  {title}",
        options=options,
        border_style="cyan",
    )


def _choice_panel(
    *,
    heading: str,
    options: list[tuple[str, str, str]],
    intro: str | None = None,
    border_style: str = "cyan",
) -> None:
    """Render a panel of numbered choices; falls back to plain text."""

    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print(f"  {heading}")
        print()
        if intro:
            print(f"  {intro}")
            print()
        for key, headline, subline in options:
            print(f"  {key}.  {headline}")
            if subline:
                print(f"      {subline}")
            print()
        return

    console = _console()
    if console is None:
        print()
        print(f"  {heading}")
        print()
        if intro:
            print(f"  {intro}")
            print()
        for key, headline, subline in options:
            print(f"  {key}.  {headline}")
            if subline:
                print(f"      {subline}")
            print()
        return

    body = Text()
    if intro:
        body.append(intro, style="")
        body.append("\n\n")
    for index, (key, headline, subline) in enumerate(options):
        if index > 0:
            body.append("\n\n")
        body.append(f"{key}.  ", style="bold cyan")
        body.append(headline, style="bold")
        if subline:
            body.append("\n    ")
            body.append(subline, style="dim")

    panel = Panel(
        body,
        title=Text(heading, style="bold"),
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Read a digit choice from stdin, looping until valid. Raises on Ctrl-C.

    The prompt is rendered as ``Type 1-N and press Enter [1]:`` so users with
    no prior CLI familiarity know exactly what action to take.
    """

    if choices:
        try:
            low = min(int(c) for c in choices)
            high = max(int(c) for c in choices)
            if low == high:
                hint = f"Type {low} and press Enter"
            else:
                hint = f"Type {low}-{high} and press Enter"
        except ValueError:
            hint = f"Type one of {', '.join(choices)} and press Enter"
    else:
        hint = "Type your choice and press Enter"

    suffix = f" [default {default}]" if default else ""
    while True:
        answer = input(f"  {hint}{suffix}: ").strip()
        if not answer and default:
            return default
        if answer in choices:
            return answer
        print(f"  please type one of: {', '.join(choices)}")


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {prompt}{suffix}: ").strip()
    if not answer and default:
        return default
    return answer


def _normalize_hf_repo_id(value: str) -> str:
    text = str(value or "").strip()
    prefix = "https://huggingface.co/"
    if text.startswith(prefix):
        text = text[len(prefix) :]
        text = text.split("?", 1)[0].split("#", 1)[0]
        for marker in ("/tree/", "/blob/", "/resolve/"):
            if marker in text:
                text = text.split(marker, 1)[0]
                break
    return text.strip("/")


def _hf_repo_id_error(value: str) -> str | None:
    text = _normalize_hf_repo_id(value)
    if not text:
        return "Please enter a Hugging Face repo id like namespace/name."
    if any(ch.isspace() for ch in text):
        return "That has spaces, so it is not a Hugging Face repo id."
    if ":" in text or text.startswith(("╭", "│", "╰")):
        return "That looks like pasted terminal output, not a Hugging Face repo id."
    if "/" not in text:
        return "Please include the namespace, for example trevon/Qwen3.5-27B-MLX-MTP."
    if text.count("/") != 1:
        return "Please enter only namespace/name, not a nested path."
    namespace, name = text.split("/", 1)
    if not namespace or not name:
        return "Please enter a Hugging Face repo id like namespace/name."
    if not _HF_REPO_ID_RE.fullmatch(text):
        return "Repo ids can only use letters, numbers, dot, dash, and underscore."
    if any(part.startswith(("-", ".")) or part.endswith(("-", ".")) for part in (namespace, name)):
        return "Repo id parts cannot start or end with dot or dash."
    if "--" in text or ".." in text:
        return "Repo ids cannot contain consecutive dashes or dots."
    return None


def _prompt_hf_repo_id(*, default: str = DEFAULT_HF_MODEL) -> str:
    saw_invalid = False
    while True:
        entered = _prompt_text(
            "Hugging Face repo id (namespace/name)",
            default=default if not saw_invalid else None,
        )
        candidate = _normalize_hf_repo_id(entered)
        error = _hf_repo_id_error(candidate)
        if error is None:
            return candidate
        saw_invalid = True
        print(f"  {error}")
        print("  Example: trevon/Qwen3.5-27B-MLX-MTP")
        print()


def _print_welcome() -> None:
    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print("Welcome to MTPLX. Three quick questions to get you set up.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    console = _console()
    if console is None:
        print()
        print("Welcome to MTPLX. Three quick questions to get you set up.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    body = Text()
    body.append("Welcome to MTPLX.\n\n", style="bold")
    body.append("Three quick questions to get you set up:\n", style="")
    body.append("  1. Which model?\n", style="dim")
    body.append("  2. Which runtime mode?\n", style="dim")
    body.append("  3. Browser chat, terminal chat, Pi, OpenCode, or Swival?\n\n", style="dim")
    body.append("For each step, type the number of your choice and press Enter.", style="italic")
    panel = Panel(
        body,
        title=Text("First-time setup", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def _print_summary(
    state: dict,
    *,
    title: str = "Ready to go",
    plain_heading: str = "Your quickstart configuration:",
) -> None:
    model_display = _model_display(state.get("model")) or "?"
    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print()
        print(f"  {plain_heading}")
        print(f"    Model:     {model_display}")
        print(f"    Mode:      {mode_label(state)}")
        print(f"    Interface: {interface_label(state.get('target'))}")
        print()
        return
    console = _console()
    if console is None:
        # Don't silently swallow the summary when rich is installed but
        # ``Console()`` couldn't initialize (no tty, weird stdout, etc.) —
        # fall back to the same plain-stdout layout as the import-fail path.
        print()
        print(f"  {plain_heading}")
        print(f"    Model:     {model_display}")
        print(f"    Mode:      {mode_label(state)}")
        print(f"    Interface: {interface_label(state.get('target'))}")
        print()
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    table.add_row("Model", model_display)
    table.add_row("Mode", mode_label(state))
    table.add_row("Interface", interface_label(state.get("target")))
    panel = Panel(
        table,
        title=Text(title, style="bold green"),
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------- screens ---------------------------------------------------------
_MAX_INSTALLED_PICKER_ROWS = 6


def _installed_models_for_screen() -> list[Any]:
    """Complete installs from the local cache for the 'On this Mac' group."""

    try:
        from mtplx.model_catalog import scan_installed_models

        return list(scan_installed_models())[:_MAX_INSTALLED_PICKER_ROWS]
    except Exception:
        return []


def _app_settings_model() -> str | None:
    """The macOS app's configured model, when the app has been set up."""

    try:
        from mtplx.app_settings import read_app_settings

        settings = read_app_settings()
    except Exception:
        return None
    return settings.model if settings is not None else None


def _installed_matches_model_ref(item: Any, ref: str | None) -> bool:
    if not ref:
        return False
    text = str(ref).strip()
    if not text:
        return False
    if str(item.path) == text or item.name == Path(text).expanduser().name:
        return True
    catalog = getattr(item, "catalog", None)
    if catalog is None:
        return False
    try:
        from mtplx.model_catalog import catalog_model_matching

        matched = catalog_model_matching(text)
    except Exception:
        return False
    return matched is not None and matched.id == catalog.id


def screen_model(
    *,
    configured: str | None = None,
    installed: list[Any] | None = None,
    app_model: str | None = None,
) -> str:
    """Render screen 1 and return the user's choice.

    Models already on this Mac (``installed``, from the model-cache scan)
    lead the list so a returning user never re-downloads what they have.
    The default selection prefers the MTPLX app's configured model when it
    is installed, then the verified default for this hardware.

    If ``configured`` is set and differs from the canonical default, it is
    surfaced for deliberate reuse. Pressing Enter still picks the verified
    default (or the app's model), so stale saved configs cannot silently
    keep an old model as the speed lane.
    """

    verified_selection = _verified_default_selection()
    verified_default = verified_selection.model
    verified_label = verified_selection.label
    show_configured = (
        bool(configured)
        and not is_verified_default_model_ref(configured)
        and not is_optimized_quality_model_ref(configured)
    )

    installed_rows = list(installed or [])
    # (title, detail, resolved value or sentinel)
    rows: list[tuple[str, str, str]] = []
    app_row_index: int | None = None
    verified_row_index: int | None = None
    verified_covered_by_install = False
    quality_covered_by_install = False
    for item in installed_rows:
        catalog = getattr(item, "catalog", None)
        title = str(item.display_name)
        covers_verified = (
            catalog is not None
            and catalog.hf_model_id == verified_selection.hf_model
        )
        if covers_verified:
            title = f"{title}  ·  verified default"
            verified_covered_by_install = True
        if catalog is not None and catalog.id == "optimized-quality":
            quality_covered_by_install = True
        rows.append((title, f"installed  ·  {_pretty_path(item.path)}", str(item.path)))
        if app_row_index is None and _installed_matches_model_ref(item, app_model):
            app_row_index = len(rows) - 1
        if covers_verified and verified_row_index is None:
            verified_row_index = len(rows) - 1
    if show_configured and not any(
        str(item.path) == str(configured) for item in installed_rows
    ):
        rows.append(
            ("Use your configured model", _pretty_path(configured), str(configured))
        )
    if not verified_covered_by_install:
        rows.append(("Verified default for this Mac", verified_label, verified_default))
        verified_row_index = len(rows) - 1
    if not quality_covered_by_install:
        rows.append(
            ("Optimized Quality", _optimized_quality_label(), "__quality__")
        )
    rows.append(
        (
            "Custom Hugging Face repo",
            "e.g. Qwen/Qwen3-Next-80B-A3B-Instruct",
            "__hf__",
        )
    )
    rows.append(
        (
            "Local folder",
            "e.g. ~/models/your-model  ·  or a parent like ~/.lmstudio/models",
            "__local__",
        )
    )

    options = [
        (str(index + 1), title, detail)
        for index, (title, detail, _value) in enumerate(rows)
    ]
    default_index = (
        app_row_index if app_row_index is not None else verified_row_index
    )
    default_choice = str((default_index or 0) + 1)

    _step_panel(step=1, total=3, title="Choose your model", options=options)
    valid_choices = [option[0] for option in options]
    choice = _prompt_choice("Select", valid_choices, default=default_choice)
    value = rows[int(choice) - 1][2]
    if value == "__quality__":
        return optimized_quality_model_ref()
    if value == "__hf__":
        return _prompt_hf_repo_id(default=verified_default)
    if value == "__local__":
        return _pick_local_model(default=str(configured) if show_configured else None)
    return value


def _pick_local_model(*, default: str | None) -> str:
    """Ask for a local folder. If it isn't a model directory itself, scan
    inside it and present a numbered list of candidate models with their
    four-tier compatibility verdict.

    Loops until the user either picks a model, types Ctrl-C, or falls back
    to the canonical HF default. Always returns a usable identifier.
    """

    pretty_default = _pretty_path(default) if default else None
    while True:
        entered = _prompt_text("Local folder path", default=pretty_default)
        if not entered:
            print("  Path is required. Falling back to verified default.")
            return _verified_default_model()

        root = Path(entered).expanduser().resolve()
        if not root.exists():
            print(f"  Path does not exist: {_pretty_path(root)}")
            print()
            continue
        if not root.is_dir():
            print(f"  Not a directory: {_pretty_path(root)}")
            print()
            continue

        # Single-model case: no scanning needed, just return.
        if _is_model_dir(root):
            return str(root)

        chosen = _scan_and_pick(root)
        if chosen is None:
            # User asked to retype the path — loop again.
            continue
        return chosen


def _scan_and_pick(root: Path) -> str | None:
    """Walk ``root`` for models, render the picker, and return the chosen
    absolute path. ``None`` means the user wants to type a different folder.
    """

    print(f"  Scanning {_pretty_path(root)} for model folders...")
    found = _scan_for_models(root)
    if found:
        print(f"  Found {len(found)} candidate model folder(s). Checking configs...")
    classified: list[ScannedModel] = []
    classify_deadline = time.monotonic() + LOCAL_SCAN_CLASSIFY_TIMEOUT_S
    for path in found:
        if classified and time.monotonic() >= classify_deadline:
            print(
                "  Compatibility scan is taking longer than expected; "
                f"showing {len(classified)} result(s)."
            )
            break
        classified.append(_classify_scanned_model(path))

    if not classified:
        print(f"  No models found under {_pretty_path(root)}.")
        print(
            "  (a model directory has a config.json at its root — "
            "for LM Studio that's <publisher>/<repo-name>/)"
        )
        print()
        return None

    classified.sort(key=lambda m: (_TIER_RANK.get(m.tier, 99), str(m.path).lower()))

    cap = 24
    visible = classified[:cap]
    hidden = len(classified) - cap

    options = _scanned_model_options(visible, root)
    options.append(
        (
            str(len(visible) + 1),
            "Type a different folder path",
            "Re-enter a model directory or a different parent.",
        )
    )

    verified = sum(1 for m in classified if m.tier == "verified")
    runnable = sum(1 for m in classified if m.tier == "arch-compatible")
    needs = sum(1 for m in classified if m.tier == "needs-verification")
    missing = sum(1 for m in classified if m.tier == "mtp-missing")
    intro = (
        f"Found {len(classified)} model(s) under {_pretty_path(root)}  ·  "
        f"{verified} verified, {runnable} runnable unverified"
    )
    if needs:
        intro += f", {needs} need verification"
    if missing:
        intro += f", {missing} missing MTP weights"
    if hidden > 0:
        intro += f"  (showing first {cap}; {hidden} not shown)"

    _choice_panel(
        heading="Pick a model",
        intro=intro,
        options=options,
        border_style="cyan",
    )

    valid = [opt[0] for opt in options]
    default_choice = "1" if visible else None
    choice = _prompt_choice("Select", valid, default=default_choice)

    idx = int(choice) - 1
    if idx == len(visible):
        return None
    return str(visible[idx].path)


def screen_mode() -> tuple[str, bool]:
    """Return (profile_name, max_mode_flag).

    Quickstart exposes the explicit product choices:

      Sustained     : native-MTP long-context path, normal fan controller
      Sustained Max : Sustained path, fans pinned 100% while running
      Burst         : old max-fan native-MTP burst lane, short contexts only

    The Stable/safe profile remains available through explicit flags, but it
    is no longer part of the default onboarding path.
    """

    _step_panel(
        step=2,
        total=3,
        title="Choose a runtime mode",
        options=[
            (
                "1",
                "Sustained  ·  long-context safe, normal fan controller",
                "Chunked prefill, no full-prompt logits, and dynamic paged KV. Pick this for large files, long documents, coding contexts, or 16K-200K prompts.",
            ),
            (
                "2",
                "Sustained Max  ·  Sustained + fans pinned at 100%",
                "Same long-context-safe Sustained runtime, plus ThermalForge pins the fans while MTPLX runs and restores them after shutdown. Needs ThermalForge installed.",
            ),
            (
                "3",
                "Burst  [not recommended; max 8K context]",
                "Old max-fan performance-cold lane. Fastest headline burst for short prompts and benchmarks only; avoid for long documents or coding contexts.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2", "3"], default="1")
    if choice == "2":
        return "sustained", True
    if choice == "3":
        return "performance-cold", True
    return "sustained", False


def _surface_url(host: str, port: int, *, path: str = "") -> str:
    raw_host = str(host or "").strip()
    base = local_url_for_bind(raw_host, int(port), path=path)
    if is_wildcard_bind(raw_host):
        return f"{bind_label(raw_host, int(port))} · local {base}"
    return base


def screen_interface(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> str:
    """Return the target string (``openwebui``, ``terminal``, ``pi``, ``opencode``, ``swival``, or ``dashboard``)."""

    chat_url = _surface_url(host, port)
    dashboard_url = _surface_url(host, port, path="/dashboard")
    _step_panel(
        step=3,
        total=3,
        title="Where do you want to chat?",
        options=[
            (
                "1",
                f"Web UI [browser at {chat_url}/]",
                "Markdown rendering · live tokens-per-second · inference settings sidebar.",
            ),
            (
                "2",
                "CLI [this terminal]",
                "Streamed answers with rich styling and a stats footer.",
            ),
            (
                "3",
                "Connect to Pi [coding agent]",
                "Writes Pi's MTPLX model config and starts the local OpenAI-compatible server.",
            ),
            (
                "4",
                "Connect to OpenCode Desktop [coding agent]",
                "Writes OpenCode's MTPLX provider config with raw reasoning and starts the server.",
            ),
            (
                "5",
                "Connect to Swival [coding agent]",
                "Prints the Swival generic-provider command and starts the server.",
            ),
            (
                "6",
                f"Live Dashboard [browser at {dashboard_url}]",
                "Live TPS gauge · acceptance · cache · memory · fans · request log. Drive load from any client (Web UI, Pi, OpenCode, hippo).",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2", "3", "4", "5", "6"], default="1")
    if choice == "1":
        return "openwebui"
    if choice == "2":
        return "terminal"
    if choice == "3":
        return "pi"
    if choice == "4":
        return "opencode"
    if choice == "5":
        return "swival"
    return "dashboard"


# Targets for which a separate "also open the live dashboard?" companion
# question makes sense. Terminal/CLI runs the model in-process with no
# server, so the dashboard can't attach. "dashboard" already IS the
# dashboard target, no point asking again.
DASHBOARD_COMPANION_TARGETS = ("openwebui", "pi", "opencode", "swival", "hermes")


def screen_dashboard_companion(
    target: str,
    *,
    default: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> bool:
    """Offer to open the live Dashboard alongside the chosen client.

    Returns True iff the user picks "Yes". Only call this for targets in
    ``DASHBOARD_COMPANION_TARGETS`` — for terminal/cli (no server) and
    "dashboard" (already opens it) the question is non-sensical.
    """

    if target not in DASHBOARD_COMPANION_TARGETS:
        return False
    dashboard_url = _surface_url(host, port, path="/dashboard")
    _step_panel(
        step=4,
        total=4,
        title="Also open the Live Dashboard in your browser?",
        options=[
            (
                "1",
                f"Yes [opens {dashboard_url} in a second tab]",
                "Live TPS gauge, acceptance, cache, memory, fans, request log. "
                "Updates in real time while your chosen client drives load.",
            ),
            (
                "2",
                "No, just the chosen client",
                "Skip the dashboard. You can always run `mtplx dashboard` "
                "later against the running server.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1" if default else "2")
    return choice == "1"


def screen_tuning_offer() -> bool:
    """Return whether the first-run interactive flow should tune depth now."""

    _step_panel(
        step=4,
        total=4,
        title="Tune MTPLX for this Mac?",
        options=[
            (
                "1",
                "Run tuning [recommended]",
                "Tests AR, D1, D2, and D3 on a short coding prompt, then saves the fastest depth for this model and Mac. This takes a few minutes; close heavy background apps for cleaner results. Fans may get loud and will be restored after tuning.",
            ),
            (
                "2",
                "Skip for now",
                "Start MTPLX with the default depth. You can run mtplx tune later.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    return choice == "1"


def screen_server_surface(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    default_open_browser: bool = False,
) -> bool:
    """Return whether the server flow should open the browser chat."""

    raw_host = str(host or "").strip()
    base = local_url_for_bind(raw_host, int(port))
    surface = (
        f"{bind_label(raw_host, int(port))} · local {base}"
        if is_wildcard_bind(raw_host)
        else base
    )
    _step_panel(
        step=3,
        total=3,
        title="How should the server start?",
        options=[
            (
                "1",
                f"API server only [{surface}/v1]",
                "Starts the OpenAI-compatible endpoint and leaves this terminal attached to the server logs.",
            ),
            (
                "2",
                f"Open browser chat too [{surface}/]",
                "Starts the same server and opens the local MTPLX chat UI after startup.",
            ),
        ],
    )
    default = "2" if default_open_browser else "1"
    choice = _prompt_choice("Select", ["1", "2"], default=default)
    return choice == "2"


def run_onboarding_screens(
    *,
    configured_model: str | None = None,
    open_dashboard_override: bool | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> dict:
    """Walk all three screens and return the chosen state dict.

    When the user picks a fan-backed mode but no fan controller is detected,
    this function offers to install ThermalForge automatically. If install is
    declined or fails, the wizard falls back to Sustained without fan boost so
    the rest of the pipeline does not promise a fan-backed mode it cannot
    deliver.

    ``open_dashboard_override`` controls the dashboard companion prompt:
    ``True``/``False`` skip the question; ``None`` (default) asks.
    """

    model = screen_model(
        configured=configured_model,
        installed=_installed_models_for_screen(),
        app_model=_app_settings_model(),
    )
    profile, max_mode = screen_mode()
    if max_mode and not ensure_thermal_control_installed():
        profile = "sustained"
        max_mode = False
    target = screen_interface(host=host, port=port)
    # Only ask about the dashboard companion for targets that spawn a
    # server (openwebui/pi/opencode/swival). Terminal runs in-process with
    # no server, "dashboard" already opens the dashboard, so neither
    # benefits from the companion question.
    if open_dashboard_override is not None:
        open_dashboard = bool(open_dashboard_override) and target in DASHBOARD_COMPANION_TARGETS
    else:
        open_dashboard = screen_dashboard_companion(target, host=host, port=port)
    state = {
        "model": model,
        "profile": profile,
        "max": max_mode,
        "target": target,
        "open_dashboard": open_dashboard,
    }
    if is_verified_default_model_ref(model):
        state["model_selection"] = _verified_default_selection().to_dict()
    return state


def run_serve_onboarding_screens(
    *,
    configured_model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    default_open_browser: bool = False,
) -> dict:
    """Walk the advanced server setup screens and return the chosen state."""

    model = screen_model(
        configured=configured_model,
        installed=_installed_models_for_screen(),
        app_model=_app_settings_model(),
    )
    profile, max_mode = screen_mode()
    if max_mode and not ensure_thermal_control_installed():
        profile = "sustained"
        max_mode = False
    open_browser = screen_server_surface(
        host=host,
        port=port,
        default_open_browser=default_open_browser,
    )
    target = "openwebui" if open_browser else "server"
    # The serve flow doesn't spawn a coding-agent client, but it still
    # spawns a server — so the dashboard companion is meaningful for the
    # "Open browser chat too" branch. For "API server only" we skip the
    # question (the user wanted minimal UI).
    open_dashboard = (
        screen_dashboard_companion(target, host=host, port=port)
        if target == "openwebui"
        else False
    )
    state = {
        "model": model,
        "profile": profile,
        "max": max_mode,
        "target": target,
        "open_browser": open_browser,
        "open_dashboard": open_dashboard,
    }
    if is_verified_default_model_ref(model):
        state["model_selection"] = _verified_default_selection().to_dict()
    return state


def ensure_thermal_control_installed() -> bool:
    """Detect a fan controller; offer to install ThermalForge if absent.

    Returns ``True`` when fan control is available after this call (so a
    fan-backed mode is honest), ``False`` otherwise. The caller is expected to drop
    ``args.max`` when this returns ``False``.

    The chooser screen mirrors the rest of the onboarding (numbered options,
    ``Type 1-N and press Enter`` prompt) so the UX is consistent — no more
    Y/N prompt sandwiched between numbered screens.
    """

    from mtplx.thermal import detect_thermal_control, install_thermal_control

    detection = detect_thermal_control()
    if detection.get("available"):
        return True

    _choice_panel(
        heading="Fan mode setup  ·  ThermalForge",
        intro=(
            "Fan-backed modes need a fan controller. ThermalForge (free, open "
            "source) is not installed on this Mac yet."
        ),
        options=[
            (
                "1",
                "Install ThermalForge now (recommended)",
                "Builds from source via git + Xcode CLI tools. One sudo password prompt.",
            ),
            (
                "2",
                "Skip — continue without fan boost",
                "No fan boost, but everything else still works.",
            ),
        ],
        border_style="yellow",
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    if choice == "2":
        _print_install_skipped()
        return False

    try:
        from rich.console import Console

        console = Console()
        console.rule("Installing ThermalForge", style="cyan")
        console.print(
            "[dim]Streaming git + swift + sudo output. "
            "Enter your password if prompted.[/dim]"
        )
    except Exception:
        print("\n--- Installing ThermalForge ---")
        print("Streaming git + swift + sudo output. Enter your password if prompted.")

    result = install_thermal_control()
    _print_install_result(result)
    return bool(result.get("ok") and result.get("daemon_ok") is not False)


def _print_install_skipped() -> None:
    try:
        from rich.console import Console

        console = Console()
        console.print(
            "[yellow]  Skipped. Continuing without fan boost — "
            "the selected fan-backed mode will run without fan boost.[/yellow]\n"
        )
    except Exception:
        print("  Skipped. Continuing without fan boost.\n")


def _print_install_result(result: dict) -> None:
    ok = bool(result.get("ok"))
    daemon_ok = result.get("daemon_ok")
    message = result.get("message") or ("ThermalForge installed." if ok else "Install failed.")

    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        prefix = "[ok]" if ok and daemon_ok is not False else "[partial]" if ok else "[fail]"
        print(f"\n  {prefix} {message}\n")
        return

    console = _console()
    if console is None:
        prefix = "[ok]" if ok and daemon_ok is not False else "[partial]" if ok else "[fail]"
        print(f"\n  {prefix} {message}\n")
        return

    if ok and daemon_ok is not False:
        title = Text("ThermalForge ready", style="bold green")
        border = "green"
    elif ok:
        title = Text("ThermalForge installed (daemon pending)", style="bold yellow")
        border = "yellow"
    else:
        title = Text("ThermalForge install did not complete", style="bold red")
        border = "red"
    panel = Panel(
        Text(message, style=""),
        title=title,
        title_align="left",
        border_style=border,
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------- top-level flow --------------------------------------------------
def _quickstart_state_is_reusable(last: dict) -> bool:
    """Return whether a saved state should be offered by Quickstart.

    Stable/safe remains a supported explicit profile, but Quickstart no longer
    advertises or reuses it as the default consumer path. The current wizard
    choices are Sustained, Sustained Max, and Burst; old Medium saved states
    are intentionally re-onboarded so users see the new tradeoff copy.
    """

    model = str(last.get("model") or "").strip()
    target = str(last.get("target") or "")
    profile = last.get("profile")
    max_mode = bool(last.get("max"))
    normalized_model = model.replace("_", "-").lower()
    normalized_model_name = Path(model).expanduser().name.replace("_", "-").lower()
    if (
        OPTIMIZED_QUALITY_MODEL_MARKER in normalized_model
        or normalized_model == "youssofal/qwen3.6-27b-mtplx-optimized"
        or normalized_model_name in LEGACY_OPTIMIZED_MODEL_NAMES
    ):
        return False
    if profile == "performance-cold" and not max_mode:
        return False
    if profile not in {"performance-cold", "sustained"}:
        return False
    if target not in {
        "openwebui",
        "open-webui",
        "web",
        "terminal",
        "cli",
        "pi",
        "pie",
        "opencode",
        "open-code",
        "oc",
        "swival",
        "sv",
        "hermes",
        "hermes-agent",
        "dashboard",
        "live-dashboard",
        "live",
    }:
        return False
    if not model or "\n" in model or "\r" in model:
        return False
    if model.startswith(("Last login:", "╭", "│", "╰")) or "Use the same configuration?" in model:
        return False
    if _hf_repo_id_error(model) is None:
        return True
    expanded = Path(model).expanduser()
    if expanded.exists():
        return True
    return model.startswith(("/", "~", "./", "../", "models/"))


def _normalize_quickstart_state(last: dict) -> dict:
    """Refresh saved verified-default refs while preserving custom models."""

    if not is_verified_default_model_ref(last.get("model")):
        return last
    selection = _verified_default_selection()
    refreshed = dict(last)
    refreshed["model"] = selection.model
    refreshed["model_selection"] = selection.to_dict()
    return refreshed


def confirm_same_as_last(last: dict) -> bool:
    """Ask the user whether to reuse the last configuration."""

    model_display = _model_display(last.get("model")) or "?"
    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print()
        print("  Last time you used:")
        print(f"    Model:     {model_display}")
        print(f"    Mode:      {mode_label(last)}")
        print(f"    Interface: {interface_label(last.get('target'))}")
        print()
        answer = input("  Use the same configuration? [Y/n] ").strip().lower()
        return answer in {"", "y", "yes", "same"}

    console = _console()
    if console is None:
        print()
        print("  Last time you used:")
        print(f"    Model:     {model_display}")
        print(f"    Mode:      {mode_label(last)}")
        print(f"    Interface: {interface_label(last.get('target'))}")
        print()
        answer = input("  Use the same configuration? [Y/n] ").strip().lower()
        return answer in {"", "y", "yes", "same"}

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    table.add_row("Model", model_display)
    table.add_row("Mode", mode_label(last))
    table.add_row("Interface", interface_label(last.get("target")))
    panel = Panel(
        table,
        title=Text("Welcome back", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()
    answer = input("  Use the same configuration? [Y/n] ").strip().lower()
    return answer in {"", "y", "yes", "same"}


def _detect_running_daemon() -> Any | None:
    """Probe for an attachable MTPLX daemon; never raises.

    Honors the ``MTPLX_START_ATTACH_PROBE=off`` kill switch inside
    ``detect_attachable_daemon`` so tests and automation stay hermetic.
    """

    try:
        from mtplx.daemon_client import detect_attachable_daemon

        return detect_attachable_daemon()
    except Exception:
        return None


def screen_attach_running_daemon(daemon: Any) -> bool:
    """Offer to chat with an already-running daemon instead of setting up.

    Returns ``True`` when the user wants to attach. The running model is
    already in memory, so attaching is the zero-cost, zero-surprise path
    and therefore the default.
    """

    owner = "the MTPLX app" if getattr(daemon, "owned_by_app", False) else "a terminal"
    model = getattr(daemon, "model", None) or "a model"
    _choice_panel(
        heading="MTPLX is already running",
        intro=(
            f"{model}  ·  port {daemon.port}  ·  started by {owner}. "
            "The model is already loaded."
        ),
        options=[
            (
                "1",
                "Chat with the running model",
                "Attach to the running server — nothing new loads.",
            ),
            (
                "2",
                "Set up something else",
                "Continue to model / mode / interface setup.",
            ),
        ],
        border_style="green",
    )
    return _prompt_choice("Select", ["1", "2"], default="1") == "1"


def _returning_user_decision(last: dict, *, app_model: str | None) -> str:
    """Returning-user choice: ``same`` | ``app`` | ``fresh``.

    When the MTPLX app has a configured model that differs from the saved
    CLI state, "Same as the MTPLX app" appears alongside "same as last
    time" so app-first users get their app setup in one keystroke.
    """

    show_app_option = bool(app_model) and not _models_equivalent(
        app_model, last.get("model")
    )
    if not show_app_option:
        return "same" if confirm_same_as_last(last) else "fresh"
    _choice_panel(
        heading="Welcome back",
        options=[
            (
                "1",
                "Same as last time",
                f"{_model_display(last.get('model')) or '?'}  ·  "
                f"{mode_label(last)}  ·  {interface_label(last.get('target'))}",
            ),
            (
                "2",
                "Same as the MTPLX app",
                f"{_model_display(app_model)}  ·  keeps your mode and interface",
            ),
            (
                "3",
                "Set up fresh",
                "Walk through model / mode / interface again.",
            ),
        ],
        border_style="cyan",
    )
    choice = _prompt_choice("Select", ["1", "2", "3"], default="1")
    if choice == "2":
        return "app"
    if choice == "3":
        return "fresh"
    return "same"


def _models_equivalent(lhs: str | None, rhs: str | None) -> bool:
    left = str(lhs or "").strip()
    right = str(rhs or "").strip()
    if not left or not right:
        return left == right
    if left == right:
        return True
    try:
        from mtplx.model_catalog import catalog_model_matching

        left_match = catalog_model_matching(left)
        right_match = catalog_model_matching(right)
    except Exception:
        return False
    return (
        left_match is not None
        and right_match is not None
        and left_match.id == right_match.id
    )


def run_quickstart_flow(
    *,
    fresh: bool = False,
    configured_model: str | None = None,
    open_dashboard_override: bool | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> dict | None:
    """Decide between attach, 'same as last time', or fresh onboarding.

    Returns the chosen config dict (with keys ``model``, ``profile``,
    ``max``, ``target``, ``open_dashboard``), an attach request
    (``{"attach": {...}}``) when the user chose to chat with an
    already-running daemon, or ``None`` if the user aborted (Ctrl-C / EOF).

    ``open_dashboard_override`` lets the CLI ``--open-dashboard`` /
    ``--no-open-dashboard`` flags bypass the wizard companion question:
    ``True`` skips the prompt and forces open, ``False`` skips and
    forces closed, ``None`` (default) asks.
    """

    last = None if fresh else load_state()
    if last is not None and not _quickstart_state_is_reusable(last):
        last = None
    refreshed_default_state = False
    if last is not None:
        normalized = _normalize_quickstart_state(last)
        refreshed_default_state = normalized != last
        last = normalized
    try:
        daemon = _detect_running_daemon()
        if daemon is not None and screen_attach_running_daemon(daemon):
            return {
                "attach": {
                    "host": daemon.host,
                    "port": daemon.port,
                    "model": daemon.model,
                    "owned_by_app": daemon.owned_by_app,
                }
            }

        def reuse_state(state: dict) -> dict:
            nonlocal refreshed_default_state
            # If the saved state says a fan-backed mode but ThermalForge
            # has gone away (uninstalled, new machine), re-offer install
            # rather than silently dump a JSON warning at runtime.
            if state.get("max") and not ensure_thermal_control_installed():
                state = dict(state)
                state["profile"] = "sustained"
                state["max"] = False
                save_state(state)
            elif refreshed_default_state:
                save_state(state)
            if open_dashboard_override is not None:
                state = dict(state)
                state["open_dashboard"] = bool(open_dashboard_override)
            return state

        if last and not fresh:
            decision = _returning_user_decision(
                last, app_model=_app_settings_model()
            )
            if decision == "same":
                return reuse_state(last)
            if decision == "app":
                app_state = dict(last)
                app_state["model"] = str(_app_settings_model())
                # The selection metadata described the previous model.
                app_state.pop("model_selection", None)
                save_state(app_state)
                refreshed_default_state = False
                return reuse_state(app_state)
        else:
            _print_welcome()
        choice = run_onboarding_screens(
            configured_model=configured_model,
            open_dashboard_override=open_dashboard_override,
            host=host,
            port=port,
        )
        save_state(choice)
        _print_summary(choice)
        return choice
    except (KeyboardInterrupt, EOFError):
        try:
            print()
        except Exception:  # pragma: no cover - stdout is closed
            pass
        return None


def _print_server_welcome() -> None:
    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print("MTPLX server setup. Three quick questions before the server starts.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    console = _console()
    if console is None:
        print()
        print("MTPLX server setup. Three quick questions before the server starts.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    body = Text()
    body.append("MTPLX server setup.\n\n", style="bold")
    body.append("Three questions before the OpenAI-compatible server starts:\n", style="")
    body.append("  1. Which model?\n", style="dim")
    body.append("  2. Which runtime mode?\n", style="dim")
    body.append("  3. API only, or also open the browser chat?\n\n", style="dim")
    body.append("For each step, type the number of your choice and press Enter.", style="italic")
    panel = Panel(
        body,
        title=Text("Server setup", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def run_serve_flow(
    *,
    fresh: bool = False,
    configured_model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    default_open_browser: bool = False,
) -> dict | None:
    """Interactive setup for bare ``mtplx serve``.

    Unlike ``run_quickstart_flow``, this intentionally does not offer "same as
    last time" reuse: ``serve`` is the advanced server command, so a bare
    interactive invocation should always let the user pick the server model and
    runtime mode before binding a port.
    """

    _ = fresh  # Symmetric with quickstart; serve always starts from choices.
    try:
        _print_server_welcome()
        choice = run_serve_onboarding_screens(
            configured_model=configured_model,
            host=host,
            port=port,
            default_open_browser=default_open_browser,
        )
        _print_summary(
            choice,
            title="Server ready",
            plain_heading="Your server configuration:",
        )
        return choice
    except (KeyboardInterrupt, EOFError):
        try:
            print()
        except Exception:  # pragma: no cover - stdout is closed
            pass
        return None


# ---------- label helpers ---------------------------------------------------
def mode_label(state: dict) -> str:
    profile = state.get("profile", "safe")
    if state.get("max") and profile == "sustained":
        return "Sustained Max  ·  long-context path + fans pinned at 100%"
    if state.get("max") and profile == "performance-cold":
        return "Burst  ·  max-fan short-context lane  ·  max 8K context"
    if profile == "performance-cold":
        return "Performance-cold  ·  legacy burst path without fan boost"
    if profile == "sustained":
        return "Sustained  ·  long-context native-MTP path  ·  normal fan controller"
    if profile in {"safe", "stable"}:
        return "Stable  ·  long-reply exact/staged path  ·  no fan control"
    return str(profile)


def interface_label(target: str | None) -> str:
    if target in ("openwebui", "open-webui", "web"):
        return "Web UI  ·  browser"
    if target in ("server", "api", "api-server"):
        return "API server  ·  no browser"
    if target in ("cli", "terminal"):
        return "CLI  ·  this terminal"
    if target in ("pi", "pie"):
        return "Pi  ·  coding agent"
    if target in ("opencode", "open-code", "oc"):
        return "OpenCode Desktop  ·  coding agent"
    if target in ("swival", "sv"):
        return "Swival  ·  coding agent"
    if target in ("dashboard", "live-dashboard", "live"):
        return "Live Dashboard  ·  browser"
    return target or "?"
