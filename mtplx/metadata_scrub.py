"""Strip machine-identifying provenance from artifact metadata before publish.

Forge stamps ``mtplx_runtime.json`` with the absolute paths it read and wrote
(``forge_inputs``), plus the operator's intended Hugging Face repo. Those are
useful locally and leak a home directory once uploaded. The helpers here
normalize such values without discarding the provenance that a downstream user
actually needs (source repo, source SHA, recipe, versions).

Pure standard library so it can run anywhere a manifest can be read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


#: Provenance keys removed outright — they name only local locations.
DROPPED_PROVENANCE_KEYS = ("intended_hf_repo",)

#: Replacement stand-in for a scrubbed absolute path.
REDACTED_PATH = "<redacted>"

#: Keys whose values are paths: any absolute path under them is local.
_PATH_KEY_RE = re.compile(r"(^|_)(path|dir|directory|file|root|location)s?$")

#: Prefixes that name a machine: a home directory or a per-user temp folder.
_MACHINE_PREFIX_RE = re.compile(r"^(/Users/|/home/|/var/folders/|/private/var/folders/)")

#: A home-relative path: ``~/...`` or ``~user/...``. A bare ``~``, ``~~`` or a tokenizer merge such as ``~ ~`` is
#: text: read as a home path it was rewritten to ``<redacted>/~`` (22 November: seven merges of a pack's
#: tokenizer.json, which then no longer encoded ``~``).
_HOME_RELATIVE_RE = re.compile(r"^~[A-Za-z0-9._-]*/")

#: Pack documents that hold the model's vocabulary, never provenance. The tokenizer library writes them, and a
#: token may look like a path (``~/``, ``/Users/``), so they are never scanned or rewritten.
VOCABULARY_DOCUMENTS = frozenset({"tokenizer.json", "vocab.json", "added_tokens.json"})
_MACHINE_PATH_IN_TEXT_RE = re.compile(
    r"(?:/Users/|/home/|/private/var/folders/|/var/folders/)[^\s\"';,)]*"
)


def _is_absolute_path(value: str) -> bool:
    return value.startswith("/") or value == "~" or bool(_HOME_RELATIVE_RE.match(value))


def _is_machine_path(value: str) -> bool:
    """A path that identifies the machine it came from."""

    if not value:
        return False
    return bool(_HOME_RELATIVE_RE.match(value)) or bool(_MACHINE_PREFIX_RE.match(value))


def _is_local_path(key: str | None, value: str) -> bool:
    """Whether ``value`` is a local path in the position ``key`` gives it.

    A home or temp path is local wherever it appears. Any other absolute
    path counts only under a key that names a path (``source_path``,
    ``output_dir``): an API route such as ``/v1/chat/completions`` or a
    tokenizer string that happens to start with a slash is data, not a
    location, and stays as it is.
    """

    if _is_machine_path(value):
        return True
    return key is not None and bool(_PATH_KEY_RE.search(key)) and _is_absolute_path(value)


def scrub_path_value(value: str) -> str:
    """Reduce an absolute local path to a non-identifying stand-in.

    A path keeps its final component (``experts.bin``,
    ``hy3-q4-mlx-mtp``) because that names the artifact, not the machine.
    Everything above it is dropped.
    """

    if not value or not _is_absolute_path(value):
        return value
    name = Path(value.rstrip("/")).name
    return f"{REDACTED_PATH}/{name}" if name else REDACTED_PATH


def scrub_text_value(value: str) -> str:
    """Redact machine paths embedded inside a free-text string."""

    return _MACHINE_PATH_IN_TEXT_RE.sub(
        lambda match: scrub_path_value(match.group(0)), value
    )


def _scrub_value(key: str | None, value: Any) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _scrub_value(child_key, child_value)
            for child_key, child_value in value.items()
            if child_key not in DROPPED_PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [_scrub_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(key, item) for item in value)
    if isinstance(value, str):
        if _is_local_path(key, value):
            return scrub_path_value(value)
        return scrub_text_value(value)
    return value


def scrub_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a publish-safe copy of a runtime-metadata dict.

    - Home and temp paths (``/Users/...``, ``/home/...``, ``/var/folders``)
      are cut down to ``<redacted>/<basename>`` wherever they appear: as a
      value, a list element, or embedded in a longer string. Other absolute
      paths are cut down only under path-named keys.
    - Machine-identifying provenance keys (``intended_hf_repo``) are removed.
    - Everything else, including ``source_repo``, ``source_sha``,
      ``forge_recipe`` and version stamps, is preserved verbatim.

    The input dict is never mutated.
    """

    if not isinstance(metadata, dict):
        raise TypeError("runtime metadata must be a dict")
    return _scrub_value(None, metadata)


def runtime_metadata_leaks(metadata: Any) -> list[str]:
    """Return every local path still present in ``metadata``.

    Intended as a publish-time assertion: an empty list means the payload
    carries no home-directory, temp-directory or path-keyed absolute path.
    """

    leaks: list[str] = []

    def walk(key: str | None, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child_key, child)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(key, item)
        elif isinstance(value, str):
            if _is_local_path(key, value):
                leaks.append(value)
            else:
                leaks.extend(_MACHINE_PATH_IN_TEXT_RE.findall(value))

    walk(None, metadata)
    return leaks


@dataclass(frozen=True)
class ScrubbedDocument:
    """A top-level JSON document of a pack that needed scrubbing."""

    name: str
    document: dict[str, Any]
    leaks: tuple[str, ...]

    @property
    def payload(self) -> bytes:
        return (json.dumps(self.document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def scrub_json_documents(directory: Path | str) -> list[ScrubbedDocument]:
    """Scrubbed copies of the top-level JSON documents that carry local paths.

    A pack directory holds its runtime contract, its config, its weight
    index and sometimes a conversion manifest. Every one that leaks comes
    back scrubbed, with what it leaked; a clean directory yields an empty
    list. Documents that do not parse are skipped, they are another gate's
    concern, and so are the vocabulary documents (``VOCABULARY_DOCUMENTS``).
    """

    documents: list[ScrubbedDocument] = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name in VOCABULARY_DOCUMENTS:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        leaks = runtime_metadata_leaks(payload)
        if leaks:
            documents.append(
                ScrubbedDocument(
                    name=path.name,
                    document=scrub_runtime_metadata(payload),
                    leaks=tuple(leaks),
                )
            )
    return documents
