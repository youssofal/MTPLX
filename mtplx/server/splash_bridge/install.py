# SPDX-License-Identifier: Apache-2.0
"""Locating a Homebrew Splash install and the model packages it has on disk.

Splash ships as a Homebrew formula: a compiled Metal engine plus a bundled
CPython that runs its own HTTP server.  Nothing here reimplements any of that.
This module only answers three questions the bridge needs before it can start
an engine: where Splash lives, which packages it publishes, and whether a given
package is already downloaded and verified.

Model downloads are delegated to Splash's own ``install/models.py prepare``,
which resolves the Hugging Face snapshot, verifies every artifact against the
package manifest, and retains the cache reference.  Reimplementing that
verification would be the one part of Splash we must never get subtly wrong.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# Splash keeps per-user state outside the Cellar so `brew upgrade` preserves
# downloaded packages; see its install/paths.py.
SPLASH_DATA = Path.home() / "Library/Application Support/Splash"
SPLASH_MODELS = SPLASH_DATA / "models"

# Used only when the install predates a catalog file. The live list comes from
# Splash's own catalog so a model added upstream appears without a code change.
FALLBACK_MODEL_IDS = (
    "incoai/Qwen3.8-27B-Splash",
    "incoai/Qwen3.6-35B-A3B-Splash",
)

# Download sizes for the published packages, for progress display only.
KNOWN_DOWNLOAD_BYTES = {
    "incoai/Qwen3.8-27B-Splash": 17_400_000_000,
    "incoai/Qwen3.6-35B-A3B-Splash": 20_900_000_000,
}


class SplashUnavailable(RuntimeError):
    """Splash is not installed, or the install is not one we recognize."""


def _brew_prefix() -> Path | None:
    try:
        done = subprocess.run(
            ["brew", "--prefix", "splash"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode:
        return None
    prefix = Path(done.stdout.strip())
    return prefix if prefix.is_dir() else None


def _prefix_from_shim() -> Path | None:
    """Read the libexec path out of the `splash` shell shim.

    The shim is two lines and ends in `exec .../libexec/python/bin/python3
    .../libexec/install/launcher.py "$@"`, so it names the install even when
    Homebrew is not on PATH (a packaged app, a login shell without brew).
    """
    shim = shutil.which("splash")
    if not shim:
        return None
    try:
        text = Path(shim).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for token in text.replace('"', " ").split():
        marker = "/libexec/install/launcher.py"
        if token.endswith(marker):
            return Path(token[: -len(marker)])
    return None


@dataclass(frozen=True)
class SplashInstall:
    """An on-disk Splash installation."""

    prefix: Path

    @property
    def libexec(self) -> Path:
        return self.prefix / "libexec"

    @property
    def python(self) -> Path:
        return self.libexec / "python/bin/python3"

    @property
    def engine_binary(self) -> Path:
        return self.libexec / "engine/splash"

    @property
    def server_script(self) -> Path:
        return self.libexec / "server/server.py"

    @property
    def models_script(self) -> Path:
        return self.libexec / "install/models.py"

    @property
    def catalog_files(self) -> tuple[Path, ...]:
        return (
            self.libexec / "install/completions/official-models.txt",
            SPLASH_DATA / "catalog/official-models.txt",
        )

    @classmethod
    def discover(cls, prefix: str | os.PathLike[str] | None = None) -> "SplashInstall":
        """Find Splash, preferring an explicit path, then brew, then the shim."""
        candidates: Iterable[Path | None] = (
            Path(prefix) if prefix else None,
            Path(os.environ["SPLASH_PREFIX"]) if os.environ.get("SPLASH_PREFIX") else None,
            _brew_prefix(),
            _prefix_from_shim(),
            Path("/opt/homebrew/opt/splash"),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            install = cls(Path(candidate))
            if install.server_script.is_file():
                return install
        raise SplashUnavailable(
            "Splash is not installed. Install it with:\n"
            "    brew install incoai/tap/splash"
        )

    def validate(self) -> None:
        """Fail early, and name the missing file, rather than at spawn time."""
        for label, path in (
            ("bundled Python", self.python),
            ("engine binary", self.engine_binary),
            ("server", self.server_script),
            ("model installer", self.models_script),
        ):
            if not path.is_file():
                raise SplashUnavailable(
                    f"Splash {label} is missing at {path}. "
                    "Try: brew reinstall incoai/tap/splash"
                )

    def version(self) -> str:
        try:
            done = subprocess.run(
                ["splash", "--version"], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return done.stdout.strip() or "unknown"

    # -- model packages ---------------------------------------------------

    def official_models(self) -> list[str]:
        """The union of the bundled catalog and Splash's refreshed cache.

        Splash takes the same union: the cache can only ever add entries, so a
        stale or missing cache never hides a package the install already knows.
        """
        found: set[str] = set()
        for path in self.catalog_files:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line in text.splitlines():
                line = line.strip()
                if line.count("/") == 1 and line.isprintable():
                    found.add(line)
        return sorted(found or FALLBACK_MODEL_IDS)

    def package_root(self, model_id: str) -> Path:
        return SPLASH_MODELS / model_id

    def is_installed(self, model_id: str) -> bool:
        """True once Splash has verified the package against its manifest."""
        return (self.package_root(model_id) / "manifest.json").is_file()

    def reasoning_efforts(self, model_id: str) -> tuple[tuple[str, ...], str | None]:
        """The thinking-effort levels the package's chat template accepts.

        Splash hands ``reasoning_effort`` to the template, so the levels are
        whatever the template checks for, and its default is what it falls
        back to when a request names none (Qwen 3.8: xhigh, medium, low;
        xhigh). ``((), None)`` for a template without effort levels.
        """
        try:
            template = (
                self.package_root(model_id) / "tokenizer" / "chat_template.jinja"
            ).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return (), None
        allowed = re.search(r"reasoning_effort\s+not\s+in\s*\(([^)]*)\)", template)
        if allowed is None:
            return (), None
        levels = tuple(re.findall(r"['\"](\w+)['\"]", allowed.group(1)))
        default = re.search(r"reasoning_effort\s*\|\s*default\(\s*['\"](\w+)['\"]", template)
        fallback = default.group(1) if default and default.group(1) in levels else None
        return levels, fallback

    def draft_tokens(self, model_id: str) -> int | None:
        """How many tokens the package's DFlash 2 draft proposes per verify.

        Compiled into the package (``execution_geometry`` in its manifest), so
        it is read rather than configured. None when the package is not
        installed or does not say.
        """
        try:
            manifest = json.loads(
                (self.package_root(model_id) / "manifest.json").read_text()
            )
            value = manifest["execution_geometry"]["draft_proposal_tokens"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return None

    def prepare(
        self,
        model_id: str,
        *,
        on_line: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Download and verify a package using Splash's own installer.

        Blocks until the package is verified. Splash takes an exclusive
        installation lock, so a concurrent `splash serve` download makes this
        wait rather than corrupt the cache.
        """
        SPLASH_MODELS.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            str(self.models_script),
            "--models",
            str(SPLASH_MODELS),
            "--model",
            model_id,
            "prepare",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {})},
        )
        assert process.stdout is not None
        for line in process.stdout:
            if on_line is not None:
                on_line(line.rstrip())
        if process.wait():
            raise SplashUnavailable(
                f"Splash could not download or verify {model_id}; see the output above"
            )
