# SPDX-License-Identifier: Apache-2.0
"""`mtplx splash`: manage the Splash engine's model packages.

The native app and any script need one stable way to ask what Splash has on
disk and to fetch what it does not, without reaching into Splash's private
libexec layout. That is all this command is. Downloading and verification are
still done by Splash's own installer, which checks every artifact against the
package manifest; this only drives it and reports progress.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mtplx.server.splash_bridge.install import (
    KNOWN_DOWNLOAD_BYTES,
    SplashInstall,
    SplashUnavailable,
)


def _packages(install: SplashInstall) -> list[dict[str, Any]]:
    rows = []
    for model_id in install.official_models():
        root = install.package_root(model_id)
        rows.append(
            {
                "id": model_id,
                "installed": install.is_installed(model_id),
                "path": str(root),
                "download_bytes": KNOWN_DOWNLOAD_BYTES.get(model_id),
            }
        )
    return rows


def cmd_splash(args: Any) -> int:
    action = getattr(args, "splash_action", None) or "list"
    as_json = bool(getattr(args, "json", False))

    try:
        install = SplashInstall.discover(getattr(args, "splash_prefix", None))
        install.validate()
    except SplashUnavailable as error:
        payload = {"ok": False, "error": str(error)}
        if as_json:
            print(json.dumps(payload))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2

    if action == "list":
        payload = {
            "ok": True,
            "engine": "splash",
            "version": install.version(),
            "packages": _packages(install),
        }
        if as_json:
            print(json.dumps(payload))
        else:
            print(payload['version'])
            for row in payload["packages"]:
                size = row["download_bytes"]
                size_text = f"{size / 1e9:.1f} GB" if size else "unknown size"
                mark = "installed" if row["installed"] else f"not downloaded · {size_text}"
                print(f"  {row['id']:34} {mark}")
        return 0

    model_id = getattr(args, "model", None)
    if not model_id:
        print("error: --model is required", file=sys.stderr)
        return 2
    known = install.official_models()
    if model_id not in known and not install.is_installed(model_id):
        print(f"error: {model_id} is not a Splash package", file=sys.stderr)
        for candidate in known:
            print(f"  {candidate}", file=sys.stderr)
        return 2

    if action == "install":
        if install.is_installed(model_id) and not getattr(args, "force", False):
            if as_json:
                print(json.dumps({"ok": True, "id": model_id, "installed": True,
                                  "changed": False}))
            else:
                print(f"{model_id} is already installed and verified")
            return 0
        try:
            # Progress goes to stdout line by line so a caller can stream it;
            # Splash's installer prints the Hugging Face transfer bars here.
            install.prepare(model_id, on_line=lambda line: print(line, flush=True))
        except SplashUnavailable as error:
            if as_json:
                print(json.dumps({"ok": False, "id": model_id, "error": str(error)}))
            else:
                print(f"error: {error}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps({"ok": True, "id": model_id, "installed": True,
                              "changed": True}))
        else:
            print(f"Installed {model_id}")
        return 0

    if action == "verify":
        # `prepare` re-verifies an installed package against its manifest and
        # re-fetches anything that does not match, so it is also the update
        # path: a changed package upstream is repaired in place.
        try:
            install.prepare(model_id, on_line=lambda line: print(line, flush=True))
        except SplashUnavailable as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"{model_id} verified")
        return 0

    print(f"error: unknown action {action!r}", file=sys.stderr)
    return 2
