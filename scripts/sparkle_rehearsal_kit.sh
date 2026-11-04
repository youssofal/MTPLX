#!/usr/bin/env bash
set -euo pipefail

# Sparkle update rehearsal kit.
#
# The auto-update chain (EdDSA verify -> install -> relaunch -> runtime
# refresh) has never been exercised against a live feed. This script
# produces everything needed to rehearse it end to end with zero user
# exposure:
#
#   MTPLX-1.0.0.dmg          install this on the test device first
#   MTPLX-1.0.1.dmg          the update Sparkle should deliver
#   site/releases/appcast.xml  signed feed listing 1.0.1
#   site/releases/notes/...    release-notes pages
#   README.md                hosting + click-through instructions
#
# The 1.0.1 app embeds a 1.0.1 runtime wheel (built from a version-patched
# source export) so the post-update venv refresh — the riskiest leg — is
# rehearsed exactly as a real upgrade would run it.
#
# Requirements match scripts/release_macos_v1.sh: a Developer ID identity,
# the Sparkle signing key, and (recommended) notarization credentials so
# the quarantined downloads open on the second device.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$ROOT/apps/MTPLXApp"
BUILD_SCRIPT="$APP_ROOT/script/build_and_run.sh"
BASE_VERSION="1.0.0"
TEST_VERSION="1.0.1"
# First release revision of each version, from the same derivation the real
# release uses (build_and_run.sh refuses any other number for a public build).
BASE_BUILD="$("$APP_ROOT/script/app_build_number.sh" "$BASE_VERSION" 1)"
TEST_BUILD="$("$APP_ROOT/script/app_build_number.sh" "$TEST_VERSION" 1)"
OUT_ROOT="${MTPLX_REHEARSAL_OUT:-$HOME/.mtplx/releases/sparkle-rehearsal-$(/bin/date -u +%Y%m%dT%H%M%SZ)}"
ASSET_BASE="${MTPLX_REHEARSAL_ASSET_BASE:-https://mtplx.com/releases/test}"
SITE_OUT="$OUT_ROOT/site"
RELEASES_OUT="$SITE_OUT/releases"
NOTES_OUT="$RELEASES_OUT/notes"
SPARKLE_ARCHIVES="$OUT_ROOT/sparkle-archives"
PYTHON_DIST="$OUT_ROOT/python"
PYTOOLS_VENV="$OUT_ROOT/python-tools-venv"

export MTPLX_SPARKLE_PUBLIC_ED_KEY="${MTPLX_SPARKLE_PUBLIC_ED_KEY:-GQ0sTm6nb5kv+Btri7wc4LqnXGZ48vIs6PGMwsI/mBM=}"
SPARKLE_PRIVATE_KEY="${MTPLX_SPARKLE_PRIVATE_KEY:-${SPARKLE_PRIVATE_KEY:-}}"
SPARKLE_PRIVATE_KEY_FILE="${MTPLX_SPARKLE_PRIVATE_KEY_FILE:-${SPARKLE_PRIVATE_KEY_FILE:-}}"
SPARKLE_KEY_ACCOUNT="${MTPLX_SPARKLE_KEY_ACCOUNT:-ed25519}"

CODESIGN_IDENTITY="${MTPLX_CODESIGN_IDENTITY:-${MTPLX_DEVELOPER_ID_APPLICATION:-}}"
if [[ -z "$CODESIGN_IDENTITY" ]]; then
  echo "error: set MTPLX_CODESIGN_IDENTITY or MTPLX_DEVELOPER_ID_APPLICATION" >&2
  exit 1
fi
if [[ -z "$SPARKLE_PRIVATE_KEY" && -z "$SPARKLE_PRIVATE_KEY_FILE" && "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" != "1" ]]; then
  echo "error: Sparkle signing key missing; set MTPLX_SPARKLE_PRIVATE_KEY or MTPLX_SPARKLE_PRIVATE_KEY_FILE (or MTPLX_SPARKLE_ALLOW_KEYCHAIN=1 for interactive runs)" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$PYTHON_DIST" "$RELEASES_OUT" "$NOTES_OUT" "$SPARKLE_ARCHIVES"

submit_notarization() {
  local artifact="$1"
  if [[ -n "${MTPLX_NOTARY_PROFILE:-}" ]]; then
    xcrun notarytool submit "$artifact" --keychain-profile "$MTPLX_NOTARY_PROFILE" --wait
  elif [[ -n "${MTPLX_ASC_KEY:-}" && -n "${MTPLX_ASC_KEY_ID:-}" && -n "${MTPLX_ASC_ISSUER_ID:-}" ]]; then
    xcrun notarytool submit "$artifact" \
      --key "$MTPLX_ASC_KEY" \
      --key-id "$MTPLX_ASC_KEY_ID" \
      --issuer "$MTPLX_ASC_ISSUER_ID" \
      --wait
  else
    echo "warning: no notarization credentials; the downloaded DMG will be blocked by Gatekeeper on the test device" >&2
    return 1
  fi
}

echo "Preparing Python build tooling"
python3 -m venv "$PYTOOLS_VENV"
"$PYTOOLS_VENV/bin/python" -m pip install --quiet --upgrade pip build markdown

build_wheel() {
  # build_wheel <version> -> echoes the wheel path. Versions other than the
  # checked-in one build from a version-patched `git archive` export so the
  # updated app carries a runtime wheel matching its own version floor.
  local version="$1"
  local source_dir="$ROOT"
  if [[ "$version" != "$BASE_VERSION" ]]; then
    source_dir="$OUT_ROOT/source-$version"
    rm -rf "$source_dir"
    mkdir -p "$source_dir"
    (cd "$ROOT" && /usr/bin/git archive HEAD) | /usr/bin/tar -x -C "$source_dir"
    "$PYTOOLS_VENV/bin/python" - "$source_dir/pyproject.toml" "$version" <<'PY'
import pathlib
import re
import sys

path, version = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8")
patched, count = re.subn(
    r'^version = "[^"]+"', f'version = "{version}"', text, count=1, flags=re.M
)
if count != 1:
    raise SystemExit("could not patch pyproject version")
path.write_text(patched, encoding="utf-8")
PY
  fi
  "$PYTOOLS_VENV/bin/python" -m build "$source_dir" --outdir "$PYTHON_DIST" --wheel >&2
  local wheel="$PYTHON_DIST/mtplx-$version-py3-none-any.whl"
  if [[ ! -f "$wheel" ]]; then
    echo "error: expected wheel missing: $wheel" >&2
    exit 1
  fi
  echo "$wheel"
}

# Bundled Python runtime, same pin as the release script.
PBS_RELEASE="${MTPLX_PBS_RELEASE:-20260602}"
PBS_PYTHON_VERSION="${MTPLX_PBS_PYTHON_VERSION:-3.14.5}"
PBS_ARTIFACT="cpython-$PBS_PYTHON_VERSION+$PBS_RELEASE-aarch64-apple-darwin-install_only_stripped.tar.gz"
PBS_SHA256="${MTPLX_PBS_SHA256:-3a0373cc39fefd494754ef555267f245c720cddbaaabf63a7c9a4269f1e56532}"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_RELEASE/$PBS_ARTIFACT"
PBS_CACHE_DIR="${MTPLX_PBS_CACHE_DIR:-$HOME/.mtplx/build-cache/python-build-standalone}"
PBS_TARBALL="$PBS_CACHE_DIR/$PBS_ARTIFACT"
PBS_EXTRACT_DIR="$OUT_ROOT/python-runtime"

echo "Preparing bundled Python runtime ($PBS_ARTIFACT)"
mkdir -p "$PBS_CACHE_DIR"
if [[ ! -f "$PBS_TARBALL" ]]; then
  /usr/bin/curl -fL --retry 3 -o "$PBS_TARBALL.partial" "$PBS_URL"
  mv "$PBS_TARBALL.partial" "$PBS_TARBALL"
fi
echo "$PBS_SHA256  $PBS_TARBALL" | /usr/bin/shasum -a 256 -c - >/dev/null
rm -rf "$PBS_EXTRACT_DIR"
mkdir -p "$PBS_EXTRACT_DIR"
/usr/bin/tar -xzf "$PBS_TARBALL" -C "$PBS_EXTRACT_DIR" --strip-components 1

build_dmg() {
  # build_dmg <version> <build-number> <wheel-path>
  local version="$1"
  local build="$2"
  local wheel="$3"
  local bundle="$OUT_ROOT/MTPLX-$version.app"
  local dmg="$OUT_ROOT/MTPLX-$version.dmg"
  local stage="$OUT_ROOT/dmg-stage-$version"

  echo "Building signed MTPLX.app $version ($build)"
  MTPLX_APP_PUBLIC_RELEASE=1 \
  MTPLX_APP_VERSION="$version" \
  MTPLX_APP_BUILD="$build" \
  MTPLX_APP_BUNDLE_DIR="$bundle" \
  MTPLX_APP_EMBED_LOCAL_RUNTIME_WRAPPER=0 \
  MTPLX_RUNTIME_WHEEL="$wheel" \
  MTPLX_REQUIRE_RUNTIME_WHEEL_RESOURCE=1 \
  MTPLX_BUNDLED_PYTHON_DIR="$PBS_EXTRACT_DIR" \
  MTPLX_REQUIRE_BUNDLED_PYTHON_RESOURCE=1 \
  MTPLX_CODESIGN_IDENTITY="$CODESIGN_IDENTITY" \
  "$BUILD_SCRIPT" --no-launch

  /usr/bin/codesign --verify --deep --strict "$bundle"

  echo "Creating MTPLX-$version.dmg"
  /bin/rm -rf "$stage"
  /bin/mkdir -p "$stage"
  /usr/bin/ditto --norsrc "$bundle" "$stage/MTPLX.app"
  /bin/ln -s /Applications "$stage/Applications"
  /usr/bin/xattr -rc "$stage" >/dev/null 2>&1 || true
  /usr/bin/hdiutil create -volname "MTPLX $version" -srcfolder "$stage" -ov -format UDZO "$dmg"
  /usr/bin/codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$dmg"
  if [[ "${MTPLX_SKIP_NOTARIZATION:-0}" != "1" ]]; then
    if submit_notarization "$dmg"; then
      xcrun stapler staple "$dmg"
    fi
  fi
  /usr/bin/ditto --norsrc "$dmg" "$SPARKLE_ARCHIVES/$(basename "$dmg")"
}

render_notes() {
  # render_notes <version> <markdown-file>
  # Renders through the same template as the real release pipeline
  # (scripts/render_release_notes.py), so the rehearsal shows the exact
  # page users get — dark-mode readability included (#367). This function
  # once carried a drifted inline copy with no stylesheet at all.
  local version="$1"
  local source_md="$2"
  "$PYTOOLS_VENV/bin/python" "$ROOT/scripts/render_release_notes.py" \
    "$source_md" "$NOTES_OUT/v$version.html" "$version"
  /usr/bin/ditto --norsrc "$NOTES_OUT/v$version.html" "$SPARKLE_ARCHIVES/MTPLX-$version.html"
}

BASE_WHEEL="$(build_wheel "$BASE_VERSION")"
TEST_WHEEL="$(build_wheel "$TEST_VERSION")"
build_dmg "$BASE_VERSION" "$BASE_BUILD" "$BASE_WHEEL"
build_dmg "$TEST_VERSION" "$TEST_BUILD" "$TEST_WHEEL"

render_notes "$BASE_VERSION" "$ROOT/docs/releases/v$BASE_VERSION.md"
TEST_NOTES_MD="$OUT_ROOT/v$TEST_VERSION-notes.md"
cat > "$TEST_NOTES_MD" <<MD
# MTPLX $TEST_VERSION (update rehearsal)

This is a test update used to rehearse MTPLX's automatic updates before
launch. It is identical to $BASE_VERSION apart from the version number.

If you can read this from the in-app update dialog, the release-notes
chain works.
MD
render_notes "$TEST_VERSION" "$TEST_NOTES_MD"

echo "Generating signed appcast"
GENERATE_APPCAST="$(
  /usr/bin/find "$APP_ROOT/.build" -path '*/bin/generate_appcast' -type f -print 2>/dev/null | /usr/bin/head -n 1
)"
if [[ -z "$GENERATE_APPCAST" ]]; then
  echo "error: Sparkle generate_appcast not found under $APP_ROOT/.build (build the app first)" >&2
  exit 1
fi
APPCAST_ARGS=()
if "$GENERATE_APPCAST" --help 2>&1 | /usr/bin/grep -q -- '--download-url-prefix'; then
  APPCAST_ARGS+=(--download-url-prefix "$ASSET_BASE/")
fi
if "$GENERATE_APPCAST" --help 2>&1 | /usr/bin/grep -q -- '--full-release-notes-url'; then
  APPCAST_ARGS+=(--full-release-notes-url "$ASSET_BASE/notes/v$TEST_VERSION.html")
fi
if [[ -n "$SPARKLE_PRIVATE_KEY" ]]; then
  printf '%s' "$SPARKLE_PRIVATE_KEY" | "$GENERATE_APPCAST" --ed-key-file - "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
elif [[ -n "$SPARKLE_PRIVATE_KEY_FILE" ]]; then
  "$GENERATE_APPCAST" --ed-key-file "$SPARKLE_PRIVATE_KEY_FILE" "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
else
  "$GENERATE_APPCAST" --account "$SPARKLE_KEY_ACCOUNT" "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
fi
APPCAST_SOURCE="$(
  /usr/bin/find "$SPARKLE_ARCHIVES" -maxdepth 1 -name '*.xml' -type f -print | /usr/bin/head -n 1
)"
if [[ -z "$APPCAST_SOURCE" ]]; then
  echo "error: generate_appcast did not produce an appcast" >&2
  exit 1
fi
/usr/bin/ditto --norsrc "$APPCAST_SOURCE" "$RELEASES_OUT/appcast.xml"
/usr/bin/ditto --norsrc "$OUT_ROOT/MTPLX-$BASE_VERSION.dmg" "$RELEASES_OUT/MTPLX-$BASE_VERSION.dmg"
/usr/bin/ditto --norsrc "$OUT_ROOT/MTPLX-$TEST_VERSION.dmg" "$RELEASES_OUT/MTPLX-$TEST_VERSION.dmg"

cat > "$OUT_ROOT/README.md" <<MD
# Sparkle update rehearsal

Generated $(/bin/date -u +%Y-%m-%dT%H:%M:%SZ). Everything under site/ is
ready to host as-is.

## Steps

1. Upload the contents of site/releases/ so they are reachable at:
   $ASSET_BASE/appcast.xml
   $ASSET_BASE/MTPLX-$TEST_VERSION.dmg
   $ASSET_BASE/notes/v$TEST_VERSION.html
2. On the second device, install MTPLX-$BASE_VERSION.dmg (drag to
   /Applications) and finish onboarding.
3. Point the installed app at the rehearsal feed before launching it:
   defaults write com.youssofal.mtplx SUFeedURL "$ASSET_BASE/appcast.xml"
4. Open MTPLX and use "Check for Updates".

## What success looks like

- The update dialog shows "MTPLX $TEST_VERSION" with the rehearsal notes.
- Install + relaunch lands on $TEST_VERSION (About window).
- After relaunch the app refreshes its runtime: mtplx --version in
  Terminal reports $TEST_VERSION, proving the post-update venv refresh.

If EdDSA verification fails, the public key baked into the app and the
key used to sign this appcast do not match.
MD

echo
echo "Rehearsal kit ready: $OUT_ROOT"
echo "  $OUT_ROOT/MTPLX-$BASE_VERSION.dmg   (install on the test device)"
echo "  $RELEASES_OUT                        (host at $ASSET_BASE)"
echo "  $OUT_ROOT/README.md                  (click-through instructions)"
