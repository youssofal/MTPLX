#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$ROOT/apps/MTPLXApp"
BUILD_SCRIPT="$APP_ROOT/script/build_and_run.sh"
VERSION="${MTPLX_RELEASE_VERSION:-$(/usr/bin/awk -F'"' '/^version = / { print $2; exit }' "$ROOT/pyproject.toml")}"
# Left empty, the build script derives it from VERSION; the appcast then reads
# the number back off the built bundle so the feed cannot rank a release
# differently from the app it ships.
APP_BUILD="${MTPLX_RELEASE_BUILD:-}"
RELEASE_TAG="${MTPLX_RELEASE_TAG:-v$VERSION}"
GITHUB_REPO="${MTPLX_GITHUB_REPO:-youssofal/mtplx}"
GITHUB_ASSET_BASE="${MTPLX_GITHUB_ASSET_BASE:-https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG}"
OUT_ROOT="${MTPLX_RELEASE_OUT:-$HOME/.mtplx/releases/$VERSION-$(/bin/date -u +%Y%m%dT%H%M%SZ)}"
APP_BUNDLE="$OUT_ROOT/MTPLX.app"
DMG="$OUT_ROOT/MTPLX-$VERSION.dmg"
DMG_STAGE="$OUT_ROOT/dmg-stage"
PYTHON_DIST="$OUT_ROOT/python"
PYTOOLS_VENV="$OUT_ROOT/python-tools-venv"
SITE_OUT="$OUT_ROOT/site"
RELEASES_OUT="$SITE_OUT/releases"
NOTES_OUT="$RELEASES_OUT/notes"
SPARKLE_ARCHIVES="$OUT_ROOT/sparkle-archives"
APP_NOTARY_ZIP="$OUT_ROOT/MTPLX-$VERSION.app.zip"

# Guard against accidentally shipping a pre-1.0 or malformed version.
if [[ ! "$VERSION" =~ ^[1-9][0-9]*\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: release version must be a stable x.y.z (>= 1.0.0), got $VERSION" >&2
  exit 1
fi

# Sparkle ranks releases purely by CFBundleVersion, and the semantic
# derivation in build_and_run.sh (major*1000000 + minor*1000 + patch) has
# drifted below shipped reality: 2.9.1 shipped as 2009009 and 2.9.2 as
# 2009010, so a derived 2.9.3 (2009003) would rank BELOW the shipped 2.9.2
# and never be offered to updaters — while 2.9.2 could be offered "back" to
# 2.9.3 users. Refuse, before any expensive gate runs, any build number
# that does not strictly advance the local shipping record.
CANDIDATE_BUILD="$APP_BUILD"
if [[ -z "$CANDIDATE_BUILD" ]]; then
  IFS='.' read -r _bn_major _bn_minor _bn_patch <<<"$VERSION"
  CANDIDATE_BUILD="$((10#$_bn_major * 1000000 + 10#$_bn_minor * 1000 + 10#$_bn_patch))"
fi
LAST_SHIPPED_BUILD=0
for _bn_manifest in "$HOME"/.mtplx/releases/*/site/releases/latest.json; do
  [[ -f "$_bn_manifest" ]] || continue
  _bn_val="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("app_build",""))' "$_bn_manifest" 2>/dev/null || true)"
  [[ "$_bn_val" =~ ^[0-9]+$ ]] || continue
  if (( _bn_val > LAST_SHIPPED_BUILD )); then LAST_SHIPPED_BUILD="$_bn_val"; fi
done
if (( LAST_SHIPPED_BUILD > 0 && CANDIDATE_BUILD <= LAST_SHIPPED_BUILD )); then
  echo "error: build number $CANDIDATE_BUILD does not advance the last shipped CFBundleVersion $LAST_SHIPPED_BUILD" >&2
  echo "       Sparkle ranks by CFBundleVersion alone, so this release would never reach existing users." >&2
  echo "       Export MTPLX_RELEASE_BUILD=$((LAST_SHIPPED_BUILD + 1)) and rerun." >&2
  exit 1
fi

export MTPLX_SPARKLE_PUBLIC_ED_KEY="${MTPLX_SPARKLE_PUBLIC_ED_KEY:-GQ0sTm6nb5kv+Btri7wc4LqnXGZ48vIs6PGMwsI/mBM=}"
SPARKLE_PRIVATE_KEY="${MTPLX_SPARKLE_PRIVATE_KEY:-${SPARKLE_PRIVATE_KEY:-}}"
SPARKLE_PRIVATE_KEY_FILE="${MTPLX_SPARKLE_PRIVATE_KEY_FILE:-${SPARKLE_PRIVATE_KEY_FILE:-}}"
SPARKLE_KEY_ACCOUNT="${MTPLX_SPARKLE_KEY_ACCOUNT:-ed25519}"

CODESIGN_IDENTITY="${MTPLX_CODESIGN_IDENTITY:-${MTPLX_DEVELOPER_ID_APPLICATION:-}}"
if [[ -z "$CODESIGN_IDENTITY" ]]; then
  echo "error: set MTPLX_CODESIGN_IDENTITY or MTPLX_DEVELOPER_ID_APPLICATION to the Developer ID Application identity" >&2
  exit 1
fi

if [[ -z "$SPARKLE_PRIVATE_KEY" && -z "$SPARKLE_PRIVATE_KEY_FILE" && "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" != "1" ]]; then
  echo "error: Sparkle appcast signing key missing; set MTPLX_SPARKLE_PRIVATE_KEY or MTPLX_SPARKLE_PRIVATE_KEY_FILE" >&2
  echo "note: set MTPLX_SPARKLE_ALLOW_KEYCHAIN=1 only for interactive local signing, because Keychain access can block unattended releases" >&2
  exit 1
fi

if [[ "${MTPLX_RELEASE_ALLOW_DIRTY:-0}" != "1" ]]; then
  if ! (cd "$ROOT" && git diff --quiet && git diff --cached --quiet); then
    echo "error: release checkout has uncommitted tracked changes" >&2
    exit 1
  fi
  if [[ -n "$(cd "$ROOT" && git ls-files --others --exclude-standard)" ]]; then
    echo "error: release checkout has untracked files" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_ROOT" "$PYTHON_DIST" "$SITE_OUT" "$RELEASES_OUT" "$NOTES_OUT" "$SPARKLE_ARCHIVES"

# Release gates: the full Python and Swift suites must be green before any
# artifact is built or signed. MTPLX_RELEASE_SKIP_TESTS=1 exists only for
# rehearsals of the packaging pipeline itself.
if [[ "${MTPLX_RELEASE_SKIP_TESTS:-0}" != "1" ]]; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "error: $ROOT/.venv is missing; create it and install '.[server,dev]' before releasing" >&2
    exit 1
  fi
  echo "Release gate: pytest"
  (cd "$ROOT" && "$ROOT/.venv/bin/python" -m pytest -q)
  echo "Release gate: swift test"
  (cd "$APP_ROOT" && swift test)
else
  echo "warning: MTPLX_RELEASE_SKIP_TESTS=1 — test gates skipped; this artifact is not release-ready" >&2
fi

# Pillar gate: live product checks against a serving daemon (vision-cache
# survival, memory ceiling, long-output decode decay). These are the
# founder-visible regressions unit tests cannot catch (2026-07-09: one
# screenshot disabled prompt caching for the rest of the session and no
# gate noticed). Requires a running daemon under verified max fans:
#   MTPLX_RELEASE_PILLAR_QA_URL=http://127.0.0.1:<port> \
#   MTPLX_RELEASE_PILLAR_QA_FAN_RPM=<measured rpm>
# Skipping prints the same not-release-ready warning as the test gates.
if [[ "${MTPLX_RELEASE_SKIP_PILLAR_QA:-0}" != "1" ]]; then
  if [[ -z "${MTPLX_RELEASE_PILLAR_QA_URL:-}" ]]; then
    echo "error: pillar gate needs MTPLX_RELEASE_PILLAR_QA_URL (a serving daemon under verified max fans)" >&2
    echo "       or MTPLX_RELEASE_SKIP_PILLAR_QA=1 to skip (artifact then not release-ready)" >&2
    exit 1
  fi
  echo "Release gate: pillar QA (vision cache / memory ceiling / decode decay)"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/pillar_gate_qa.py" \
    --base-url "$MTPLX_RELEASE_PILLAR_QA_URL" \
    --fan-rpm-verified "${MTPLX_RELEASE_PILLAR_QA_FAN_RPM:-0}"
  # Agent-session gate: the coding-agent turn loop every harness drives
  # (OpenCode, Pi, Hermes, Claude Code, Cline), judged from the engine's own
  # receipts -- warm-turn dead time, bank hits after tool calls, hidden
  # postcommit waits, O(1) generation-final snapshots, decode floor, stream
  # errors. 2026-09-03: three engine defects cost a 14-minute OpenCode task
  # 146 s and read as "decode 21 tok/s"; none was visible to a unit test or a
  # single-request benchmark, all fail this gate.
  echo "Release gate: agent-session QA (warm-turn dead time / bank hits / postcommit / decode floor)"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/agent_session_gate.py" \
    --base-url "$MTPLX_RELEASE_PILLAR_QA_URL" \
    --context-tokens "${MTPLX_RELEASE_AGENT_GATE_CONTEXT_TOKENS:-40000}" \
    --fan-rpm-verified "${MTPLX_RELEASE_PILLAR_QA_FAN_RPM:-0}"
else
  echo "warning: MTPLX_RELEASE_SKIP_PILLAR_QA=1 — pillar gate skipped; this artifact is not release-ready" >&2
fi

RELEASE_NOTES_MD="$ROOT/docs/releases/v$VERSION.md"
if [[ ! -f "$RELEASE_NOTES_MD" ]]; then
  echo "error: release notes source missing: $RELEASE_NOTES_MD" >&2
  echo "write the user-facing notes for v$VERSION before releasing" >&2
  exit 1
fi

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
    echo "error: notarization credentials missing; set MTPLX_NOTARY_PROFILE or App Store Connect key env" >&2
    exit 1
  fi
}

echo "Building Python artifacts for mtplx==$VERSION"
# A stale mtplx.egg-info/SOURCES.txt resurrects files MANIFEST.in excludes
# (it once leaked the internal LOG.md into the sdist); always build from a
# clean manifest.
rm -rf "$ROOT/mtplx.egg-info"
python3 -m venv "$PYTOOLS_VENV"
"$PYTOOLS_VENV/bin/python" -m pip install --upgrade pip build twine markdown
"$PYTOOLS_VENV/bin/python" -m build "$ROOT" --outdir "$PYTHON_DIST"
"$PYTOOLS_VENV/bin/python" -m twine check "$PYTHON_DIST"/*
PYTHON_WHEEL="$PYTHON_DIST/mtplx-$VERSION-py3-none-any.whl"
if [[ ! -f "$PYTHON_WHEEL" ]]; then
  echo "error: expected runtime wheel missing: $PYTHON_WHEEL" >&2
  exit 1
fi

# Bundled Python interpreter: python-build-standalone (Astral), pinned by
# release tag + sha256 so every build ships the exact same interpreter.
# This is what removes the "Install Homebrew" wall on pristine Macs.
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
if ! echo "$PBS_SHA256  $PBS_TARBALL" | /usr/bin/shasum -a 256 -c - >/dev/null 2>&1; then
  echo "error: bundled Python tarball failed its sha256 pin: $PBS_TARBALL" >&2
  echo "expected $PBS_SHA256" >&2
  exit 1
fi
rm -rf "$PBS_EXTRACT_DIR"
mkdir -p "$PBS_EXTRACT_DIR"
/usr/bin/tar -xzf "$PBS_TARBALL" -C "$PBS_EXTRACT_DIR" --strip-components 1
if [[ ! -x "$PBS_EXTRACT_DIR/bin/python3" ]]; then
  echo "error: bundled Python extraction failed; $PBS_EXTRACT_DIR/bin/python3 missing" >&2
  exit 1
fi

# Build the optional sparse-prefill consumer for the exact bundled Python.
# Keep the pure wheel beside it: pip/app tag selection declines this binary
# on older macOS or a different Python ABI rather than breaking installation.
NATIVE_BUILD_VENV="$OUT_ROOT/native-build-venv"
NATIVE_DIST="$OUT_ROOT/native-wheels"
"$PBS_EXTRACT_DIR/bin/python3" -m venv "$NATIVE_BUILD_VENV"
"$NATIVE_BUILD_VENV/bin/python" -m pip install \
  build wheel setuptools 'cmake>=3.27' 'mlx==0.32.2' 'nanobind==2.15.0'
MACOSX_DEPLOYMENT_TARGET=15.0 "$NATIVE_BUILD_VENV/bin/python" -m build \
  --wheel --no-isolation "$ROOT/native_extensions/qsa_kernels" --outdir "$NATIVE_DIST"
NATIVE_WHEELS=("$NATIVE_DIST"/mtplx_qsa_kernels-*.whl)
if [[ "${#NATIVE_WHEELS[@]}" != "1" || ! -f "${NATIVE_WHEELS[0]}" ]]; then
  echo "error: expected exactly one native QSA wheel" >&2
  exit 1
fi
NATIVE_RUNTIME_WHEEL="$("$NATIVE_BUILD_VENV/bin/python" \
  "$ROOT/scripts/bundle_native_runtime_wheel.py" "$PYTHON_WHEEL" \
  "${NATIVE_WHEELS[0]}" --out "$PYTHON_DIST" --codesign-identity "$CODESIGN_IDENTITY")"
"$PYTOOLS_VENV/bin/python" -m twine check "$NATIVE_RUNTIME_WHEEL"

# Notarization gate: every Mach-O inside the runtime wheel must carry the
# Developer ID and a secure timestamp. The app's signing pass cannot reach
# into the wheel, and the notary service rejected 2.11.2's first submission
# on exactly these two files; catch it here, not after a 20-minute upload.
NATIVE_CHECK_DIR="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/mtplx-native-wheel-check.XXXXXX")"
/usr/bin/unzip -q -o "$NATIVE_RUNTIME_WHEEL" -d "$NATIVE_CHECK_DIR"
NATIVE_SIGNED=0
while IFS= read -r member; do
  [[ -n "$member" ]] || continue
  /usr/bin/codesign --verify --strict "$member"
  details="$(/usr/bin/codesign -dvvv "$member" 2>&1)"
  if ! /usr/bin/grep -q 'Authority=Developer ID Application' <<<"$details" \
     || ! /usr/bin/grep -q '^Timestamp=' <<<"$details"; then
    echo "error: ${member#"$NATIVE_CHECK_DIR"/} is not Developer ID signed with a secure timestamp; notarization would reject it" >&2
    exit 1
  fi
  NATIVE_SIGNED=$((NATIVE_SIGNED + 1))
done < <(/usr/bin/find "$NATIVE_CHECK_DIR" \( -name '*.so' -o -name '*.dylib' \) -type f)
if [[ "$NATIVE_SIGNED" -lt 2 ]]; then
  echo "error: expected the QSA extension and its kernel library inside $NATIVE_RUNTIME_WHEEL, found $NATIVE_SIGNED signed Mach-O files" >&2
  exit 1
fi
echo "Native runtime wheel: $NATIVE_SIGNED Mach-O members Developer ID signed with secure timestamps"

echo "Building signed MTPLX.app"
MTPLX_APP_PUBLIC_RELEASE=1 \
MTPLX_APP_VERSION="$VERSION" \
MTPLX_APP_BUILD="$APP_BUILD" \
MTPLX_APP_BUNDLE_DIR="$APP_BUNDLE" \
MTPLX_APP_EMBED_LOCAL_RUNTIME_WRAPPER=0 \
MTPLX_RUNTIME_WHEEL="$PYTHON_WHEEL" \
MTPLX_NATIVE_RUNTIME_WHEEL="$NATIVE_RUNTIME_WHEEL" \
MTPLX_REQUIRE_RUNTIME_WHEEL_RESOURCE=1 \
MTPLX_BUNDLED_PYTHON_DIR="$PBS_EXTRACT_DIR" \
MTPLX_REQUIRE_BUNDLED_PYTHON_RESOURCE=1 \
MTPLX_REQUIRE_THERMALFORGE_RESOURCE=1 \
MTPLX_CODESIGN_IDENTITY="$CODESIGN_IDENTITY" \
"$BUILD_SCRIPT" --no-launch

APP_BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP_BUNDLE/Contents/Info.plist")"
echo "Built MTPLX.app $VERSION ($APP_BUILD)"

/usr/bin/codesign --verify --deep --strict --verbose=4 "$APP_BUNDLE"

# Library-validation gate: without this entitlement on the bundled
# interpreter, every pip-installed wheel's linker-signed extension is
# rejected with "different Team IDs" on macOS 15 and earlier and the
# engine dies on its first numpy import on customer Macs (macOS 26
# relaxed the rule, so dev machines never reproduce it).
PYTHON_BIN_DIR="$APP_BUNDLE/Contents/Resources/PythonRuntime/bin"
ENTITLED=0
for candidate in "$PYTHON_BIN_DIR"/python3*; do
  [[ -f "$candidate" && ! -L "$candidate" ]] || continue
  /usr/bin/file "$candidate" | /usr/bin/grep -q 'Mach-O' || continue
  if /usr/bin/codesign -d --entitlements - "$candidate" 2>/dev/null \
    | /usr/bin/grep -q 'disable-library-validation'; then
    ENTITLED=1
  else
    echo "error: $candidate lacks com.apple.security.cs.disable-library-validation; wheels will not load on macOS 15" >&2
    exit 1
  fi
done
if [[ "$ENTITLED" != "1" ]]; then
  echo "error: no bundled python interpreter found to verify the library-validation entitlement" >&2
  exit 1
fi

if /usr/bin/grep -R -I -E '/Users/youssof|[0-9a-f]+-dirty|MTPLXLocalRuntimeWrapperPath' "$APP_BUNDLE" >/dev/null 2>&1; then
  echo "error: signed app contains a local path, dirty marker, or local runtime wrapper key" >&2
  exit 1
fi

if [[ "${MTPLX_SKIP_NOTARIZATION:-0}" != "1" ]]; then
  echo "Submitting app for notarization"
  /usr/bin/ditto -c -k --keepParent --norsrc "$APP_BUNDLE" "$APP_NOTARY_ZIP"
  submit_notarization "$APP_NOTARY_ZIP"
  xcrun stapler staple "$APP_BUNDLE"
  xcrun stapler validate "$APP_BUNDLE"
  /usr/sbin/spctl --assess --type execute --verbose "$APP_BUNDLE"
fi

echo "Creating DMG"
/bin/rm -rf "$DMG_STAGE"
/bin/mkdir -p "$DMG_STAGE"
/usr/bin/ditto --norsrc "$APP_BUNDLE" "$DMG_STAGE/MTPLX.app"
/bin/ln -s /Applications "$DMG_STAGE/Applications"
/usr/bin/xattr -rc "$DMG_STAGE" >/dev/null 2>&1 || true
/usr/bin/find "$DMG_STAGE" -depth -exec /usr/bin/xattr -c {} + >/dev/null 2>&1 || true

/usr/bin/hdiutil create \
  -volname "MTPLX $VERSION" \
  -srcfolder "$DMG_STAGE" \
  -ov \
  -format UDZO \
  "$DMG"

/usr/bin/codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$DMG"
/usr/bin/codesign --verify --deep --strict --verbose=4 "$APP_BUNDLE"

if [[ "${MTPLX_SKIP_NOTARIZATION:-0}" != "1" ]]; then
  echo "Submitting DMG for notarization"
  submit_notarization "$DMG"
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG"
  /usr/sbin/spctl --assess --type execute --verbose "$APP_BUNDLE"
  /usr/sbin/spctl --assess --type open --context context:primary-signature --verbose "$DMG"
else
  echo "warning: notarization skipped; this artifact is not release-ready" >&2
fi

DMG_SHA256="$(/usr/bin/shasum -a 256 "$DMG" | /usr/bin/awk '{print $1}')"
DMG_SIZE="$(/usr/bin/stat -f%z "$DMG")"
printf '%s  %s\n' "$DMG_SHA256" "$(basename "$DMG")" > "$DMG.sha256"

RELEASE_NOTES_URL="https://mtplx.com/releases/notes/v$VERSION.html"
DMG_URL="$GITHUB_ASSET_BASE/$(basename "$DMG")"

# Sparkle's "what's new" dialog and the hosted notes page render the real
# release notes authored in docs/releases/v$VERSION.md (gated above).
# scripts/render_release_notes.py owns the page template — light AND dark
# mode readable (#367) — and the rehearsal kit renders through the same
# file, so do not inline a template copy here again.
"$PYTOOLS_VENV/bin/python" "$ROOT/scripts/render_release_notes.py" \
  "$RELEASE_NOTES_MD" "$NOTES_OUT/v$VERSION.html" "$VERSION"

python3 - "$RELEASES_OUT/latest.json" "$VERSION" "$APP_BUILD" "$DMG_URL" "$DMG_SHA256" "$DMG_SIZE" "$RELEASE_NOTES_URL" <<'PY'
import datetime
import json
import sys

path, version, build, dmg_url, sha, size, notes = sys.argv[1:]
payload = {
    "app_version": version,
    "app_build": build,
    "minimum_cli_version": version,
    "recommended_cli_version": version,
    "dmg_url": dmg_url,
    "dmg_sha256": sha,
    "dmg_size_bytes": int(size),
    "pypi_version": version,
    "homebrew_formula_version": version,
    "release_notes_url": notes,
    "published_at": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

/usr/bin/ditto --norsrc "$DMG" "$SPARKLE_ARCHIVES/$(basename "$DMG")"
/usr/bin/ditto --norsrc "$NOTES_OUT/v$VERSION.html" "$SPARKLE_ARCHIVES/MTPLX-$VERSION.html"
/usr/bin/ditto --norsrc "$NOTES_OUT/v$VERSION.html" "$NOTES_OUT/MTPLX-$VERSION.html"

GENERATE_APPCAST="$(
  /usr/bin/find "$APP_ROOT/.build" -path '*/bin/generate_appcast' -type f -print 2>/dev/null | /usr/bin/head -n 1
)"
if [[ -z "$GENERATE_APPCAST" ]]; then
  echo "error: Sparkle generate_appcast tool not found under $APP_ROOT/.build" >&2
  exit 1
fi

GENERATE_KEYS="$(
  /usr/bin/find "$APP_ROOT/.build" -path '*/bin/generate_keys' -type f -print 2>/dev/null | /usr/bin/head -n 1
)"
if [[ -z "$GENERATE_KEYS" ]]; then
  echo "error: Sparkle generate_keys tool not found under $APP_ROOT/.build" >&2
  exit 1
fi

if [[ -z "$SPARKLE_PRIVATE_KEY" && -z "$SPARKLE_PRIVATE_KEY_FILE" && "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" == "1" ]]; then
  KEYCHAIN_PUBLIC_KEY="$("$GENERATE_KEYS" -p --account "$SPARKLE_KEY_ACCOUNT")"
  if [[ "$KEYCHAIN_PUBLIC_KEY" != "$MTPLX_SPARKLE_PUBLIC_ED_KEY" ]]; then
    echo "error: Sparkle Keychain public key does not match MTPLX_SPARKLE_PUBLIC_ED_KEY" >&2
    echo "  keychain[$SPARKLE_KEY_ACCOUNT]: $KEYCHAIN_PUBLIC_KEY" >&2
    echo "  app build public key: $MTPLX_SPARKLE_PUBLIC_ED_KEY" >&2
    exit 1
  fi
fi

APPCAST_ARGS=()
if "$GENERATE_APPCAST" --help 2>&1 | /usr/bin/grep -q -- '--download-url-prefix'; then
  APPCAST_ARGS+=(--download-url-prefix "$GITHUB_ASSET_BASE/")
fi
if "$GENERATE_APPCAST" --help 2>&1 | /usr/bin/grep -q -- '--full-release-notes-url'; then
  APPCAST_ARGS+=(--full-release-notes-url "$RELEASE_NOTES_URL")
fi

if [[ -n "$SPARKLE_PRIVATE_KEY" ]]; then
  printf '%s' "$SPARKLE_PRIVATE_KEY" | "$GENERATE_APPCAST" --ed-key-file - "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
elif [[ -n "$SPARKLE_PRIVATE_KEY_FILE" ]]; then
  "$GENERATE_APPCAST" --ed-key-file "$SPARKLE_PRIVATE_KEY_FILE" "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
elif [[ "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" == "1" ]]; then
  "$GENERATE_APPCAST" --account "$SPARKLE_KEY_ACCOUNT" "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
else
  echo "error: Sparkle appcast signing key missing; set MTPLX_SPARKLE_PRIVATE_KEY or MTPLX_SPARKLE_PRIVATE_KEY_FILE" >&2
  echo "note: set MTPLX_SPARKLE_ALLOW_KEYCHAIN=1 only for interactive local signing, because Keychain access can block unattended releases" >&2
  exit 1
fi

APPCAST_SOURCE="$(
  /usr/bin/find "$SPARKLE_ARCHIVES" -maxdepth 1 -name '*.xml' -type f -print | /usr/bin/head -n 1
)"
if [[ -z "$APPCAST_SOURCE" ]]; then
  echo "error: generate_appcast did not produce an XML appcast" >&2
  exit 1
fi
/usr/bin/ditto --norsrc "$APPCAST_SOURCE" "$RELEASES_OUT/appcast.xml"

python3 - "$RELEASES_OUT/appcast.xml" "$RELEASE_NOTES_URL" "$VERSION" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
release_notes_url = sys.argv[2]
version = sys.argv[3]
xml = path.read_text(encoding="utf-8")
replacements = [
    f"https://mtplx.com/releases/MTPLX-{version}.html",
    f"https://mtplx.com/releases/notes/MTPLX-{version}.html",
]
for old in replacements:
    xml = xml.replace(old, release_notes_url)
if release_notes_url not in xml:
    raise SystemExit("error: appcast does not contain the expected release notes URL")
path.write_text(xml, encoding="utf-8")
PY

if ! /usr/bin/grep -q 'sparkle:edSignature' "$RELEASES_OUT/appcast.xml"; then
  echo "error: appcast is missing Sparkle EdDSA signature" >&2
  exit 1
fi

cat > "$SITE_OUT/download.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=$DMG_URL">
<title>Download MTPLX</title>
<p><a href="$DMG_URL">Download MTPLX $VERSION</a></p>
HTML

cat <<SUMMARY

Release artifact staged locally.
DMG: $DMG
SHA256: $DMG_SHA256
Size: $DMG_SIZE
Website payload: $SITE_OUT

No upload has been performed by this script.
SUMMARY
