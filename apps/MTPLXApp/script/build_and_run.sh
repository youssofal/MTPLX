#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
APP_NAME="MTPLXApp"

canonical_bundle_dir() {
  local raw="$1"
  local parent
  local base
  parent="$(dirname "$raw")"
  base="$(basename "$raw")"
  mkdir -p "$parent"
  parent="$(cd "$parent" && pwd -P)"
  printf '%s/%s' "$parent" "$base"
}

DEFAULT_BUNDLE_DIR="$(canonical_bundle_dir "$ROOT/dist/$APP_NAME.app")"
BUNDLE_DIR="$(canonical_bundle_dir "${MTPLX_APP_BUNDLE_DIR:-$DEFAULT_BUNDLE_DIR}")"
BUILD_CONFIG="${MTPLX_APP_BUILD_CONFIG:-release}"
BUILD_DIR="$ROOT/.build/$BUILD_CONFIG"
EXECUTABLE="$BUILD_DIR/$APP_NAME"
ICNS_SOURCE="$ROOT/Resources/AppIcon.icns"
THERMALFORGE_SOURCE="${MTPLX_THERMALFORGE_BINARY:-$HOME/.mtplx/bin/thermalforge}"
REQUIRE_THERMALFORGE_RESOURCE="${MTPLX_REQUIRE_THERMALFORGE_RESOURCE:-0}"
RUNTIME_WHEEL_SOURCE="${MTPLX_RUNTIME_WHEEL:-}"
NATIVE_RUNTIME_WHEEL_SOURCE="${MTPLX_NATIVE_RUNTIME_WHEEL:-}"
REQUIRE_RUNTIME_WHEEL_RESOURCE="${MTPLX_REQUIRE_RUNTIME_WHEEL_RESOURCE:-0}"
BUNDLED_PYTHON_DIR="${MTPLX_BUNDLED_PYTHON_DIR:-}"
REQUIRE_BUNDLED_PYTHON_RESOURCE="${MTPLX_REQUIRE_BUNDLED_PYTHON_RESOURCE:-0}"
APP_VERSION="${MTPLX_APP_VERSION:-$(/usr/bin/awk -F'"' '/^version = / { print $2; exit }' "$REPO_ROOT/pyproject.toml" 2>/dev/null || true)}"
APP_VERSION="${APP_VERSION:-1.0.0}"
# Sparkle ranks updates by CFBundleVersion alone. script/app_build_number.sh
# is the one place the number is derived (see its header): a build of this
# version carries its base number unless the caller names a release revision
# through MTPLX_APP_BUILD (scripts/release_macos_v1.sh does).
BASE_APP_BUILD="$("$ROOT/script/app_build_number.sh" "$APP_VERSION" 2>/dev/null || true)"
APP_BUILD="${MTPLX_APP_BUILD:-$BASE_APP_BUILD}"
APP_BUILD="${APP_BUILD:-$(/bin/date +%Y%m%d%H%M)}"
# The deployment target: Package.swift's `.macOS(.v14)`, the Info.plist's
# LSMinimumSystemVersion and the linker's platform_version all carry it.
MACOS_DEPLOYMENT_TARGET="14.0"
# The SDK the app compiles against. Xcode 27's `swift build` stamps the
# binary's LC_BUILD_VERSION sdk with the deployment target (sdk 14.0) although
# it compiles against MacOSX27.0.sdk, and macOS draws a binary that claims an
# SDK older than 26 in the pre-Liquid Glass style. The linker gets the real
# SDK below, and the gate after linking refuses a binary stamped below 26.
MACOS_SDK_VERSION="$(/usr/bin/xcrun --show-sdk-version)"
# Contributors whose newest SDK is older than 26 (Xcode 16 on macOS 14) can
# still build a local app with this set; a public release never can.
ALLOW_PRE_GLASS_SDK="${MTPLX_APP_ALLOW_PRE_GLASS_SDK:-0}"
EMBED_LOCAL_RUNTIME_WRAPPER="${MTPLX_APP_EMBED_LOCAL_RUNTIME_WRAPPER:-0}"
PUBLIC_RELEASE="${MTPLX_APP_PUBLIC_RELEASE:-0}"
SPARKLE_FEED_URL="${MTPLX_APPCAST_URL:-https://mtplx.com/releases/appcast.xml}"
SPARKLE_PUBLIC_ED_KEY="${MTPLX_SPARKLE_PUBLIC_ED_KEY:-GQ0sTm6nb5kv+Btri7wc4LqnXGZ48vIs6PGMwsI/mBM=}"
SPARKLE_CHECK_INTERVAL_SECONDS="${MTPLX_SPARKLE_CHECK_INTERVAL_SECONDS:-86400}"
TRIM_SWIFTUI_MATH_FONTS="${MTPLX_TRIM_SWIFTUI_MATH_FONTS:-1}"
BUNDLE_IDENTIFIER_EXPLICIT=0
BUNDLE_DISPLAY_NAME_EXPLICIT=0
if [[ -n "${MTPLX_APP_BUNDLE_IDENTIFIER:-}" ]]; then
  BUNDLE_IDENTIFIER="$MTPLX_APP_BUNDLE_IDENTIFIER"
  BUNDLE_IDENTIFIER_EXPLICIT=1
else
  BUNDLE_IDENTIFIER="com.youssofal.mtplx"
fi
if [[ -n "${MTPLX_APP_DISPLAY_NAME:-}" ]]; then
  BUNDLE_DISPLAY_NAME="$MTPLX_APP_DISPLAY_NAME"
  BUNDLE_DISPLAY_NAME_EXPLICIT=1
else
  BUNDLE_DISPLAY_NAME="MTPLX"
fi
VERIFY=0
NO_LAUNCH=0
ISOLATED_BUNDLE_IDENTIFIER=0
APP_ARGS=()
export COPYFILE_DISABLE=1

bundle_id_suffix() {
  local raw="$1"
  local suffix
  suffix="$(printf '%s' "$raw" \
    | /usr/bin/tr '[:upper:]' '[:lower:]' \
    | /usr/bin/sed -E 's/[^a-z0-9]+/./g; s/^[.]+//; s/[.]+$//; s/[.]+/./g')"
  printf '%s' "$suffix"
}

display_suffix() {
  local raw="$1"
  local suffix
  suffix="${raw#mtplx-v1-}"
  suffix="${suffix#mtplx-}"
  suffix="$(printf '%s' "$suffix" \
    | /usr/bin/sed -E 's/[^[:alnum:]]+/ /g; s/^ +//; s/ +$//')"
  printf '%s' "$suffix"
}

trim_swiftui_math_fonts() {
  [[ "$TRIM_SWIFTUI_MATH_FONTS" == "1" ]] || return 0
  local fonts_dir="$BUNDLE_DIR/swiftui-math_SwiftUIMath.bundle/mathFonts.bundle"
  [[ -d "$fonts_dir" ]] || return 0
  /usr/bin/find "$fonts_dir" -type f \( -name '*.otf' -o -name '*.plist' \) \
    ! -name 'latinmodern-math.otf' \
    ! -name 'latinmodern-math.plist' \
    -exec /bin/rm -f {} +
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify)
      VERIFY=1
      ;;
    --no-launch)
      NO_LAUNCH=1
      ;;
    --isolated-bundle-id|--qa-bundle-id)
      ISOLATED_BUNDLE_IDENTIFIER=1
      ;;
    --bundle-id|--bundle-identifier)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: missing value for bundle id" >&2
        exit 2
      fi
      BUNDLE_IDENTIFIER="$1"
      BUNDLE_IDENTIFIER_EXPLICIT=1
      ;;
    --bundle-id=*|--bundle-identifier=*)
      BUNDLE_IDENTIFIER="${1#*=}"
      BUNDLE_IDENTIFIER_EXPLICIT=1
      ;;
    --display-name)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: missing value for display name" >&2
        exit 2
      fi
      BUNDLE_DISPLAY_NAME="$1"
      BUNDLE_DISPLAY_NAME_EXPLICIT=1
      ;;
    --display-name=*)
      BUNDLE_DISPLAY_NAME="${1#*=}"
      BUNDLE_DISPLAY_NAME_EXPLICIT=1
      ;;
    --settings-path|--app-settings|--mtplx-app-settings|--mtplx-settings-path)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: missing value for settings path" >&2
        exit 2
      fi
      APP_ARGS+=("--mtplx-app-settings" "$1")
      ;;
    --settings-path=*|--app-settings=*)
      APP_ARGS+=("--mtplx-app-settings" "${1#*=}")
      ;;
    --mtplx-app-settings=*|--mtplx-settings-path=*)
      APP_ARGS+=("$1")
      ;;
    --chat-store-path|--mtplx-chat-store|--mtplx-chat-store-path)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: missing value for chat store path" >&2
        exit 2
      fi
      APP_ARGS+=("--mtplx-chat-store-path" "$1")
      ;;
    --chat-store-path=*|--mtplx-chat-store=*|--mtplx-chat-store-path=*)
      APP_ARGS+=("--mtplx-chat-store-path" "${1#*=}")
      ;;
    --)
      shift
      APP_ARGS+=("$@")
      break
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$ISOLATED_BUNDLE_IDENTIFIER" == "1" && "$BUNDLE_IDENTIFIER_EXPLICIT" == "0" ]]; then
  suffix="$(bundle_id_suffix "$(basename "$REPO_ROOT")")"
  if [[ -n "$suffix" ]]; then
    BUNDLE_IDENTIFIER="$BUNDLE_IDENTIFIER.$suffix"
  fi
fi
if [[ "$ISOLATED_BUNDLE_IDENTIFIER" == "1" && "$BUNDLE_DISPLAY_NAME_EXPLICIT" == "0" ]]; then
  suffix="$(display_suffix "$(basename "$REPO_ROOT")")"
  if [[ -n "$suffix" ]]; then
    BUNDLE_DISPLAY_NAME="MTPLX $suffix"
  else
    BUNDLE_DISPLAY_NAME="MTPLX QA"
  fi
fi

cd "$ROOT"

app_pids() {
  /bin/ps -axo pid=,command= | while read -r pid command; do
    [[ -n "${pid:-}" ]] || continue
    if [[ "${command:-}" == "$BUNDLE_DIR/Contents/MacOS/$APP_NAME"* ]]; then
      echo "$pid"
    fi
  done
}

misdirected_app_pids() {
  /bin/ps -axo pid=,command= | while read -r pid command; do
    [[ -n "${pid:-}" ]] || continue
    if [[ "${command:-}" != "$BUNDLE_DIR/Contents/MacOS/$APP_NAME"* ]] \
      && [[ "${command:-}" == *".app/Contents/MacOS/$APP_NAME"* ]]; then
      echo "$pid"
    fi
  done
}

kill_tree() {
  local pid="$1"
  while read -r child; do
    [[ -n "$child" ]] || continue
    kill_tree "$child"
  done < <(/bin/ps -axo pid=,ppid= | /usr/bin/awk -v parent="$pid" '$2 == parent { print $1 }')
  /bin/kill "$pid" >/dev/null 2>&1 || true
}

# Packaging must not interrupt an app the user is currently using. Refuse
# to overwrite the exact running target; a separate bundle can be built safely.
if [[ "$NO_LAUNCH" == "1" ]]; then
  if [[ -n "$(app_pids)" ]]; then
    echo "error: target bundle is running; set MTPLX_APP_BUNDLE_DIR to a separate output path" >&2
    exit 1
  fi
else
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    kill_tree "$pid"
  done < <(app_pids)
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    kill_tree "$pid"
  done < <(misdirected_app_pids)
  for _ in {1..50}; do
    if [[ -z "$(app_pids)" && -z "$(misdirected_app_pids)" ]]; then
      break
    fi
    sleep 0.1
  done
fi

if [[ "$PUBLIC_RELEASE" == "1" ]]; then
  # A public release is a release revision of its version's base number:
  # above every QA build of that version, below the next version's.
  if [[ -z "$BASE_APP_BUILD" ]]; then
    echo "error: public release version '$APP_VERSION' has no derivable build number" >&2
    exit 1
  fi
  if [[ ! "$APP_BUILD" =~ ^[0-9]+$ ]] \
    || (( APP_BUILD <= BASE_APP_BUILD || APP_BUILD > BASE_APP_BUILD + 9 )); then
    echo "error: public release CFBundleVersion '$APP_BUILD' is not a release revision of $APP_VERSION ($((BASE_APP_BUILD + 1)) to $((BASE_APP_BUILD + 9)))" >&2
    echo "       derive it with script/app_build_number.sh $APP_VERSION <1-9>" >&2
    exit 1
  fi
  if [[ "$ALLOW_PRE_GLASS_SDK" == "1" ]]; then
    echo "error: MTPLX_APP_ALLOW_PRE_GLASS_SDK is for local builds; a public release must be stamped with SDK 26 or later" >&2
    exit 1
  fi
fi

swift build -c "$BUILD_CONFIG" --product "$APP_NAME" \
  -Xlinker -platform_version -Xlinker macos \
  -Xlinker "$MACOS_DEPLOYMENT_TARGET" -Xlinker "$MACOS_SDK_VERSION"

/bin/rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/Contents/MacOS" "$BUNDLE_DIR/Contents/Resources" "$BUNDLE_DIR/Contents/Frameworks"
/usr/bin/ditto --norsrc "$EXECUTABLE" "$BUNDLE_DIR/Contents/MacOS/$APP_NAME"

SPARKLE_FRAMEWORK="$(
  /usr/bin/find "$ROOT/.build" -path '*/Sparkle.framework' -type d -prune -print 2>/dev/null | /usr/bin/head -n 1
)"
if [[ -n "$SPARKLE_FRAMEWORK" ]]; then
  /usr/bin/ditto --norsrc "$SPARKLE_FRAMEWORK" "$BUNDLE_DIR/Contents/Frameworks/Sparkle.framework"
else
  echo "error: Sparkle.framework was not found in SwiftPM build artifacts" >&2
  echo "run swift package resolve in apps/MTPLXApp and rebuild" >&2
  exit 1
fi

if ! /usr/bin/otool -l "$BUNDLE_DIR/Contents/MacOS/$APP_NAME" | /usr/bin/grep -q '@executable_path/../Frameworks'; then
  /usr/bin/install_name_tool -add_rpath '@executable_path/../Frameworks' "$BUNDLE_DIR/Contents/MacOS/$APP_NAME"
fi

# Gate: macOS picks the controls' style (Liquid Glass or the old look) from
# the SDK stamped in the main binary, so read the stamp back off the bundle.
BUILD_STAMP="$(/usr/bin/vtool -show-build "$BUNDLE_DIR/Contents/MacOS/$APP_NAME")"
STAMPED_MINOS="$(printf '%s\n' "$BUILD_STAMP" | /usr/bin/awk '$1 == "minos" { print $2; exit }')"
STAMPED_SDK="$(printf '%s\n' "$BUILD_STAMP" | /usr/bin/awk '$1 == "sdk" { print $2; exit }')"
STAMPED_SDK_MAJOR="${STAMPED_SDK%%.*}"
if [[ ! "$STAMPED_SDK_MAJOR" =~ ^[0-9]+$ ]]; then
  echo "error: no SDK stamp in $BUNDLE_DIR/Contents/MacOS/$APP_NAME:" >&2
  printf '%s\n' "$BUILD_STAMP" >&2
  exit 1
fi
if (( STAMPED_SDK_MAJOR < 26 )); then
  if [[ "$ALLOW_PRE_GLASS_SDK" == "1" ]]; then
    echo "warning: $APP_NAME is stamped sdk $STAMPED_SDK (built with SDK $MACOS_SDK_VERSION); macOS draws it without Liquid Glass" >&2
  else
    echo "error: $APP_NAME is stamped sdk $STAMPED_SDK (built with SDK $MACOS_SDK_VERSION); macOS would draw it without Liquid Glass" >&2
    if (( ${MACOS_SDK_VERSION%%.*} >= 26 )); then
      echo "       the linker ignored -platform_version macos $MACOS_DEPLOYMENT_TARGET $MACOS_SDK_VERSION; check the toolchain's swift build" >&2
    else
      echo "       build with Xcode 26 or later, or set MTPLX_APP_ALLOW_PRE_GLASS_SDK=1 for a local build on an older SDK" >&2
    fi
    exit 1
  fi
fi
echo "$APP_NAME stamped minos $STAMPED_MINOS sdk $STAMPED_SDK, build $APP_BUILD"

# App-owned resources are copied flat into `Contents/Resources` and resolved
# through `Bundle.main`. Do not copy SwiftPM-generated resource bundles here:
# stale bundles can linger in `.build`, and bundles beside `Contents/` make a
# Developer ID app unsignable.
if [[ -f "$ROOT/Sources/MTPLXAppHost/Resources/Wordmark/Wordmark.png" ]]; then
  /usr/bin/ditto --norsrc \
    "$ROOT/Sources/MTPLXAppHost/Resources/Wordmark/Wordmark.png" \
    "$BUNDLE_DIR/Contents/Resources/Wordmark.png"
fi
if [[ -d "$ROOT/Sources/MTPLXAppHost/Resources/Brands" ]]; then
  /usr/bin/find "$ROOT/Sources/MTPLXAppHost/Resources/Brands" -maxdepth 1 -type f -name '*.png' -print0 \
    | while IFS= read -r -d '' brand_png; do
        /usr/bin/ditto --norsrc "$brand_png" "$BUNDLE_DIR/Contents/Resources/$(basename "$brand_png")"
      done
fi
if [[ -d "$ROOT/Sources/MTPLXAppCore/Resources/StepAdapters" ]]; then
  mkdir -p "$BUNDLE_DIR/Contents/Resources/StepAdapters"
  /usr/bin/ditto --norsrc \
    "$ROOT/Sources/MTPLXAppCore/Resources/StepAdapters" \
    "$BUNDLE_DIR/Contents/Resources/StepAdapters"
fi
# Localized string tables. Each <code>.lproj is copied to the standard
# Contents/Resources/<code>.lproj location so Bundle.main resolves it at
# runtime (L10n opens each lproj as its own bundle). The list mirrors
# AppLanguage.allCases (a unit test keeps them in sync); a missing or
# unparsable table fails the build so a broken language never ships.
LOCALIZATION_SOURCE="$ROOT/Sources/MTPLXAppCore/Resources/Localization"
LOCALIZATION_CODES=(en zh-Hans es hi ar pt-BR fr ru ja de ko id tr)
for code in "${LOCALIZATION_CODES[@]}"; do
  table="$LOCALIZATION_SOURCE/$code.lproj/Localizable.strings"
  if [[ ! -f "$table" ]]; then
    echo "error: localization table missing for $code at $table" >&2
    exit 1
  fi
  if ! /usr/bin/plutil -lint "$table" >/dev/null 2>&1; then
    echo "error: localization table for $code does not parse: $table" >&2
    exit 1
  fi
  mkdir -p "$BUNDLE_DIR/Contents/Resources/$code.lproj"
  /usr/bin/ditto --norsrc "$table" "$BUNDLE_DIR/Contents/Resources/$code.lproj/Localizable.strings"
done
if [[ -f "$THERMALFORGE_SOURCE" ]]; then
  mkdir -p "$BUNDLE_DIR/Contents/Resources/ThermalForge"
  /usr/bin/ditto --norsrc \
    "$THERMALFORGE_SOURCE" \
    "$BUNDLE_DIR/Contents/Resources/ThermalForge/thermalforge"
  /bin/chmod 755 "$BUNDLE_DIR/Contents/Resources/ThermalForge/thermalforge"
elif [[ "$REQUIRE_THERMALFORGE_RESOURCE" == "1" ]]; then
  echo "error: ThermalForge resource missing at $THERMALFORGE_SOURCE" >&2
  echo "set MTPLX_THERMALFORGE_BINARY or run mtplx max --install before building the release app" >&2
  exit 1
else
  echo "warning: ThermalForge resource missing at $THERMALFORGE_SOURCE; app will fall back to source install" >&2
fi
if [[ -n "$RUNTIME_WHEEL_SOURCE" && -f "$RUNTIME_WHEEL_SOURCE" ]]; then
  mkdir -p "$BUNDLE_DIR/Contents/Resources/Runtime"
  /usr/bin/ditto --norsrc \
    "$RUNTIME_WHEEL_SOURCE" \
    "$BUNDLE_DIR/Contents/Resources/Runtime/$(basename "$RUNTIME_WHEEL_SOURCE")"
elif [[ "$REQUIRE_RUNTIME_WHEEL_RESOURCE" == "1" ]]; then
  echo "error: runtime wheel resource missing at $RUNTIME_WHEEL_SOURCE" >&2
  echo "set MTPLX_RUNTIME_WHEEL to the mtplx release wheel before building the release app" >&2
  exit 1
fi
if [[ -n "$NATIVE_RUNTIME_WHEEL_SOURCE" ]]; then
  if [[ ! -f "$NATIVE_RUNTIME_WHEEL_SOURCE" || ! -f "$RUNTIME_WHEEL_SOURCE" ]]; then
    echo "error: native runtime requires both native and pure fallback wheels" >&2
    exit 1
  fi
  mkdir -p "$BUNDLE_DIR/Contents/Resources/Runtime/Native"
  /usr/bin/ditto --norsrc "$NATIVE_RUNTIME_WHEEL_SOURCE" \
    "$BUNDLE_DIR/Contents/Resources/Runtime/Native/$(basename "$NATIVE_RUNTIME_WHEEL_SOURCE")"
  /usr/bin/ditto --norsrc "$REPO_ROOT/scripts/select_runtime_wheel.py" \
    "$BUNDLE_DIR/Contents/Resources/Runtime/select_runtime_wheel.py"
fi
# Bundled Python interpreter (python-build-standalone install_only_stripped
# tree). With it in Contents/Resources/PythonRuntime, the app can build its
# runtime venv on Macs that have no Homebrew or Xcode at all.
if [[ -n "$BUNDLED_PYTHON_DIR" && -x "$BUNDLED_PYTHON_DIR/bin/python3" ]]; then
  /usr/bin/ditto --norsrc \
    "$BUNDLED_PYTHON_DIR" \
    "$BUNDLE_DIR/Contents/Resources/PythonRuntime"
elif [[ "$REQUIRE_BUNDLED_PYTHON_RESOURCE" == "1" ]]; then
  echo "error: bundled Python runtime missing at $BUNDLED_PYTHON_DIR" >&2
  echo "set MTPLX_BUNDLED_PYTHON_DIR to an extracted python-build-standalone tree (bin/python3) before building the release app" >&2
  exit 1
elif [[ -n "$BUNDLED_PYTHON_DIR" ]]; then
  echo "warning: MTPLX_BUNDLED_PYTHON_DIR set but $BUNDLED_PYTHON_DIR/bin/python3 is not executable; skipping bundled Python" >&2
fi

# Icon: build_app_icon.py emits Resources/AppIcon.icns once and we copy
# it into every bundle build. The icon name in the plist must match the
# basename of the file inside Contents/Resources (without extension).
if [[ -f "$ICNS_SOURCE" ]]; then
  /usr/bin/ditto --norsrc "$ICNS_SOURCE" "$BUNDLE_DIR/Contents/Resources/AppIcon.icns"
else
  echo "warning: $ICNS_SOURCE missing; bundle will use a generic icon" >&2
fi

LOCAL_RUNTIME_WRAPPER_PLIST=""
if [[ "$EMBED_LOCAL_RUNTIME_WRAPPER" == "1" ]]; then
  LOCAL_RUNTIME_WRAPPER_PLIST="$(cat <<LOCAL_RUNTIME_PLIST
  <key>MTPLXLocalRuntimeWrapperPath</key>
  <string>$REPO_ROOT/bin/mtplx</string>
LOCAL_RUNTIME_PLIST
)"
fi

cat > "$BUNDLE_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIconName</key>
  <string>AppIcon</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_IDENTIFIER</string>
  <key>CFBundleLocalizations</key>
  <array>
$(for code in "${LOCALIZATION_CODES[@]}"; do printf '    <string>%s</string>\n' "$code"; done)
  </array>
  <key>CFBundleName</key>
  <string>$BUNDLE_DISPLAY_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$APP_VERSION</string>
  <key>CFBundleVersion</key>
  <string>$APP_BUILD</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.developer-tools</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MACOS_DEPLOYMENT_TARGET</string>
  <key>MTPLXAllowLocalRuntimeWrapper</key>
  <$([[ "$EMBED_LOCAL_RUNTIME_WRAPPER" == "1" ]] && printf true || printf false)/>
$LOCAL_RUNTIME_WRAPPER_PLIST
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
  <key>SUEnableAutomaticChecks</key>
  <true/>
  <key>SUFeedURL</key>
  <string>$SPARKLE_FEED_URL</string>
  <key>SUPublicEDKey</key>
  <string>$SPARKLE_PUBLIC_ED_KEY</string>
  <key>SUScheduledCheckInterval</key>
  <integer>$SPARKLE_CHECK_INTERVAL_SECONDS</integer>
</dict>
</plist>
PLIST

if [[ "$PUBLIC_RELEASE" == "1" ]]; then
  if [[ ! "$APP_BUILD" =~ ^[0-9]+$ ]]; then
    echo "error: public release CFBundleVersion must be numeric, got '$APP_BUILD'" >&2
    exit 1
  fi
  if [[ "$EMBED_LOCAL_RUNTIME_WRAPPER" == "1" ]]; then
    echo "error: public release bundle must not embed a local runtime wrapper" >&2
    exit 1
  fi
  if [[ -z "$SPARKLE_PUBLIC_ED_KEY" ]]; then
    echo "error: MTPLX_SPARKLE_PUBLIC_ED_KEY is required for public release builds" >&2
    exit 1
  fi
  # A public bundle without these resources installs fine and then cannot
  # self-heal its runtime on user Macs — the failure only surfaces on
  # first engine start, far from the build. Fail here instead. (The wheel
  # copy above silently skips when MTPLX_RUNTIME_WHEEL doesn't resolve —
  # this script cd's to apps/MTPLXApp, so relative repo-root paths dangle;
  # pass an absolute path.)
  if ! /bin/ls "$BUNDLE_DIR/Contents/Resources/Runtime/"*.whl >/dev/null 2>&1; then
    echo "error: public release bundle has no bundled runtime wheel under Contents/Resources/Runtime" >&2
    echo "set MTPLX_RUNTIME_WHEEL to the release wheel (absolute path)" >&2
    exit 1
  fi
  if [[ ! -x "$BUNDLE_DIR/Contents/Resources/PythonRuntime/bin/python3" ]]; then
    echo "error: public release bundle has no bundled Python runtime under Contents/Resources/PythonRuntime" >&2
    echo "set MTPLX_BUNDLED_PYTHON_DIR to an extracted python-build-standalone tree" >&2
    exit 1
  fi
  # The dirty marker matches git-describe shapes (hex-dirty) rather than a
  # bare "-dirty", which false-positives on prose in the bundled Python
  # stdlib ("quick-n-dirty" in idlelib).
  if /usr/bin/grep -R -I -E '/Users/youssof|[0-9a-f]+-dirty|MTPLXLocalRuntimeWrapperPath' "$BUNDLE_DIR" >/dev/null 2>&1; then
    echo "error: public release bundle contains a local path, dirty marker, or local runtime wrapper key" >&2
    exit 1
  fi
fi

# macOS's icon cache for an app keys on bundle id + mtime. Touching the
# bundle ensures Finder/Dock pick up the new .icns without a manual
# cache flush after the first build.
/usr/bin/touch "$BUNDLE_DIR"

if command -v /usr/sbin/dot_clean >/dev/null 2>&1; then
  /usr/sbin/dot_clean -m "$BUNDLE_DIR" >/dev/null 2>&1 || true
fi
/usr/bin/xattr -rc "$BUNDLE_DIR" >/dev/null 2>&1 || true
/usr/bin/find "$BUNDLE_DIR" -depth -exec /usr/bin/xattr -c {} + >/dev/null 2>&1 || true
if [[ -n "${MTPLX_CODESIGN_IDENTITY:-${MTPLX_DEVELOPER_ID_APPLICATION:-}}" ]]; then
  CODESIGN_IDENTITY="${MTPLX_CODESIGN_IDENTITY:-${MTPLX_DEVELOPER_ID_APPLICATION:-}}"
  sign_release_code() {
    local target="$1"
    [[ -e "$target" ]] || return 0
    /usr/bin/codesign --force --options runtime --timestamp --sign "$CODESIGN_IDENTITY" "$target" >/dev/null
  }

  # The bundled interpreter runs the app-owned venv, and pip-installed
  # wheels ship ad-hoc linker-signed extensions. Under our hardened
  # team signature, library validation rejects those with "different
  # Team IDs" on macOS 15 and earlier (macOS 26 relaxed the rule,
  # which is why dev machines never reproduced it) — the engine then
  # dies on its first numpy import on every customer Mac. Hardened
  # runtime stays on; notarization accepts this entitlement.
  PYTHON_ENTITLEMENTS="$(/usr/bin/mktemp -t python-runtime-entitlements)"
  /bin/cat > "$PYTHON_ENTITLEMENTS" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.disable-library-validation</key>
  <true/>
</dict>
</plist>
PLIST
  sign_python_interpreter() {
    local target="$1"
    [[ -e "$target" ]] || return 0
    /usr/bin/codesign --force --options runtime --timestamp \
      --entitlements "$PYTHON_ENTITLEMENTS" \
      --sign "$CODESIGN_IDENTITY" "$target" >/dev/null
  }

  while IFS= read -r nested_bundle; do
    [[ -n "$nested_bundle" ]] || continue
    sign_release_code "$nested_bundle"
  done < <(/usr/bin/find "$BUNDLE_DIR/Contents/Frameworks" -depth -type d \( -name '*.xpc' -o -name '*.app' \) -print)

  while IFS= read -r nested_executable; do
    [[ -n "$nested_executable" ]] || continue
    if /usr/bin/file "$nested_executable" | /usr/bin/grep -q 'Mach-O'; then
      sign_release_code "$nested_executable"
    fi
  done < <(/usr/bin/find "$BUNDLE_DIR/Contents/Frameworks" -type f -perm +111 -print)

  if [[ -f "$BUNDLE_DIR/Contents/Resources/ThermalForge/thermalforge" ]]; then
    sign_release_code "$BUNDLE_DIR/Contents/Resources/ThermalForge/thermalforge"
  fi

  # The bundled Python runtime carries its own Mach-O executables and
  # shared objects; every one must be signed for notarization. The
  # interpreter executables under bin/ additionally need the
  # library-validation entitlement (see above) — entitlements only
  # take effect on main executables, so libraries keep plain signing.
  if [[ -d "$BUNDLE_DIR/Contents/Resources/PythonRuntime" ]]; then
    while IFS= read -r python_macho; do
      [[ -n "$python_macho" ]] || continue
      if /usr/bin/file "$python_macho" | /usr/bin/grep -q 'Mach-O'; then
        case "$python_macho" in
          "$BUNDLE_DIR/Contents/Resources/PythonRuntime/bin/"*)
            sign_python_interpreter "$python_macho"
            ;;
          *)
            sign_release_code "$python_macho"
            ;;
        esac
      fi
    done < <(/usr/bin/find "$BUNDLE_DIR/Contents/Resources/PythonRuntime" -type f \( -perm +111 -o -name '*.dylib' -o -name '*.so' \) -print)
  fi

  while IFS= read -r nested_framework; do
    [[ -n "$nested_framework" ]] || continue
    sign_release_code "$nested_framework"
  done < <(/usr/bin/find "$BUNDLE_DIR/Contents/Frameworks" -depth -type d -name '*.framework' -print)

  /usr/bin/codesign --force --options runtime --timestamp --sign "$CODESIGN_IDENTITY" "$BUNDLE_DIR" >/dev/null
elif [[ "${MTPLX_CODESIGN_BUNDLE:-0}" == "1" ]]; then
  /usr/bin/codesign --force --deep --sign - "$BUNDLE_DIR" >/dev/null
fi

if [[ "$NO_LAUNCH" == "1" ]]; then
  echo "$APP_NAME built at $BUNDLE_DIR"
  exit 0
fi

launch_app() {
  if [[ "$BUNDLE_DIR" != "$DEFAULT_BUNDLE_DIR" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      kill_tree "$pid"
    done < <(misdirected_app_pids)
    local custom_open_status=0
    if [[ ${#APP_ARGS[@]} -gt 0 ]]; then
      /usr/bin/open -n "$BUNDLE_DIR" --args "${APP_ARGS[@]}" || custom_open_status=$?
    else
      /usr/bin/open -n "$BUNDLE_DIR" || custom_open_status=$?
    fi
    if [[ "$custom_open_status" -eq 0 ]]; then
      return 0
    fi
    echo "warning: open failed for $BUNDLE_DIR; launching exact bundle executable directly" >&2
    if [[ ${#APP_ARGS[@]} -gt 0 ]]; then
      "$BUNDLE_DIR/Contents/MacOS/$APP_NAME" "${APP_ARGS[@]}" >/dev/null 2>&1 &
    else
      "$BUNDLE_DIR/Contents/MacOS/$APP_NAME" >/dev/null 2>&1 &
    fi
    return 0
  fi

  local open_status=0
  if [[ ${#APP_ARGS[@]} -gt 0 ]]; then
    /usr/bin/open -n "$BUNDLE_DIR" --args "${APP_ARGS[@]}" || open_status=$?
  else
    /usr/bin/open -n "$BUNDLE_DIR" || open_status=$?
  fi
  if [[ "$open_status" -eq 0 ]]; then
    return 0
  fi

  echo "warning: open failed for $BUNDLE_DIR; launching exact bundle executable directly" >&2
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    kill_tree "$pid"
  done < <(misdirected_app_pids)
  if [[ ${#APP_ARGS[@]} -gt 0 ]]; then
    "$BUNDLE_DIR/Contents/MacOS/$APP_NAME" "${APP_ARGS[@]}" >/dev/null 2>&1 &
  else
    "$BUNDLE_DIR/Contents/MacOS/$APP_NAME" >/dev/null 2>&1 &
  fi
}

launch_app

if [[ "$VERIFY" == "1" ]]; then
  for _ in {1..30}; do
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      sleep 1
      if /bin/ps -p "$pid" >/dev/null 2>&1; then
        echo "$APP_NAME launched (pid $pid)"
        exit 0
      fi
    done < <(app_pids)
    sleep 0.2
  done
  echo "$APP_NAME did not launch from $BUNDLE_DIR" >&2
  if [[ -n "$(app_pids)" ]]; then
    echo "running $APP_NAME processes:" >&2
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      /bin/ps -p "$pid" -o pid=,command= >&2 || true
    done < <(app_pids)
  fi
  exit 1
fi
