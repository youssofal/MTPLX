#!/usr/bin/env bash
# Removes the rpaths that point into this Mac's toolchains from a linked
# binary, then fails if any absolute rpath other than /usr/lib/swift is left.
#
#   strip_build_rpaths.sh BINARY
#
# SwiftPM's build system adds a Swift back-deploy rpath into every active
# toolchain (<toolchain>/usr/lib/swift-6.2/macosx) because the app deploys to
# macOS 14. Under Xcode 27 that is two paths: Xcode.app's own toolchain and
# the Metal toolchain's /var/run cryptex mount, whose name changes per mount.
# Neither exists on a user's Mac and the app loads nothing through them: its
# one @rpath library, Sparkle, comes from Contents/Frameworks.
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "usage: strip_build_rpaths.sh BINARY" >&2
  exit 2
fi
binary="$1"

rpaths() {
  /usr/bin/otool -l "$binary" | /usr/bin/awk '
    $1 == "cmd" { in_rpath = ($2 == "LC_RPATH"); next }
    in_rpath && $1 == "path" {
      sub(/^[ \t]*path /, ""); sub(/ \(offset [0-9]+\)$/, ""); print; in_rpath = 0
    }'
}

# Read the whole list before changing the file.
listed="$(rpaths)"
while IFS= read -r rpath; do
  case "$rpath" in
    */usr/lib/swift-*) /usr/bin/install_name_tool -delete_rpath "$rpath" "$binary" ;;
  esac
done <<< "$listed"

left="$(rpaths | /usr/bin/grep '^/' | /usr/bin/grep -v -x '/usr/lib/swift' || true)"
if [[ -n "$left" ]]; then
  echo "error: $binary still carries build-machine rpaths:" >&2
  printf '  %s\n' "$left" >&2
  exit 1
fi
