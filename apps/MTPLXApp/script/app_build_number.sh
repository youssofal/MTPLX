#!/usr/bin/env bash
# Prints the app's CFBundleVersion for a version: the one place it is derived.
#
#   app_build_number.sh VERSION [REVISION]
#
# Sparkle ranks updates by CFBundleVersion alone, so every build of a newer
# version must rank above every build of an older one, whatever produced it.
# The number is major*1000000 + minor*1000 + patch*10 + REVISION:
#
# - REVISION 0 (the default) is what local and QA builds carry, so a QA build
#   of 2.12.3 (2012030) ranks above every 2.12.2 build and is never offered
#   2.12.2 as an update.
# - REVISION 1 to 9 are release builds (scripts/release_macos_v1.sh takes the
#   first one above everything this Mac has staged), so a release ranks above
#   the QA builds of its own version and below the next version's.
#
# Releases up to 2.12.2 used the low digits as a per-build counter (2.12.2
# shipped as 2012011); the tens digit for the patch clears that history from
# 2.12.3 on. A field that would carry into the one above it fails closed.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: app_build_number.sh VERSION [REVISION]" >&2
  exit 2
fi

version="$1"
revision="${2:-0}"
# A dotted suffix (a dev build's 2.12.4.dev0) ranks with its base version.
if [[ ! "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(\..*)?$ ]]; then
  echo "error: version must be major.minor.patch, got '$version'" >&2
  exit 1
fi
# 10# keeps a leading zero from being read as octal.
major=$((10#${BASH_REMATCH[1]}))
minor=$((10#${BASH_REMATCH[2]}))
patch=$((10#${BASH_REMATCH[3]}))
if [[ ! "$revision" =~ ^[0-9]$ ]]; then
  echo "error: revision must be a single digit 0-9, got '$revision'" >&2
  exit 1
fi
if (( minor > 999 || patch > 99 )); then
  echo "error: $version does not fit the build number (minor <= 999, patch <= 99)" >&2
  exit 1
fi
printf '%d\n' "$((major * 1000000 + minor * 1000 + patch * 10 + 10#$revision))"
