#!/bin/zsh
# Build mtplx._engine for one Python, test it, then put it where the package
# looks for it.  The module in native/build/<SOABI>/ is replaced only after the
# staged build passed the self-test, and by a rename, so there is always a
# working module on disk.
#
#   native/build.sh <python> [--no-swap] [--asan] [--no-selftest] [--stage <dir>]
#
# --no-swap      build and test only; the module stays in the stage directory
# --no-selftest  do not run the self-test (a machine without an Apple GPU);
#                implies --no-swap
# --stage <dir>  build there instead of native/build-stage (release builds,
#                which must leave the source tree alone)
#
# MTPLX_NANOBIND_CMAKE_DIR  nanobind's cmake directory, when the interpreter
#                           does not have nanobind 2.15.0 installed
# MTPLX_ENGINE_HOLD_FILE    while this file exists nothing is built or tested
# MACOSX_DEPLOYMENT_TARGET  the oldest macOS the module loads on (default 14.0)
set -u
here=${0:A:h}
usage="usage: build.sh <python> [--no-swap] [--asan] [--no-selftest] [--stage <dir>]"
python=${1:?$usage}
shift
swap=1
asan=OFF
selftest=1
stage_root=$here/build-stage
while (( $# )); do
  case $1 in
    --no-swap) swap=0 ;;
    --asan) asan=ON; swap=0 ;;
    --no-selftest) selftest=0; swap=0 ;;
    --stage) shift; stage_root=${1:?$usage} ;;
    *) echo "unknown option $1" >&2; echo "$usage" >&2; exit 2 ;;
  esac
  shift
done

hold=${MTPLX_ENGINE_HOLD_FILE:-}
if [[ -n $hold && -e $hold ]]; then
  echo "HOLD: $hold exists, nothing built" >&2
  exit 75
fi

tag=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))') || exit 1
suffix=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') || exit 1
stage=$stage_root/$tag
[[ $asan == ON ]] && stage=$stage_root/$tag-asan
mkdir -p "$stage/out" || exit 1
stage=${stage:A}

# A cmake installed beside the interpreter (a build venv) wins over the one on
# PATH, so a release build uses the tools its environment pinned.
cmake=${python:h}/cmake
[[ -x $cmake ]] || cmake=cmake
if ! command -v "$cmake" >/dev/null; then
  echo "cmake not found (pip install 'cmake>=3.27' or brew install cmake)" >&2
  exit 1
fi

cmake_args=(
  -S "$here" -B "$stage/cmake"
  -DCMAKE_BUILD_TYPE=Release
  "-DPython_EXECUTABLE=$python"
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$stage/out"
  "-DMTPLX_ENGINE_ASAN=$asan"
)
if [[ -n ${MTPLX_NANOBIND_CMAKE_DIR:-} ]]; then
  cmake_args+=("-Dnanobind_ROOT=$MTPLX_NANOBIND_CMAKE_DIR")
fi
if [[ -x ${python:h}/ninja ]] || command -v ninja >/dev/null; then
  cmake_args+=(-G Ninja)
  [[ -x ${python:h}/ninja ]] && cmake_args+=("-DCMAKE_MAKE_PROGRAM=${python:h}/ninja")
fi
if command -v ccache >/dev/null; then
  cmake_args+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_OBJCXX_COMPILER_LAUNCHER=ccache)
fi

"$cmake" "${cmake_args[@]}" > "$stage/configure.log" 2>&1
rc=$?
if (( rc != 0 )); then
  tail -30 "$stage/configure.log" >&2
  echo "configure failed ($rc)" >&2
  exit $rc
fi
"$cmake" --build "$stage/cmake" --parallel > "$stage/build.log" 2>&1
rc=$?
if (( rc != 0 )); then
  grep -E "error|Error" -A6 "$stage/build.log" | head -80 >&2
  echo "build failed ($rc)" >&2
  exit $rc
fi
grep -E "warning:" "$stage/build.log" | sort | uniq -c | head -20

module=$stage/out/_engine$suffix
if [[ ! -f $module ]]; then
  echo "no module at $module" >&2
  exit 1
fi
echo "built $module"
[[ $asan == ON ]] && exit 0
if (( ! selftest )); then
  echo "self-test skipped on request: not swapped"
  exit 0
fi

if [[ -n $hold && -e $hold ]]; then
  echo "HOLD: $hold appeared, built but not tested or swapped" >&2
  exit 75
fi
MTPLX_NATIVE_ENGINE_MODULE=$module PYTHONPATH=${here:h} "$python" -m mtplx.engine.selftest > "$stage/selftest.log" 2>&1
rc=$?
tail -12 "$stage/selftest.log"
if (( rc != 0 )); then
  echo "self-test failed ($rc): not swapped" >&2
  exit $rc
fi

if (( swap )); then
  dest=$here/build/$tag
  mkdir -p "$dest"
  cp "$module" "$dest/.incoming$suffix" && mv -f "$dest/.incoming$suffix" "$dest/_engine$suffix" || exit 1
  cat "$here/VERSION" > "$dest/VERSION"
  echo "swapped into $dest/_engine$suffix (version $(cat "$here/VERSION"))"
fi
