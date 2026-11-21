cmake_minimum_required(VERSION 3.27)

# The native executor of MTPLX: the Python module mtplx._engine.
#
# It shares MLX's array type with mlx.core, so it must be built with the same
# nanobind MLX was built with (2.15.0 for mlx 0.32.2) and in MLX's type domain.
# Configure with -DPython_EXECUTABLE=<the interpreter that will load it>; a
# nanobind that is not installed in that interpreter can be given with
# -Dnanobind_ROOT=<.../nanobind/cmake>.

# MTPLX supports macOS 14; newer Metal calls are guarded at run time.  The
# target must be known before the compilers are examined, so it is set ahead of
# project().  MACOSX_DEPLOYMENT_TARGET in the environment overrides it.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "14.0"
      CACHE STRING "Oldest macOS the module loads on")
endif()

project(mtplx_engine LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MTPLX_ENGINE_ASAN "Build with AddressSanitizer (tests only)" OFF)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

if(NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# Receipts baked into the module: the loader refuses a module built against
# another MLX, because the module reads MLX's array layout directly.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import mlx.core as mx; print(mx.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MTPLX_ENGINE_MLX_VERSION)
file(STRINGS "${nanobind_ROOT}/../include/nanobind/nanobind.h" _nb_version_lines
     REGEX "#define NB_VERSION_(MAJOR|MINOR|PATCH) ")
string(REGEX MATCH "NB_VERSION_MAJOR ([0-9]+)" _ "${_nb_version_lines}")
set(_nb_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "NB_VERSION_MINOR ([0-9]+)" _ "${_nb_version_lines}")
set(_nb_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "NB_VERSION_PATCH ([0-9]+)" _ "${_nb_version_lines}")
set(_nb_patch "${CMAKE_MATCH_1}")
set(MTPLX_ENGINE_NANOBIND_VERSION "${_nb_major}.${_nb_minor}.${_nb_patch}")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" MTPLX_ENGINE_VERSION LIMIT_COUNT 1)

message(STATUS "mtplx._engine ${MTPLX_ENGINE_VERSION}: python ${Python_VERSION}, "
               "mlx ${MTPLX_ENGINE_MLX_VERSION}, nanobind ${MTPLX_ENGINE_NANOBIND_VERSION}")

nanobind_add_module(
  _engine
  NB_STATIC
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings/module.mm
  ${CMAKE_CURRENT_LIST_DIR}/engine/history_rows.cpp
  ${CMAKE_CURRENT_LIST_DIR}/engine/kernels.mm
  ${CMAKE_CURRENT_LIST_DIR}/engine/recorder.mm
  ${CMAKE_CURRENT_LIST_DIR}/engine/executor.mm
  ${CMAKE_CURRENT_LIST_DIR}/engine/limits.mm)

target_compile_options(_engine PRIVATE -fobjc-arc -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(
  _engine
  PRIVATE MTPLX_ENGINE_VERSION="${MTPLX_ENGINE_VERSION}"
          MTPLX_ENGINE_MLX_VERSION="${MTPLX_ENGINE_MLX_VERSION}"
          MTPLX_ENGINE_NANOBIND_VERSION="${MTPLX_ENGINE_NANOBIND_VERSION}")
target_link_libraries(_engine PRIVATE mlx "-framework Metal" "-framework Foundation")

if(MTPLX_ENGINE_ASAN)
  target_compile_options(_engine PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
  target_link_options(_engine PRIVATE -fsanitize=address)
endif()

# The module finds libmlx where the mlx wheel puts it, next to the installed
# package (site-packages/mtplx/_engine*.so -> site-packages/mlx/lib).  No
# absolute path into the build environment is left in the module: loaded from
# anywhere else, it binds to the libmlx that mlx.core already brought in.
set_target_properties(
  _engine
  PROPERTIES BUILD_WITH_INSTALL_RPATH ON
             INSTALL_RPATH "@loader_path/../mlx/lib"
             OBJCXX_VISIBILITY_PRESET hidden)
