cmake_minimum_required(VERSION 3.27)

project(mtplx_native_qsa_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# ---------------------------------------------------------------------------
# nanobind selection, and the ABI guard that has to be a CONFIGURE-time error.
#
# This extension can only cast an ``mlx.core.array`` if it shares mlx.core's
# nanobind type registry.  Two nanobind modules share one exactly when they
# agree on the capsule key ``__nb_internals_<abi_tag>_<domain>__``
# (nanobind src/nb_internals.cpp, nb_module_exec).  The domain is NB_DOMAIN=mlx
# on both sides.  The tag is ``"v" NB_INTERNALS_VERSION ... "_"
# NB_PLATFORM_ABI_TAG`` (nanobind src/nb_abi.h), and NB_INTERNALS_VERSION moves
# between nanobind releases -- 19 in 2.12.0, 20 in 2.13.0, 21 in 2.15.0.
#
# When it does not match, the build SUCCEEDS, the module IMPORTS, and then
# every function taking an array raises
#
#   TypeError: ...(): incompatible function arguments.
#     1. ...(queries: mlx::core::array, ...)
#     Invoked with types: mlx.core.array, ...
#
# -- the C++ name in the signature being the tell that the type was never
# resolved.  That is what the first guarded run hit, and the earlier kernel had
# the same latent defect.  Failing loudly here costs one configure; not
# failing costs a whole guarded GPU window.
#
# Override with -DMTPLX_NANOBIND_DIR=<nanobind package root> (the directory
# containing cmake/ and src/), e.g. a uv tool install of a matching release.
# NEVER "fix" a mismatch by defining NB_INTERNALS_VERSION: that macro guards
# the LAYOUT of the shared nb_internals struct, so forcing the tag to match
# while the struct differs makes two modules scribble on each other.
# ---------------------------------------------------------------------------
set(MTPLX_NANOBIND_DIR "" CACHE PATH
    "nanobind package root to build against (default: the interpreter's)")

if(MTPLX_NANOBIND_DIR)
  set(nanobind_ROOT "${MTPLX_NANOBIND_DIR}/cmake")
  set(MTPLX_NANOBIND_SRC "${MTPLX_NANOBIND_DIR}")
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  get_filename_component(MTPLX_NANOBIND_SRC "${nanobind_ROOT}" DIRECTORY)
endif()
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# --- read NB_INTERNALS_VERSION out of the nanobind we are about to use ------
set(MTPLX_NB_INTERNALS "")
foreach(_hdr "src/nb_abi.h" "src/nb_internals.h")
  if(NOT MTPLX_NB_INTERNALS AND EXISTS "${MTPLX_NANOBIND_SRC}/${_hdr}")
    file(STRINGS "${MTPLX_NANOBIND_SRC}/${_hdr}" _nb_lines
         REGEX "define[ 	]+NB_INTERNALS_VERSION[ 	]+[0-9]+")
    foreach(_nb_line IN LISTS _nb_lines)
      if(NOT MTPLX_NB_INTERNALS)
        string(REGEX MATCH "NB_INTERNALS_VERSION[ 	]+([0-9]+)" _m "${_nb_line}")
        if(CMAKE_MATCH_1)
          set(MTPLX_NB_INTERNALS "${CMAKE_MATCH_1}")
        endif()
      endif()
    endforeach()
  endif()
endforeach()

# --- read the ABI tag mlx.core was actually built with ---------------------
set(MTPLX_MLX_NB_INTERNALS "")
file(GLOB MTPLX_MLX_CORE "${MLX_ROOT}/core*.so")
if(MTPLX_MLX_CORE)
  list(GET MTPLX_MLX_CORE 0 MTPLX_MLX_CORE_SO)
  file(STRINGS "${MTPLX_MLX_CORE_SO}" _mlx_tags
       REGEX "v[0-9]+[0-9a-zA-Z._-]*_[0-9a-zA-Z_]*libcpp[0-9a-zA-Z_]*")
  foreach(_tag IN LISTS _mlx_tags)
    if(NOT MTPLX_MLX_NB_INTERNALS)
      string(REGEX MATCH "v([0-9]+)[0-9a-zA-Z._-]*_[0-9a-zA-Z_]*libcpp" _m "${_tag}")
      if(CMAKE_MATCH_1)
        set(MTPLX_MLX_NB_INTERNALS "${CMAKE_MATCH_1}")
      endif()
    endif()
  endforeach()
endif()

if(MTPLX_NB_INTERNALS AND MTPLX_MLX_NB_INTERNALS)
  if(NOT MTPLX_NB_INTERNALS STREQUAL MTPLX_MLX_NB_INTERNALS)
    message(FATAL_ERROR
      "nanobind ABI mismatch: mlx.core (${MTPLX_MLX_CORE_SO}) was built with "
      "nanobind internals v${MTPLX_MLX_NB_INTERNALS}, but this build would use "
      "v${MTPLX_NB_INTERNALS} from ${MTPLX_NANOBIND_SRC}.\n"
      "The two would get separate __nb_internals_<tag>_mlx__ capsules, the "
      "extension could not resolve mlx::core::array, and EVERY call would "
      "raise TypeError at run time.\n"
      "Re-run with -DMTPLX_NANOBIND_DIR=<nanobind package root whose "
      "src/nb_abi.h says NB_INTERNALS_VERSION ${MTPLX_MLX_NB_INTERNALS}>.\n"
      "Compare src/nb_abi.h in the nanobind that built mlx.core with the "
      "one this build resolved.")
  endif()
  message(STATUS
    "nanobind internals v${MTPLX_NB_INTERNALS} matches mlx.core; "
    "the extension will share its type registry")
else()
  message(WARNING
    "could not compare nanobind ABI tags (nanobind='${MTPLX_NB_INTERNALS}', "
    "mlx.core='${MTPLX_MLX_NB_INTERNALS}'); if calls fail with "
    "'incompatible function arguments' listing mlx::core::array, compare "
    "the two nanobind ABI tags by hand")
endif()

add_library(mtplx_native_qsa_ext)
target_sources(
  mtplx_native_qsa_ext
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa.cpp
         ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa_decode.cpp)
target_include_directories(mtplx_native_qsa_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mtplx_native_qsa_ext PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mtplx_native_qsa_ext_metallib
    TITLE
    mtplx_native_qsa_ext
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    DEPS
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/steel_qsa_sparse_gqa.h
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa_params.h
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/steel_qsa_sparse_gqa_decode.h
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa_decode_params.h
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mtplx_native_qsa_ext mtplx_native_qsa_ext_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mtplx_native_qsa_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
