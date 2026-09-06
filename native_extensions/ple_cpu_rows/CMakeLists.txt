cmake_minimum_required(VERSION 3.27)

project(mtplx_native_ple_cpu_rows_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# CPU-stream PLE row staging primitive for the cached async PLE lane
# (mtplx/ple_cached_aux.py). The module links the active MLX wheel and must be
# built against the exact serving nanobind/mlx ABI; scripts/fable/setup_over100_venv.sh
# builds it the same way it builds native_extensions/qsa_sparse_gqa. Every
# translation unit lives in this directory -- there is no staging step and no
# absolute compiled-in path.

find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

set(MTPLX_NANOBIND_DIR "" CACHE PATH
    "nanobind package root to build against (default: interpreter's)")
if(MTPLX_NANOBIND_DIR)
  set(nanobind_ROOT "${MTPLX_NANOBIND_DIR}/cmake")
  set(MTPLX_NANOBIND_SRC "${MTPLX_NANOBIND_DIR}")
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE MTPLX_NANOBIND_RESULT)
  if(NOT MTPLX_NANOBIND_RESULT EQUAL 0)
    message(FATAL_ERROR "the active interpreter cannot locate nanobind")
  endif()
  get_filename_component(MTPLX_NANOBIND_SRC "${nanobind_ROOT}" DIRECTORY)
endif()
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MTPLX_MLX_PACKAGE_ROOT
  RESULT_VARIABLE MTPLX_MLX_RESULT)
if(NOT MTPLX_MLX_RESULT EQUAL 0)
  message(FATAL_ERROR "the active interpreter cannot locate MLX headers")
endif()
set(MLX_DIR "${MTPLX_MLX_PACKAGE_ROOT}/share/cmake/MLX")
find_package(MLX CONFIG REQUIRED PATHS "${MLX_DIR}" NO_DEFAULT_PATH)

# nanobind's type registry is part of the MLX Python ABI.  Refuse a build when
# the selected nanobind internals version differs from the loaded mlx.core
# binary; otherwise the extension can build but cannot cast mlx arrays.
set(MTPLX_NB_INTERNALS "")
foreach(_hdr "src/nb_abi.h" "src/nb_internals.h")
  if(NOT MTPLX_NB_INTERNALS AND EXISTS "${MTPLX_NANOBIND_SRC}/${_hdr}")
    file(STRINGS "${MTPLX_NANOBIND_SRC}/${_hdr}" _nb_lines
         REGEX "define[ \\t]+NB_INTERNALS_VERSION[ \\t]+[0-9]+")
    foreach(_nb_line IN LISTS _nb_lines)
      if(NOT MTPLX_NB_INTERNALS)
        string(REGEX MATCH "NB_INTERNALS_VERSION[ \\t]+([0-9]+)" _m
               "${_nb_line}")
        if(CMAKE_MATCH_1)
          set(MTPLX_NB_INTERNALS "${CMAKE_MATCH_1}")
        endif()
      endif()
    endforeach()
  endif()
endforeach()
if(NOT MTPLX_NB_INTERNALS)
  message(FATAL_ERROR
    "could not read NB_INTERNALS_VERSION from selected nanobind source; "
    "refusing to build an unchecked MLX extension")
endif()

set(MTPLX_MLX_NB_INTERNALS "")
file(GLOB MTPLX_MLX_CORE
     "${MTPLX_MLX_PACKAGE_ROOT}/core*.so"
     "${MTPLX_MLX_PACKAGE_ROOT}/core*.dylib")
if(NOT MTPLX_MLX_CORE)
  message(FATAL_ERROR
    "the active MLX package root has no core*.so/core*.dylib; refusing to "
    "skip the nanobind ABI check")
endif()
list(GET MTPLX_MLX_CORE 0 MTPLX_MLX_CORE_SO)
file(STRINGS "${MTPLX_MLX_CORE_SO}" _mlx_tags
     REGEX "v[0-9]+[0-9a-zA-Z._-]*_[0-9a-zA-Z_]*libcpp[0-9a-zA-Z_]*")
foreach(_tag IN LISTS _mlx_tags)
  if(NOT MTPLX_MLX_NB_INTERNALS)
    string(REGEX MATCH "v([0-9]+)[0-9a-zA-Z._-]*_[0-9a-zA-Z_]*libcpp"
           _m "${_tag}")
    if(CMAKE_MATCH_1)
      set(MTPLX_MLX_NB_INTERNALS "${CMAKE_MATCH_1}")
    endif()
  endif()
endforeach()
if(NOT MTPLX_MLX_NB_INTERNALS)
  message(FATAL_ERROR
    "could not read nanobind internals ABI from the active mlx.core; "
    "refusing to build an unchecked extension")
endif()
if(MTPLX_NB_INTERNALS AND MTPLX_MLX_NB_INTERNALS AND
   NOT MTPLX_NB_INTERNALS STREQUAL MTPLX_MLX_NB_INTERNALS)
  message(FATAL_ERROR
    "nanobind ABI mismatch: extension v${MTPLX_NB_INTERNALS}, "
    "mlx.core v${MTPLX_MLX_NB_INTERNALS}")
endif()

add_library(mtplx_native_ple_cpu_rows)
target_sources(
  mtplx_native_ple_cpu_rows
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/cached_sidecar_primitive.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cached_sidecar_producer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ple_cpu_rows.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sidecar_primitive.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sidecar_producer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/host_provider.cpp)
target_include_directories(
  mtplx_native_ple_cpu_rows
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mtplx_native_ple_cpu_rows PUBLIC mlx)

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mtplx_native_ple_cpu_rows)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
