# CMake 3.17 adds FindCUDAToolkit.
#
# Kokkos's build system claims that before CMake 3.20,
# NVHPC was identified as PGI.
#
# FetchContent_MakeAvailable gets new features in CMake 3.24.
cmake_minimum_required(VERSION 3.24)
project(miniWeather-mdspan
  VERSION 0.0.1
  LANGUAGES CXX
)
include(FetchContent)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

# Option to override which version of the C++ Standard to use
set(MINIWEATHER_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD")

# Decide which version of the C++ Standard version to use
if(MINIWEATHER_CXX_STANDARD STREQUAL "17")
  if("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++17 standard")
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(FATAL_ERROR "Requested MINIWEATHER_CXX_STANDARD \"17\" not supported by provided C++ compiler")
  endif()
elseif(MINIWEATHER_CXX_STANDARD STREQUAL "20")
  if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
  else()
    message(FATAL_ERROR "Requested MINIWEATHER_CXX_STANDARD \"20\" not supported by provided C++ compiler")
  endif()
elseif(MINIWEATHER_CXX_STANDARD STREQUAL "23")
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++23 standard")
    set(CMAKE_CXX_STANDARD 23)
  else()
    message(FATAL_ERROR "Requested MINIWEATHER_CXX_STANDARD \"23\" not supported by provided C++ compiler")
  endif()
else()
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 23)
    message(STATUS "Detected support for C++23 standard")
  elseif("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 20)
    message(STATUS "Detected support for C++20 standard")
  elseif("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Detected support for C++17 standard")
  else()
    message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++17 or newer.")
  endif()
endif()

include(FindOpenACC)
find_package(OpenACC)
if (OpenACC_FOUND)
  message(STATUS "Found OpenACC: ${OpenACC_VERSION}")
  message(STATUS "OpenACC_CXX_FLAGS: ${OpenACC_CXX_FLAGS}")
  message(STATUS "OpenACC_CXX_OPTIONS: ${OpenACC_CXX_OPTIONS}")
else()
  message(STATUS "OpenACC not found")
endif()

# Find CUDA Toolkit; it's not required,
# but other things (like CUB) depend on it.
find_package(CUDAToolkit)

set(MINIWEATHER_CUB_FOUND FALSE)

if (CUDAToolkit_FOUND AND (NOT (DEFINED CUB_INCLUDE_DIR)))
  # Check whether CUDA Toolkit's include directory has CUB in it.
  find_path(CUB_INCLUDE_DIR
    NAMES cub/cub.cuh
    PATHS
      ${CUDAToolkit_INCLUDE_DIRS}
      /usr/include
      /usr/local/include
    DOC "Path to CUB include directory."
  )
  message(STATUS "Discovered CUB_INCLUDE_DIR: ${CUB_INCLUDE_DIR}")
  if (CUDAToolkit_VERSION VERSION_LESS "12.8.0")
    message(STATUS "CUDAToolkit_VERSION 12.6.85 is known to lack ForEachInExtents.  \
Your CUDAToolkit_VERSION is ${CUDAToolkit_VERSION}.")
  endif()
endif()
 
if (DEFINED CUB_INCLUDE_DIR)
  set(CUB_TEST_FILE_0 "${CUB_INCLUDE_DIR}/cub/cub.cuh")
  set(CUB_TEST_FILE_0_FOUND FALSE)
  # This was added to later versions of CUB.
  set(CUB_TEST_FILE_1 "${CUB_INCLUDE_DIR}/cub/device/device_for_each.cuh")
  set(CUB_TEST_FILE_1_FOUND FALSE)

  if (EXISTS "${CUB_TEST_FILE_0}")
    message(STATUS "User-defined CUB_INCLUDE_DIR ${CUB_INCLUDE_DIR} contains cub/cub.cuh")
  else()
    message(STATUS "User-defined CUB_INCLUDE_DIR ${CUB_INCLUDE_DIR} does NOT contain cub/cub.cuh")
  endif()

  if (EXISTS "${CUB_TEST_FILE_1}")
    message(STATUS "User-defined CUB_INCLUDE_DIR ${CUB_INCLUDE_DIR} contains cub/device/device_for_each.cuh")
  else()
    message(STATUS "User-defined CUB_INCLUDE_DIR ${CUB_INCLUDE_DIR} does NOT contain cub/device/device_for_each.cuh")
  endif()

  set(MINIWEATHER_CUB_FOUND (${CUB_TEST_FILE_0_FOUND} AND ${CUB_TEST_FILE_1_FOUND}))
endif()

# Please see Kokkos' CMake instructions:
#
# https://kokkos.org/kokkos-core-wiki/get-started/integrating-kokkos-into-your-cmake-project.html
#
# If you want to use a previously installed Kokkos,
# then set the CMake variable Kokkos_ROOT to the path
# of your Kokkos installation.  Otherwise, CMake will
# attempt to download Kokkos for you.
#
# Users can set FETCHCONTENT_SOURCE_DIR_KOKKOS
# to a local path to Kokkos, to override automatic downloading.

set(Kokkos_ENABLE_SERIAL ON CACHE INTERNAL "")
set(Kokkos_ENABLE_OPENACC ON CACHE INTERNAL "")

FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        4.6.00
)
FetchContent_MakeAvailable(Kokkos)

# The above defines kokkos_POPULATED (lower-case name), but NOT Kokkos_POPULATED.
if (NOT kokkos_POPULATED)
  message(FATAL_ERROR "Kokkos was not found")
endif()

# Use the mdspan version that Kokkos installed, if Kokkos was indeed installed.
# Note that this is not a complete mdspan source tree; it just includes the headers.

if (kokkos_POPULATED)
  set(MINIWEATHER_MDSPAN_INCLUDE "${Kokkos_SOURCE_DIR}/tpls/mdspan/include")
  message(STATUS "Using Kokkos' mdspan headers: ${MINIWEATHER_MDSPAN_INCLUDE}")
  # Don't use target_link_libraries in this case.
  include_directories("${MINIWEATHER_MDSPAN_INCLUDE}")
else()
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        stable
    FIND_PACKAGE_ARGS NAMES mdspan
  )
  FetchContent_MakeAvailable(mdspan)

  # Add mdspan include directory
  message(STATUS "mdspan_SOURCE_DIR: ${mdspan_SOURCE_DIR}")
  message(STATUS "mdspan_BINARY_DIR: ${mdspan_BINARY_DIR}")

  set(MINIWEATHER_MDSPAN_INCLUDE "${mdspan_SOURCE_DIR}/include")
  message(STATUS "Using FetchContent's mdspan headers: ${MINIWEATHER_MDSPAN_INCLUDE}")
  # Don't use include_directories here; target_link_libraries
  # (see below) will get mdspan's include directory just fine.
endif()

add_executable(test_unique_mdarray test_unique_mdarray.cpp)
# If building with Kokkos, we've already added the mdspan headers
# to the include path (see above).
if (NOT kokkos_POPULATED)
  target_link_libraries(test_unique_mdarray PRIVATE std::mdspan)
endif()
target_compile_options(test_unique_mdarray PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)

message(STATUS "Enabled miniWeather_serial")
add_executable(miniWeather_serial miniWeather_mdspan.cpp miniWeather_common.cpp)
target_include_directories(miniWeather_serial PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(miniWeather_serial PRIVATE MINIWEATHER_SERIAL)

# If building with Kokkos, we've already added the mdspan headers
# to the include path (see above).
if (NOT kokkos_POPULATED)
  target_link_libraries(miniWeather_serial PRIVATE std::mdspan)
endif()
target_compile_options(miniWeather_serial PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)

if (kokkos_POPULATED)
  message(STATUS "Enabled miniWeather_kokkos_serial")
  add_executable(miniWeather_kokkos_serial miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_kokkos_serial PRIVATE "${PROJECT_SOURCE_DIR}")

  target_link_libraries(miniWeather_kokkos_serial PRIVATE Kokkos::kokkos)
  target_compile_definitions(miniWeather_kokkos_serial PRIVATE MINIWEATHER_KOKKOS)
  target_compile_definitions(miniWeather_kokkos_serial PRIVATE MINIWEATHER_KOKKOS_SERIAL)

  # We got mdspan from Kokkos.
  target_compile_options(miniWeather_kokkos_serial PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()

if (kokkos_POPULATED AND OpenACC_FOUND)
  set(Kokkos_ENABLE_OPENACC ON)

  message(STATUS "Enabled miniWeather_kokkos_openacc")
  add_executable(miniWeather_kokkos_openacc miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_kokkos_openacc PRIVATE "${PROJECT_SOURCE_DIR}")

  target_link_libraries(miniWeather_kokkos_openacc PRIVATE Kokkos::kokkos)
  target_compile_definitions(miniWeather_kokkos_openacc PRIVATE MINIWEATHER_KOKKOS)
  target_compile_definitions(miniWeather_kokkos_openacc PRIVATE MINIWEATHER_KOKKOS_OPENACC)

  # We got mdspan from Kokkos.
  target_link_libraries(miniWeather_kokkos_openacc PRIVATE OpenACC::OpenACC_CXX)
  target_compile_options(miniWeather_kokkos_openacc PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()

# We don't have a "pure OpenACC" version yet.
if (FALSE)
if (OpenACC_FOUND)
  message(STATUS "Enabled miniWeather_openacc")
  add_executable(miniWeather_openacc miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_openacc PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(miniWeather_openacc PRIVATE MINIWEATHER_OPENACC)

  # If building with Kokkos, we've already added the mdspan headers
  # to the include path (see above).
  if (NOT kokkos_POPULATED)
    target_link_libraries(miniWeather_openacc PRIVATE std::mdspan)
  endif()
  target_link_libraries(miniWeather_openacc PRIVATE OpenACC::OpenACC_CXX)
  target_compile_options(miniWeather_openacc PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()
endif()

set(MINIWEATHER_HAVE_STDPAR (${CMAKE_CXX_STANDARD} GREATER_EQUAL 23))
# Option to enable stdpar
set(MINIWEATHER_ENABLE_STDPAR ${MINIWEATHER_HAVE_STDPAR} CACHE BOOL "Enable stdpar using the -stdpar flag")

if (MINIWEATHER_ENABLE_STDPAR)
  message(STATUS "Enabled miniWeather_stdpar_cpu")
  add_executable(miniWeather_stdpar_cpu miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_stdpar_cpu PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(miniWeather_stdpar_cpu PRIVATE MINIWEATHER_STDPAR)
  target_compile_definitions(miniWeather_stdpar_cpu PRIVATE MINIWEATHER_STDPAR_CPU)

  # If building with Kokkos, we've already added the mdspan headers
  # to the include path (see above).
  if (NOT kokkos_POPULATED)
    target_link_libraries(miniWeather_stdpar_cpu PRIVATE std::mdspan)
  endif()
  target_compile_options(miniWeather_stdpar_cpu PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
  target_compile_options(miniWeather_stdpar_cpu PRIVATE "-stdpar=multicore")
  target_link_options(miniWeather_stdpar_cpu PRIVATE "-stdpar=multicore")
endif()

if (MINIWEATHER_ENABLE_STDPAR)
  message(STATUS "Enabled miniWeather_stdpar_gpu")
  add_executable(miniWeather_stdpar_gpu miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_stdpar_gpu PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(miniWeather_stdpar_gpu PRIVATE MINIWEATHER_STDPAR)
  target_compile_definitions(miniWeather_stdpar_gpu PRIVATE MINIWEATHER_STDPAR_GPU)

  # If building with Kokkos, we've already added the mdspan headers
  # to the include path (see above).
  if (NOT kokkos_POPULATED)
    target_link_libraries(miniWeather_stdpar_gpu PRIVATE std::mdspan)
  endif()
  target_compile_options(miniWeather_stdpar_gpu PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
  target_compile_options(miniWeather_stdpar_gpu PRIVATE "-stdpar=gpu")
  target_link_options(miniWeather_stdpar_gpu PRIVATE "-stdpar=gpu")
endif()

if (MINIWEATHER_ENABLE_STDPAR AND OpenACC_FOUND)
  message(STATUS "Enabled miniWeather_stdpar_openacc")
  add_executable(miniWeather_stdpar_openacc miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_stdpar_openacc PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(miniWeather_stdpar_openacc PRIVATE MINIWEATHER_STDPAR)
  target_compile_definitions(miniWeather_stdpar_openacc PRIVATE MINIWEATHER_STDPAR_OPENACC)

  # If building with Kokkos, we've already added the mdspan headers
  # to the include path (see above).
  if (NOT kokkos_POPULATED)
    target_link_libraries(miniWeather_stdpar_openacc PRIVATE std::mdspan)
  endif()
  target_link_libraries(miniWeather_stdpar_openacc PRIVATE OpenACC::OpenACC_CXX)
  target_compile_options(miniWeather_stdpar_openacc PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
  target_compile_options(miniWeather_stdpar_openacc PRIVATE "-stdpar=gpu:acc")
  target_link_options(miniWeather_stdpar_openacc PRIVATE "-stdpar=gpu:acc")
endif()

# Option to enable CUB
set(MINIWEATHER_ENABLE_CUB ${MINIWEATHER_CUB_FOUND} CACHE BOOL "Enable CUB")

if (MINIWEATHER_ENABLE_CUB)
  message(STATUS "Enabled miniWeather_cub")
  add_executable(miniWeather_cub miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_cub PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(miniWeather_cub PRIVATE MINIWEATHER_CUB)

  # If building with Kokkos, we've already added the mdspan headers
  # to the include path (see above).
  if (NOT kokkos_POPULATED)
    target_link_libraries(miniWeather_cub PRIVATE std::mdspan)
  endif()
  target_compile_options(miniWeather_cub PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
  # Building CUB with nvc++ requires the "-cuda" flag
  target_compile_options(miniWeather_cub PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>>:
      -cuda>
  )
  target_link_options(miniWeather_cub PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:NVHPC>>:
      -cuda>
  )
endif()
