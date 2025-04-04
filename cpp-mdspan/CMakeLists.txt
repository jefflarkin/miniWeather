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

message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")

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

if (DEFINED CUB_INCLUDE_DIR)
  if (EXISTS "${CUB_INCLUDE_DIR}/cub/cub.cuh")
    message(STATUS "User-defined CUB_INCLUDE_DIR: ${CUB_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "CUB_INCLUDE_DIR=\"{CUB_INCLUDE_DIR}\" is \
 defined but \"${CUB_INCLUDE_DIR}/cub/cub.cuh\" does not exist.")
  endif()
else()
  if (CUDAToolkit_FOUND)
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
message(STATUS "Kokkos_SOURCE_DIR: ${Kokkos_SOURCE_DIR}")
message(STATUS "Kokkos_BINARY_DIR: ${Kokkos_BINARY_DIR}")
#message(STATUS "Kokkos_INCLUDE_DIRS: ${Kokkos_INCLUDE_DIRS}") # this is not defined; why?
message(STATUS "kokkos_SOURCE_DIR: ${kokkos_SOURCE_DIR}")
message(STATUS "kokkos_BINARY_DIR: ${kokkos_BINARY_DIR}")
message(STATUS "kokkos_INCLUDE_DIRS: ${kokkos_INCLUDE_DIRS}")

# Use the mdspan version that Kokkos installed, if Kokkos was indeed installed.
# Note that this is not a complete mdspan source tree; it just includes the headers.

#if ((DEFINED Kokkos_SOURCE_DIR) AND (EXISTS "${Kokkos_SOURCE_DIR}"))
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
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)

add_executable(miniWeather_serial miniWeather_mdspan.cpp miniWeather_common.cpp)
target_include_directories(miniWeather_serial PRIVATE "${PROJECT_SOURCE_DIR}")

# If building with Kokkos, we've already added the mdspan headers
# to the include path (see above).
if (NOT kokkos_POPULATED)
  target_link_libraries(miniWeather_serial PRIVATE std::mdspan)
endif()
target_compile_options(miniWeather_serial PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)

if (kokkos_POPULATED)
  add_executable(miniWeather_kokkos miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_kokkos PRIVATE "${PROJECT_SOURCE_DIR}")

  target_link_libraries(miniWeather_kokkos PRIVATE Kokkos::kokkos)
  # We got mdspan from Kokkos.
  target_compile_options(miniWeather_kokkos PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()

if (OpenACC_FOUND)
  add_executable(miniWeather_openacc miniWeather_mdspan.cpp miniWeather_common.cpp)
  target_include_directories(miniWeather_openacc PRIVATE "${PROJECT_SOURCE_DIR}")

  # If building with Kokkos, we've already added the mdspan headers
  # to the include path (see above).
  if (NOT kokkos_POPULATED)
    target_link_libraries(miniWeather_openacc PRIVATE std::mdspan)
  endif()
  target_link_libraries(miniWeather_openacc PRIVATE OpenACC::OpenACC_CXX)
  target_compile_options(miniWeather_openacc PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()



