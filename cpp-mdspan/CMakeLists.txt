# CMake 3.17 adds FindCUDAToolkit.
#
# Kokkos's build system claims that before CMake 3.20,
# NVHPC was identified as PGI.
cmake_minimum_required(VERSION 3.20)
project(miniWeather-mdspan
  VERSION 0.0.1
  LANGUAGES CXX
)
include(FetchContent)

message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")

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

find_package(mdspan QUIET)
if (NOT mdspan_FOUND)
  message(STATUS "No installed mdspan found, fetching from Github")
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        stable
  )
  FetchContent_GetProperties(mdspan)
  if(NOT mdspan_POPULATED)
    FetchContent_Populate(mdspan)
    add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

# Add mdspan include directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${mdspan_SOURCE_DIR}/include" "${CUB_INCLUDE_DIR}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  include(FindOpenACC)
  find_package(OpenACC REQUIRED)
  if (OpenACC_FOUND)
    message(STATUS "Found OpenACC: ${OpenACC_VERSION}")
    message(STATUS "OpenACC_CXX_FLAGS: ${OpenACC_CXX_FLAGS}")
    message(STATUS "OpenACC_CXX_OPTIONS: ${OpenACC_CXX_OPTIONS}")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_OPTIONS}")
  else()
    message(FATAL_ERROR "OpenACC not found")
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
find_package(Kokkos CONFIG)
if (Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
  message(STATUS "No installed Kokkos found, fetching from Github")
  include(FetchContent)
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        4.6.00
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

add_executable(serial miniWeather_mdspan.cpp)
target_link_libraries(serial INTERFACE std::mdspan)
target_compile_options(serial PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)

if (Kokkos_FOUND)
  add_executable(serial_kokkos miniWeather_mdspan.cpp)
  target_link_libraries(serial_kokkos INTERFACE std::mdspan)
  target_link_libraries(serial_kokkos PRIVATE Kokkos::kokkos)
  target_compile_options(serial_kokkos PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()

if (OpenACC_FOUND)
  add_executable(serial_openacc miniWeather_mdspan.cpp)
  target_link_libraries(serial_openacc INTERFACE std::mdspan)
  target_link_libraries(serial_openacc PRIVATE OpenACC::OpenACC_CXX)
  target_compile_options(serial_openacc PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      -Wall>
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4>
  )
endif()

add_executable(test_unique_mdarray test_unique_mdarray.cpp)
target_link_libraries(test_unique_mdarray INTERFACE std::mdspan)
target_compile_options(test_unique_mdarray PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)


