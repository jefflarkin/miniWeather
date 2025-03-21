cmake_minimum_required(VERSION 3.12)
project(miniWeather-mdspan
  VERSION 0.0.1
  LANGUAGES CXX
)

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
  include(FetchContent)
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
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${mdspan_SOURCE_DIR}/include")

add_executable(serial miniWeather_mdspan.cpp)
target_link_libraries(serial INTERFACE std::mdspan)
target_compile_options(serial PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)

add_executable(test_unique_mdarray test_unique_mdarray.cpp)
target_link_libraries(test_unique_mdarray INTERFACE std::mdspan)
target_compile_options(test_unique_mdarray PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall>
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4>
)


