cmake_minimum_required(VERSION 3.18)

project(oif-toy-example LANGUAGES C CXX)

# Enforce using `-std=c11`, without any extensions like `gnu11`.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Disable optimizations for Debug build type")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
endif()

# Incorporate additions by X/Open 7, that includes POSIX 17 additions and allow
# to use things like the `M_PI` constant in the code without warnings from
# static analyzers.
add_compile_definitions(_XOPEN_SOURCE=700)

# DOWNLOAD_EXTRACT_TIMESTAMP = TRUE
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Copy all built library targets to a common directory, so that it is easy to
# find and load them.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add additional cmake module to find packages.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

add_subdirectory(vendor)

add_subdirectory(oif)
add_subdirectory(oif_impl)

add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)
