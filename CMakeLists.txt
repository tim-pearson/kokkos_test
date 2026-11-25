cmake_minimum_required(VERSION 3.25)
project(kokkos_tests CXX)

# Change the version here or on the command line: -DKOKKOS_VERSION=5.1.1
# Tag name must match a tag in github.com/kokkos/kokkos (e.g. 5.1.1 or 5.1.01).
set(KOKKOS_VERSION "5.1.1" CACHE STRING "Kokkos git tag/branch to fetch")
# Optional: use a local checkout instead of fetching, -DKOKKOS_SRC_DIR=/path/to/kokkos
set(KOKKOS_SRC_DIR "" CACHE PATH "Local Kokkos source (overrides KOKKOS_VERSION)")

# CUDA (benchmarks on the GPU) or SERIAL (dev machine: LSP / compile_commands.json)
set(KOKKOS_BACKEND "CUDA" CACHE STRING "Kokkos backend: CUDA or SERIAL")
set_property(CACHE KOKKOS_BACKEND PROPERTY STRINGS CUDA SERIAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(KOKKOS_BACKEND STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA   ON  CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON  CACHE BOOL "" FORCE)
  set(Kokkos_ARCH_ADA89    ON  CACHE BOOL "" FORCE)   # RTX 4070 = sm_89
elseif(KOKKOS_BACKEND STREQUAL "SERIAL")
  set(Kokkos_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON  CACHE BOOL "" FORCE)
else()
  message(FATAL_ERROR "KOKKOS_BACKEND must be CUDA or SERIAL")
endif()
message(STATUS "KOKKOS_BACKEND = ${KOKKOS_BACKEND}")

if(KOKKOS_SRC_DIR)
  add_subdirectory(${KOKKOS_SRC_DIR} kokkos_build)
else()
  include(FetchContent)
  FetchContent_Declare(kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        ${KOKKOS_VERSION}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(kokkos)
endif()

# One executable per question: add a new file + one line here.
foreach(t bench_branch tile_zero)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE Kokkos::kokkos)
endforeach()
