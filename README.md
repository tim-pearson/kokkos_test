# Kokkos mini tests

    cmake -B build -DKOKKOS_VERSION=5.1.1      # change version here
    cmake --build build -j
    ./build/bench_branch                       # Q1
    ./build/tile_zero 0 0; ./build/tile_zero 8 8   # Q2

Local Kokkos checkout instead of fetching: `-DKOKKOS_SRC_DIR=/path/to/kokkos`.
Each version needs its own build dir (e.g. `-B build-5.1.1`).

## Q1: also compare the generated code (timing alone can hide differences)
    cuobjdump --dump-resource-usage build/bench_branch     # registers per kernel
    cuobjdump -sass build/bench_branch | grep -c MUFU      # leftover sin in the B=true kernels?
