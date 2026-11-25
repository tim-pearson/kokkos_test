// Q2: setting a tile extent to 0 in MDRangePolicy deadlocks?
// Usage: ./tile_zero <tile0> <tile1>     e.g. ./tile_zero 0 0   or  ./tile_zero 8 8
// A watchdog thread aborts with exit code 2 if the kernel hasn't finished in 10 s.
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const int t0 = argc > 2 ? std::atoi(argv[1]) : 0;
    const int t1 = argc > 2 ? std::atoi(argv[2]) : 0;
    std::printf("tile = {%d, %d}\n", t0, t1);
    std::thread([] {
      std::this_thread::sleep_for(std::chrono::seconds(10));
      std::printf("DEADLOCK/HANG: no completion after 10 s\n");
      std::fflush(stdout);
      std::_Exit(2);
    }).detach();

    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::View<int**> a("a", 64, 64);
    Policy p({0, 0}, {64, 64}, {t0, t1});
    Kokkos::parallel_for("tile0", p, KOKKOS_LAMBDA(int i, int j) { a(i, j) = i + j; });
    Kokkos::fence();
    std::printf("OK: completed\n");
  }
  Kokkos::finalize();
  return rc;
}
