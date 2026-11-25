// Q1: inside a device function, is `if (B)` (B a template param) as fast as
// `if constexpr (B)`? Dead branch is made register-heavy so that if it is NOT
// eliminated, it shows up as register pressure / code size / time.
#include <Kokkos_Core.hpp>
#include <cstdio>

KOKKOS_INLINE_FUNCTION double light(double x) { return x * 1.0001 + 1.0; }

KOKKOS_INLINE_FUNCTION double heavy(double x) {
  double a[16];
  for (int k = 0; k < 16; ++k) a[k] = x + k;
  for (int it = 0; it < 64; ++it)
    for (int k = 0; k < 16; ++k) a[k] = Kokkos::sin(a[k]) * a[(k + 1) & 15] + 1e-3;
  double s = 0;
  for (int k = 0; k < 16; ++k) s += a[k];
  return s;
}

using V = Kokkos::View<double*>;

template <bool B>
void kernel_constexpr(const V& v) {
  Kokkos::parallel_for("constexpr", v.extent(0), KOKKOS_LAMBDA(int i) {
    double x = v(i);
    if constexpr (B) x = light(x); else x = heavy(x);
    v(i) = x;
  });
}

template <bool B>
void kernel_plainif(const V& v) {
  Kokkos::parallel_for("plainif", v.extent(0), KOKKOS_LAMBDA(int i) {
    double x = v(i);
    if (B) x = light(x); else x = heavy(x);
    v(i) = x;
  });
}

template <class F>
double time_ms(F f, const V& v, int reps) {
  Kokkos::deep_copy(v, 1.0);
  f(v); Kokkos::fence();  // warmup
  Kokkos::Timer t;
  for (int r = 0; r < reps; ++r) f(v);
  Kokkos::fence();
  return t.seconds() * 1e3 / reps;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 1 << 24, reps = 200;
    V v("v", N);
    std::printf("Kokkos %d.%d.%d, N=%d, reps=%d\n", KOKKOS_VERSION / 10000,
                (KOKKOS_VERSION / 100) % 100, KOKKOS_VERSION % 100, N, reps);
    for (int trial = 0; trial < 3; ++trial) {
      std::printf("trial %d\n", trial);
      std::printf("  B=true  if constexpr: %.4f ms\n", time_ms(kernel_constexpr<true>, v, reps));
      std::printf("  B=true  plain if    : %.4f ms\n", time_ms(kernel_plainif<true>, v, reps));
      std::printf("  B=false if constexpr: %.4f ms\n", time_ms(kernel_constexpr<false>, v, reps));
      std::printf("  B=false plain if    : %.4f ms\n", time_ms(kernel_plainif<false>, v, reps));
    }
  }
  Kokkos::finalize();
}
