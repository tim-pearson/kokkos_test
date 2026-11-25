// Q1: inside a device function, is `if (X)` (X a template parameter) as good as
// `if constexpr (X)`? Tested for 3 kinds of template parameter:
//   Bool : template <bool B>
//   Mode : template <int M>, three-way chain (0 light / 1 medium / 2 heavy)
//   Type : template <class T>, condition sizeof(T) == 4
// Each case has a Cexpr_* selector (if constexpr) and an If_* selector (plain if),
// both called from the SAME kernel template, so the only difference is the keyword.
// The branch that should be dead is made register-heavy so a leftover shows up in
// REG/STACK (cuobjdump --dump-resource-usage) and in the SASS (see check_sass.sh).
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdio>
#include <type_traits>

template <class T> KOKKOS_INLINE_FUNCTION T light(T x) { return x * T(1.0001) + T(1); }

template <class T> KOKKOS_INLINE_FUNCTION T medium(T x) {
  for (int k = 0; k < 8; ++k) x = x * T(1.0001) + T(0.5);
  return x;
}

template <class T> KOKKOS_INLINE_FUNCTION T heavy(T x) {
  T a[16];
  for (int k = 0; k < 16; ++k) a[k] = x + T(k);
  for (int it = 0; it < 2; ++it)
    for (int k = 0; k < 16; ++k) a[k] = Kokkos::sin(a[k]) * a[(k + 1) & 15] + T(1e-3);
  T s = 0;
  for (int k = 0; k < 16; ++k) s += a[k];
  return s;
}

// ---- Bool ------------------------------------------------------------------
template <bool B> struct Cexpr_Bool {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if constexpr (B) return light(x); else return heavy(x);
  }
};
template <bool B> struct If_Bool {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if (B) return light(x); else return heavy(x);
  }
};

// ---- Mode (int, three-way) ---------------------------------------------------
template <int M> struct Cexpr_Mode {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if constexpr (M == 0) return light(x);
    else if constexpr (M == 1) return medium(x);
    else return heavy(x);
  }
};
template <int M> struct If_Mode {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if (M == 0) return light(x);
    else if (M == 1) return medium(x);
    else return heavy(x);
  }
};

// ---- Type (condition on sizeof(T)) -------------------------------------------
struct Cexpr_Type {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if constexpr (sizeof(T) == 4) return light(x); else return heavy(x);
  }
};
struct If_Type {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if (sizeof(T) == 4) return light(x); else return heavy(x);
  }
};

// ---- Optional: a branch that only compiles for some types -------------------
// Build with -DDEMO_COMPILE_FAIL: the plain-if version must fail to compile
// (double has no .nonexistent()), the if constexpr version is fine.
#ifdef DEMO_COMPILE_FAIL
struct Cexpr_OnlyForSome {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if constexpr (std::is_same_v<T, float>) return x.nonexistent(); else return x;
  }
};
struct If_OnlyForSome {
  template <class T> static KOKKOS_INLINE_FUNCTION T apply(T x) {
    if (std::is_same_v<T, float>) return x.nonexistent(); else return x;
  }
};
#endif

// ---- one kernel template for everything --------------------------------------
template <class Sel, class T>
void run_kernel(const Kokkos::View<T*>& v) {
  Kokkos::parallel_for("k", v.extent(0), KOKKOS_LAMBDA(int i) { v(i) = Sel::apply(v(i)); });
}

// min over `trials` batches of `reps` launches, ms per launch
template <class Sel, class T>
double bench(int N, int reps, int trials) {
  Kokkos::View<T*> v("v", N);
  Kokkos::deep_copy(v, T(1));
  run_kernel<Sel, T>(v); Kokkos::fence();  // warmup
  double best = 1e30;
  for (int t = 0; t < trials; ++t) {
    Kokkos::Timer timer;
    for (int r = 0; r < reps; ++r) run_kernel<Sel, T>(v);
    Kokkos::fence();
    best = std::min(best, timer.seconds() * 1e3 / reps);
  }
  return best;
}

template <class SelC, class SelI, class T>
void row(const char* name, int N, int reps, int trials) {
  std::printf("  %-18s if constexpr %9.4f ms | plain if %9.4f ms\n", name,
              bench<SelC, T>(N, reps, trials), bench<SelI, T>(N, reps, trials));
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 1 << 24, reps = 20, trials = 5;
    std::printf("Kokkos %d.%d.%d  N=%d reps=%d trials=%d (min over trials)\n",
                KOKKOS_VERSION / 10000, (KOKKOS_VERSION / 100) % 100,
                KOKKOS_VERSION % 100, N, reps, trials);
    row<Cexpr_Bool<true>,  If_Bool<true>,  double>("Bool true",  N, reps, trials);
    row<Cexpr_Bool<false>, If_Bool<false>, double>("Bool false", N, reps, trials);
    row<Cexpr_Mode<0>, If_Mode<0>, double>("Mode 0 (light)",  N, reps, trials);
    row<Cexpr_Mode<1>, If_Mode<1>, double>("Mode 1 (medium)", N, reps, trials);
    row<Cexpr_Mode<2>, If_Mode<2>, double>("Mode 2 (heavy)",  N, reps, trials);
    row<Cexpr_Type, If_Type, float>("Type float",  N, reps, trials);
    row<Cexpr_Type, If_Type, double>("Type double", N, reps, trials);
#ifdef DEMO_COMPILE_FAIL
    run_kernel<Cexpr_OnlyForSome, double>(Kokkos::View<double*>("w", 1));
    run_kernel<If_OnlyForSome,    double>(Kokkos::View<double*>("w", 1));
#endif
  }
  Kokkos::finalize();
}
