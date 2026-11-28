// ZGEMM emulation with the Ozaki-II scheme and the 2M algorithm (SYCL).
//
// See ../blas-zgemm-cuda/main.cu and ../blas-zgemm-cuda/ozaki2.h.  The
// benchmark measures C = A * B for complex double matrices with
//
//   1. native ZGEMM   - oneMKL (oneMath) gemm
//   2. Ozaki-II 4M    - an Ozaki-II implementation built on INT8 GEMMs, using
//      Ozaki-II 2M      four or two real products per modulus
//
// The INT8 GEMMs use gemm_batch with INT8 inputs and INT32 outputs: oneMKL on
// Intel GPUs, and oneMath on NVIDIA (cuBLAS) and AMD (rocBLAS) GPUs.  The
// oneMath cuBLAS and rocBLAS backends support this combination with
//   https://github.com/uxlfoundation/oneMath/pull/803 (cuBLAS)
//   https://github.com/uxlfoundation/oneMath/pull/804 (rocBLAS)
//
// Accuracy is checked against native ZGEMM (and a host reference for small
// sizes), or, on devices without a native ZGEMM, against a host reference on
// sampled columns.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <complex>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>
#ifdef USE_ONEMATH
#include <oneapi/math.hpp>
namespace onelib = oneapi::math;
#else
#include <oneapi/mkl.hpp>
namespace onelib = oneapi::mkl;
#endif
#include "../blas-zgemm-cuda/ozaki2.h"

//============================================================================
// INT8 GEMM: C[b] (M x N, column-major, ldc) = A[b]^T B[b] with A[b] (K x M)
// and B[b] (K x N) both k-contiguous (lda = ldb = K).  M and N are multiples
// of OZ_MN_ALIGN, K is a multiple of OZ_K_ALIGN.
//============================================================================
static void int8_gemm_tn(sycl::queue &q, int M, int N, int K,
                         const int8_t *A, size_t sA, const int8_t *B, size_t sB,
                         int32_t *C, int ldc, size_t sC, int batch) {
  onelib::blas::column_major::gemm_batch(
      q, onelib::transpose::trans, onelib::transpose::nontrans,
      M, N, K, 1.0f, A, K, sA, B, K, sB, 0.0f, C, ldc, sC, batch);
}

//============================================================================
// Ozaki-II kernels
//============================================================================

// Largest magnitudes (oz_abs_hi) of the rows of A (m x k, column-major), from
// which the scale exponents follow (oz_scale_exponent).  A work-group covers
// 64 rows (one contiguous 1 KB segment per 64 work-items and column) and a
// slice of `kslice` columns, so that the grid can fill the device for any
// shape; the slices combine through atomics on the zero-initialized amax.
static void row_amax(sycl::queue &q, const sycl::double2 *A, int m, int k,
                     int kslice, unsigned *amax) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<unsigned, 1> smax(sycl::range<1>(256), h);
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(k + kslice - 1) / kslice,
                                                    (size_t)(m + 63) / 64 * 256),
                                     sycl::range<2>(1, 256)),
                   [=](sycl::nd_item<2> it) {
      const int t = it.get_local_id(1), x = t & 63, y = t >> 6;
      const int r = it.get_group(1) * 64 + x;
      const int kb = it.get_group(0) * kslice;
      const int ke = sycl::min(kb + kslice, k);
      unsigned v = 0;
      if (r < m)
#pragma unroll 4
        for (int kk = kb + y; kk < ke; kk += 4) {
          const sycl::double2 a = A[(size_t)kk * m + r];
          v = sycl::max(v, sycl::max(oz_abs_hi(a.x()), oz_abs_hi(a.y())));
        }
      smax[y * 64 + x] = v;
      sycl::group_barrier(it.get_group());
      if (y == 0 && r < m) {
        for (int i = 1; i < 4; i++) v = sycl::max(v, smax[i * 64 + x]);
        if (v)
          sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(amax[r]).fetch_max(v);
      }
    });
  });
}

// Largest magnitudes of the columns of B (k x n, column-major): one
// work-group per column.
static void col_amax(sycl::queue &q, const sycl::double2 *B, int n, int k,
                     unsigned *amax) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n * 256), sycl::range<1>(256)),
                 [=](sycl::nd_item<1> it) {
    const size_t c = it.get_group(0);
    const sycl::double2 *col = B + c * k;
    unsigned v = 0;
#pragma unroll 4
    for (int kk = it.get_local_id(0); kk < k; kk += 256) {
      const sycl::double2 a = col[kk];
      v = sycl::max(v, sycl::max(oz_abs_hi(a.x()), oz_abs_hi(a.y())));
    }
    v = sycl::reduce_over_group(it.get_group(), v, sycl::maximum<unsigned>());
    if (it.get_local_id(0) == 0) amax[c] = v;
  });
}

// Scales a complex operand to integers and writes its INT8 residues for the
// moduli j0 .. j0+G-1.  The output is k-contiguous ("TN" layout):
//   out[((2*jl + s) * rows_pad + r) * kp + pos]
// with s = 0/1 holding (Re, Im) for 4M or (Re + iota Im, Re - iota Im) for 2M.
// The k of each chunk of S = min(256, kp - k0) are permuted (the GEMMs are
// invariant under the same permutation of k in A and B): pos = k0 + 8 kk + q
// holds k = k0 + kk + (S / 8) q.  A work-group covers 8 rows x one chunk, and
// a work-item the 8 k of one (row, kk), packed into one 8-byte word per
// plane.  Every 64 work-items store 2 rows x 256 contiguous bytes (short
// runs in many rows are slow, mostly from address translation), and reads
//   B (TRANS = false, k contiguous): 2 rows x 512 contiguous bytes, with 32
//     items per row (kk) which store their own words;
//   A (TRANS = true, rows contiguous): 8 rows x 8 k (128-byte runs), with 8
//     items per k whose words reach the storing items through local memory.
// Consecutive work-groups advance along the contiguous dimension of the input.
template <bool TRANS, bool IS2M>
static void residues(sycl::queue &q, const sycl::double2 *in, int ld, int rows,
                     int k, const unsigned *amax, int8_t *out,
                     int rows_pad, int kp, int j0, int G, const OzakiParams *pp) {
  const size_t nchunk = (size_t)(kp + 255) / 256, nrow = (size_t)rows_pad / 8;
  q.submit([&](sycl::handler &h) {
    // [buffer][s][kk * 9 + row]; the padding avoids bank conflicts
    sycl::local_accessor<sycl::uint2, 1> tile(sycl::range<1>(TRANS ? 2 * 2 * 32 * 9 : 1), h);
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(TRANS ? nchunk : nrow,
                                                    (TRANS ? nrow : nchunk) * 256),
                                     sycl::range<2>(1, 256)),
                   [=](sycl::nd_item<2> it) {
      const OzakiParams &p = *pp;
      const int t = it.get_local_id(1);
      const int r0 = (TRANS ? it.get_group(1) : it.get_group(0)) * 8;
      const int k0 = (TRANS ? it.get_group(0) : it.get_group(1)) * 256;
      const int st = sycl::min(kp - k0, 256) / 8;
      const int rl = TRANS ? (t & 7) : (t >> 5), kk = TRANS ? (t >> 3) : (t & 31);
      const int r = r0 + rl;
      const int e = (r < rows) ? oz_scale_exponent(amax[r], p.bits) : 0;

      // Unconditional loads (from a valid address) let all of them be in flight.
      OzFixed x[8];
      const size_t rc = (r < rows) ? r : 0;
#pragma unroll
      for (int qq = 0; qq < 8; qq++) {
        const int kq = k0 + kk + st * qq;
        const bool valid = r < rows && kk < st && kq < k;
        const size_t kc = valid ? kq : 0;
        const sycl::double2 a = TRANS ? in[kc * ld + rc] : in[rc * ld + kc];
        x[qq] = oz_fixed(valid ? a.x() : 0.0, valid ? a.y() : 0.0, e);
      }

      const size_t plane = (size_t)rows_pad * kp;
      const int srl = t >> 5, skk = t & 31;  // the word this item stores
      int8_t *dst = out + (size_t)(r0 + srl) * kp + k0 + 8 * skk;
      for (int jl = 0; jl < G; jl++) {
        int s0[8], s1[8];
#pragma unroll
        for (int qq = 0; qq < 8; qq++) oz_residue_pair<IS2M>(x[qq], j0 + jl, p, s0[qq], s1[qq]);
        sycl::uint2 w0(oz_pack4(s0), oz_pack4(s0 + 4));
        sycl::uint2 w1(oz_pack4(s1), oz_pack4(s1 + 4));
        if (TRANS) {
          // Double buffered: a buffer is rewritten two moduli later, after
          // the next barrier, by which time all its reads are done.
          const int bo = (jl & 1) * 2 * 32 * 9;
          tile[bo + kk * 9 + rl] = w0;
          tile[bo + 32 * 9 + kk * 9 + rl] = w1;
          sycl::group_barrier(it.get_group());
          w0 = tile[bo + skk * 9 + srl];
          w1 = tile[bo + 32 * 9 + skk * 9 + srl];
        }
        if (skk < st) {
          *reinterpret_cast<sycl::uint2 *>(dst + (2 * jl) * plane) = w0;
          *reinterpret_cast<sycl::uint2 *>(dst + (2 * jl + 1) * plane) = w1;
        }
      }
    });
  });
}

// On Intel GPUs the CRT sums need more registers per item than SIMD32 offers.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#define OZ_ACC_SG_ATTR [[sycl::reqd_sub_group_size(16)]]
#else
#define OZ_ACC_SG_ATTR
#endif

// Folds the INT32 products of moduli j0 .. j0+G-1 into the CRT sums and, in
// the last pass, reconstructs and unscales C.
//   2M: products [2*jl + s] of size mp x np (C+ and C-)
//   4M: product [jl] of size 2mp x 2np = [ArBr ArBi; AiBr AiBi]
template <bool IS2M, bool FIRST, bool LAST>
static void accumulate(sycl::queue &q, const int32_t *Cint, int m, int n,
                       int mp, int np, int j0, int G, const OzakiParams *pp,
                       OzAcc *acc, const unsigned *amaxA, const unsigned *amaxB,
                       sycl::double2 *C) {
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(n, (size_t)(m + 255) / 256 * 256),
                                   sycl::range<2>(1, 256)),
                 [=](sycl::nd_item<2> it) OZ_ACC_SG_ATTR {
    const OzakiParams &p = *pp;
    const int i = it.get_global_id(1);
    const int j = it.get_global_id(0);
    if (i >= m) return;
    const size_t idx = (size_t)j * m + i;
    const size_t plane = (size_t)mp * np;

    OzSums a = {};
    for (int jl = 0; jl < G; jl++) {
      if (IS2M) {
        const int32_t *c = Cint + 2 * jl * plane + (size_t)j * mp + i;
        oz_add_2m(a, c[0], c[plane], j0 + jl, p);
      } else {
        const size_t ld = 2 * (size_t)mp;
        const int32_t *c = Cint + 4 * jl * plane;
        oz_add_4m(a, c[(size_t)j * ld + i], c[(size_t)j * ld + mp + i],
                  c[(size_t)(np + j) * ld + i], c[(size_t)(np + j) * ld + mp + i],
                  j0 + jl, p);
      }
    }
    OzAcc s = {0, 0, 0, 0};
    if (!FIRST) s = acc[idx];
    oz_fold(a, s);
    if (LAST) {
      const int sc = p.shift - oz_scale_exponent(amaxA[i], p.bits) -
                     oz_scale_exponent(amaxB[j], p.bits);
      C[idx] = sycl::double2(sycl::ldexp(oz_reconstruct(s.r1, s.r2, p), sc),
                             sycl::ldexp(oz_reconstruct(s.i1, s.i2, p), sc));
    } else {
      acc[idx] = s;
    }
  });
}

struct OzakiWorkspace {
  unsigned *amaxA, *amaxB;  // row (column) maxima of |A| (|B|), oz_abs_hi
  int8_t *resA, *resB;
  int32_t *Cint;
  OzAcc *acc;
  OzakiParams *P;  // device copy, indexed by the kernels
  int G2, G4;      // moduli per pass for 2M and 4M
  int kslice;      // k slice of a row_amax work-group
};

template <bool IS2M>
static void launch_accumulate(sycl::queue &q, bool first, bool last, const int32_t *Cint,
                              int m, int n, int mp, int np, int j0, int G,
                              const OzakiWorkspace &w, sycl::double2 *C) {
  if (first && last)
    accumulate<IS2M, true, true>(q, Cint, m, n, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
  else if (first)
    accumulate<IS2M, true, false>(q, Cint, m, n, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
  else if (last)
    accumulate<IS2M, false, true>(q, Cint, m, n, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
  else
    accumulate<IS2M, false, false>(q, Cint, m, n, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
}

// C = A * B with Ozaki-II, using the 2M (two real products per modulus) or
// the 4M (four real products per modulus) algorithm.
template <bool IS2M>
static void ozaki_zgemm(sycl::queue &q, int m, int n, int k, const OzakiParams &P,
                        const sycl::double2 *A, const sycl::double2 *B,
                        sycl::double2 *C, const OzakiWorkspace &w) {
  const int mp = oz_round_up(m, OZ_MN_ALIGN);
  const int np = oz_round_up(n, OZ_MN_ALIGN);
  const int kp = oz_round_up(k, OZ_K_ALIGN);
  const int G = IS2M ? w.G2 : w.G4;

  q.memset(w.amaxA, 0, m * sizeof(unsigned));
  row_amax(q, A, m, k, w.kslice, w.amaxA);
  col_amax(q, B, n, k, w.amaxB);

  for (int j0 = 0; j0 < P.num_moduli; j0 += G) {
    const int g = (P.num_moduli - j0 < G) ? P.num_moduli - j0 : G;
    residues<true, IS2M>(q, A, m, m, k, w.amaxA, w.resA, mp, kp, j0, g, w.P);
    residues<false, IS2M>(q, B, k, n, k, w.amaxB, w.resB, np, kp, j0, g, w.P);

    if (IS2M)
      int8_gemm_tn(q, mp, np, kp, w.resA, (size_t)mp * kp, w.resB, (size_t)np * kp,
                   w.Cint, mp, (size_t)mp * np, 2 * g);
    else  // [Ar; Ai] (2mp x kp) times [Br Bi] (kp x 2np) yields all four products.
      int8_gemm_tn(q, 2 * mp, 2 * np, kp, w.resA, 2 * (size_t)mp * kp, w.resB,
                   2 * (size_t)np * kp, w.Cint, 2 * mp, 4 * (size_t)mp * np, g);

    launch_accumulate<IS2M>(q, j0 == 0, j0 + g >= P.num_moduli, w.Cint, m, n, mp,
                            np, j0, g, w, C);
  }
}

// Time `repeat` invocations of `fn` after one warm-up call; returns ms/call.
template <typename F>
static double time_ms(sycl::queue &q, F fn, int repeat) {
  fn();
  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) fn();
  q.wait();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / repeat;
}

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 6) {
    printf("Usage: %s <m> <n> <k> <repeat> [num moduli]\n", argv[0]);
    printf("  C = A * B for complex double matrices A (m x k), B (k x n)\n");
    printf("  num moduli: Ozaki-II moduli, 1..%d (default 16)\n", OZ_MAX_MODULI);
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int num_moduli = (argc > 5) ? atoi(argv[5]) : 16;

  OzakiParams P;
  if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0 || n > OZ_MAX_N ||
      k > OZ_MAX_K || !ozaki_setup(num_moduli, k, P)) {
    printf("Unsupported configuration (m, n, k and repeat must be positive, n "
           "must be <= %d, k must be <= %d and the moduli must provide at "
           "least one bit per input)\n", OZ_MAX_N, OZ_MAX_K);
    return 1;
  }

  const size_t mk = (size_t)m * k, kn = (size_t)k * n, mn = (size_t)m * n;
  const double flops = 8.0 * (double)m * n * k;  // real flops of a ZGEMM

  printf("ZGEMM m=%d n=%d k=%d\n", m, n, k);
  printf("Ozaki-II: %d moduli (M ~ 2^%.1f), %d bits per scaled input\n",
         P.num_moduli, P.log2M, P.bits);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // ---- host data -----------------------------------------------------------
  std::vector<double> hA(2 * mk), hB(2 * kn), hC_native(2 * mn), hC_4m(2 * mn),
      hC_2m(2 * mn);
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &v : hA) v = dist(gen);
  for (auto &v : hB) v = dist(gen);

  // ---- device data ---------------------------------------------------------
  sycl::double2 *dA = sycl::malloc_device<sycl::double2>(mk, q);
  sycl::double2 *dB = sycl::malloc_device<sycl::double2>(kn, q);
  sycl::double2 *dC = sycl::malloc_device<sycl::double2>(mn, q);
  if (!dA || !dB || !dC) {
    printf("Failed to allocate the matrices\n");
    return 1;
  }
  q.memcpy(dA, hA.data(), mk * sizeof(sycl::double2));
  q.memcpy(dB, hB.data(), kn * sizeof(sycl::double2));
  q.wait();
  const std::complex<double> *zA = reinterpret_cast<const std::complex<double> *>(dA);
  const std::complex<double> *zB = reinterpret_cast<const std::complex<double> *>(dB);
  std::complex<double> *zC = reinterpret_cast<std::complex<double> *>(dC);

  // ---- workspace (halve the budget until the allocations succeed) ---------
  const size_t mp = oz_round_up(m, OZ_MN_ALIGN);
  const size_t np = oz_round_up(n, OZ_MN_ALIGN);
  const size_t kp = oz_round_up(k, OZ_K_ALIGN);
  const size_t acc_bytes = mn * sizeof(OzAcc);
  // Free device memory, like cudaMemGetInfo / hipMemGetInfo; fall back to the
  // total memory when the device does not report free memory.
  const sycl::device dev = q.get_device();
  const size_t free_mem =
      dev.has(sycl::aspect::ext_intel_free_memory)
          ? (size_t)dev.get_info<sycl::ext::intel::info::device::free_memory>()
          : (size_t)dev.get_info<sycl::info::device::global_mem_size>();
  const size_t cap = free_mem / 5 * 2;
  size_t budget = (cap > acc_bytes) ? cap - acc_bytes : 0;

  OzakiWorkspace w = {};
  for (;;) {
    w.G2 = ozaki_group_size(P.num_moduli, true, mp, np, kp, budget);
    w.G4 = ozaki_group_size(P.num_moduli, false, mp, np, kp, budget);
    const int Gmax = (w.G2 > w.G4) ? w.G2 : w.G4;
    const size_t cint_elems = ((size_t)w.G2 * 2 > (size_t)w.G4 * 4 ? (size_t)w.G2 * 2
                                                                     : (size_t)w.G4 * 4) * mp * np;
    w.amaxA = sycl::malloc_device<unsigned>(mp, q);
    w.amaxB = sycl::malloc_device<unsigned>(np, q);
    w.resA = sycl::malloc_device<int8_t>((size_t)Gmax * 2 * mp * kp, q);
    w.resB = sycl::malloc_device<int8_t>((size_t)Gmax * 2 * np * kp, q);
    w.Cint = sycl::malloc_device<int32_t>(cint_elems, q);
    w.acc = (w.G2 < P.num_moduli || w.G4 < P.num_moduli)
                ? (OzAcc *)sycl::malloc_device(acc_bytes, q) : nullptr;
    const bool ok = w.amaxA && w.amaxB && w.resA && w.resB && w.Cint &&
                    (w.acc || (w.G2 == P.num_moduli && w.G4 == P.num_moduli));
    if (ok) break;
    for (void *ptr : {(void *)w.amaxA, (void *)w.amaxB, (void *)w.resA, (void *)w.resB,
                      (void *)w.Cint, (void *)w.acc})
      if (ptr) sycl::free(ptr, q);
    if (w.G2 == 1 && w.G4 == 1) {
      printf("Failed to allocate the Ozaki-II workspace\n");
      return 1;
    }
    budget /= 2;
  }
  printf("Moduli per pass: 2M %d, 4M %d\n", w.G2, w.G4);
  {
    // About 8 work-groups per compute unit for the row maxima.
    int cus = dev.get_info<sycl::ext::oneapi::info::device::num_compute_units>();
    const int slices = (8 * cus + (m + 63) / 64 - 1) / ((m + 63) / 64);
    w.kslice = oz_round_up((k + slices - 1) / slices, 4);
  }
  w.P = sycl::malloc_device<OzakiParams>(1, q);
  if (!w.P) {
    printf("Failed to allocate the Ozaki-II parameters\n");
    return 1;
  }
  q.memcpy(w.P, &P, sizeof(OzakiParams)).wait();

  const std::complex<double> zalpha(1.0, 0.0), zbeta(0.0, 0.0);

  // ---- native ZGEMM --------------------------------------------------------
  auto run_native = [&]() {
    onelib::blas::column_major::gemm(
        q, onelib::transpose::nontrans, onelib::transpose::nontrans,
        m, n, k, zalpha, zA, m, zB, k, zbeta, zC, m);
  };
  // Devices without FP64 hardware (e.g. Intel Arc GPUs with FP64 emulation)
  // may not provide a native ZGEMM; the results are then checked against a
  // host reference on sampled columns.
  bool native = true;
  try {
    run_native();
    q.memcpy(hC_native.data(), dC, mn * sizeof(sycl::double2)).wait();
  } catch (const std::exception &e) {
    printf("Native ZGEMM is not available on this device: %s\n", e.what());
    native = false;
  }

  if (native && (size_t)m * n * k <= (size_t)256 * 256 * 256) {
    std::vector<double> hC_host(2 * mn);
    reference_zgemm(m, n, k, hA.data(), hB.data(), hC_host.data());
    double ne, ce;
    compare(hC_native.data(), hC_host.data(), mn, ne, ce);
    printf("Native ZGEMM vs host reference: normwise error = %e, "
           "componentwise error = %e\n", ne, ce);
  }

  const double native_ms = native ? time_ms(q, run_native, repeat) : 0.0;

  const int nsamp = (n < 16) ? n : 16;
  std::vector<int> cols(nsamp);
  std::vector<double> hC_ref;
  if (!native) {
    hC_ref.resize(2 * (size_t)m * nsamp);
    for (int s = 0; s < nsamp; s++) {
      cols[s] = (nsamp > 1) ? (int)((long long)s * (n - 1) / (nsamp - 1)) : 0;
      reference_zgemm(m, 1, k, hA.data(), hB.data() + 2 * (size_t)cols[s] * k,
                      hC_ref.data() + 2 * (size_t)s * m);
    }
  }
  // Errors against the native result, or the host reference on sampled columns.
  auto check = [&](const std::vector<double> &C, double &ne, double &ce) {
    if (native) {
      compare(C.data(), hC_native.data(), mn, ne, ce);
      return;
    }
    std::vector<double> Cs(hC_ref.size());
    for (int s = 0; s < nsamp; s++)
      std::copy_n(C.data() + 2 * (size_t)cols[s] * m, 2 * (size_t)m, Cs.data() + 2 * (size_t)s * m);
    compare(Cs.data(), hC_ref.data(), (size_t)m * nsamp, ne, ce);
  };
  const char *ref_name = native ? "native" : "host (sampled)";

  // ---- Ozaki-II 4M and 2M --------------------------------------------------
  auto run_4m = [&]() { ozaki_zgemm<false>(q, m, n, k, P, dA, dB, dC, w); };
  auto run_2m = [&]() { ozaki_zgemm<true>(q, m, n, k, P, dA, dB, dC, w); };

  run_4m();
  q.memcpy(hC_4m.data(), dC, mn * sizeof(sycl::double2)).wait();
  const double m4_ms = time_ms(q, run_4m, repeat);

  run_2m();
  q.memcpy(hC_2m.data(), dC, mn * sizeof(sycl::double2)).wait();
  const double m2_ms = time_ms(q, run_2m, repeat);

  double m4_ne, m4_ce, m2_ne, m2_ce;
  check(hC_4m, m4_ne, m4_ce);
  check(hC_2m, m2_ne, m2_ce);
  const bool identical = memcmp(hC_2m.data(), hC_4m.data(), mn * sizeof(sycl::double2)) == 0;

  // ---- report --------------------------------------------------------------
  if (native)
    printf("Native ZGEMM:           %10.4f ms  %10.2f GFLOPS\n", native_ms, flops / (native_ms * 1e6));
  printf("Ozaki-II 4M ZGEMM:      %10.4f ms  %10.2f GFLOPS\n", m4_ms, flops / (m4_ms * 1e6));
  printf("Ozaki-II 2M ZGEMM:      %10.4f ms  %10.2f GFLOPS\n", m2_ms, flops / (m2_ms * 1e6));
  printf("Ozaki-II 4M vs %s:     normwise error = %e, componentwise error = %e\n",
         ref_name, m4_ne, m4_ce);
  printf("Ozaki-II 2M vs %s:     normwise error = %e, componentwise error = %e\n",
         ref_name, m2_ne, m2_ce);
  printf("Ozaki-II 2M and 4M results are bitwise identical: %s\n", identical ? "yes" : "no");
  printf("Speedup of Ozaki-II 2M over Ozaki-II 4M:  %.2fx\n", m4_ms / m2_ms);
  if (native)
    printf("Speedup of Ozaki-II 2M over native ZGEMM: %.2fx\n", native_ms / m2_ms);
  printf("Average execution time of Ozaki-II 2M ZGEMM: %f (ms)\n", m2_ms);

  const double tol = 1e-13 + std::ldexp(32.0, -P.bits);
  printf("%s\n", (identical && m2_ne <= tol) ? "PASS" : "FAIL");

  sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
  sycl::free(w.amaxA, q); sycl::free(w.amaxB, q);
  sycl::free(w.resA, q); sycl::free(w.resB, q);
  sycl::free(w.Cint, q);
  if (w.acc) sycl::free(w.acc, q);
  sycl::free(w.P, q);
  return 0;
}
