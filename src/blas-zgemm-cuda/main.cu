// ZGEMM emulation with the Ozaki-II scheme and the 2M algorithm.
//
// Starting with CUDA Toolkit 13.4 cuBLAS accelerates ZGEMM on integer Tensor
// Cores with the Ozaki-II scheme combined with the "2M" algorithm (two real
// GEMMs per complex GEMM instead of four).  With Ozaki-II every addition,
// subtraction and product is an exact integer operation, so the complex
// multiplication method no longer affects accuracy.
//
//   https://developer.nvidia.com/blog/faster-scientific-simulations-with-new-fp64-algorithms-in-nvidia-cublas/
//
// The benchmark measures C = A * B for complex double matrices with
//
//   1. native ZGEMM            - cublasZgemm on the native FP64 pipeline
//   2. cuBLAS emulated ZGEMM   - cublasZgemm with the fixed-point emulation
//                                math mode; the algorithm is chosen by the
//                                installed cuBLAS
//   3. Ozaki-II 4M             - an Ozaki-II implementation built on INT8
//      Ozaki-II 2M               cuBLAS GEMMs (see ozaki2.h), using four or
//                                two real products per modulus
//
// Accuracy is checked against native ZGEMM (and a host reference for small
// sizes).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ozaki2.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),     \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st = (call);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__,         \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

//============================================================================
// Ozaki-II kernels
//============================================================================

// Largest magnitudes (oz_abs_hi) of the rows of A (m x k, column-major), from
// which the scale exponents follow (oz_scale_exponent).  A block covers 64
// rows (one contiguous 1 KB segment per 64 threads and column) and a slice of
// `kslice` columns, so that the grid can fill the device for any shape; the
// slices combine through atomics on the zero-initialized `amax`.
__global__ void __launch_bounds__(256)
row_amax_kernel(const double2 *__restrict__ A, int m, int k, int kslice,
                unsigned *__restrict__ amax) {
  __shared__ unsigned smax[4][64];
  const int x = threadIdx.x & 63, y = threadIdx.x >> 6;
  const int r = blockIdx.x * 64 + x;
  const int kb = blockIdx.y * kslice;
  const int ke = (kb + kslice < k) ? kb + kslice : k;
  unsigned v = 0;
  if (r < m)
#pragma unroll 4
    for (int kk = kb + y; kk < ke; kk += 4) {
      const double2 a = A[(size_t)kk * m + r];
      v = max(v, max(oz_abs_hi(a.x), oz_abs_hi(a.y)));
    }
  smax[y][x] = v;
  __syncthreads();
  if (y == 0 && r < m) {
    for (int t = 1; t < 4; t++) v = max(v, smax[t][x]);
    if (v) atomicMax(amax + r, v);
  }
}

// Largest magnitudes of the columns of B (k x n, column-major): one block
// per column.
__global__ void __launch_bounds__(256)
col_amax_kernel(const double2 *__restrict__ B, int k, unsigned *__restrict__ amax) {
  __shared__ unsigned smax[8];
  const double2 *col = B + (size_t)blockIdx.x * k;
  unsigned v = 0;
#pragma unroll 4
  for (int kk = threadIdx.x; kk < k; kk += 256) {
    const double2 a = col[kk];
    v = max(v, max(oz_abs_hi(a.x), oz_abs_hi(a.y)));
  }
  for (int off = 16; off > 0; off >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, off));
  if ((threadIdx.x & 31) == 0) smax[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int t = 1; t < 8; t++) v = max(v, smax[t]);
    amax[blockIdx.x] = v;
  }
}

// Scales a complex operand to integers and writes its INT8 residues for the
// moduli j0 .. j0+G-1.  The output is k-contiguous ("TN" layout):
//   out[((2*jl + s) * rows_pad + r) * kp + pos]
// with s = 0/1 holding (Re, Im) for 4M or (Re + iota Im, Re - iota Im) for 2M.
// The k of each chunk of S = min(256, kp - k0) are permuted (the GEMMs are
// invariant under the same permutation of k in A and B): pos = k0 + 8 kk + q
// holds k = k0 + kk + (S / 8) q.  A block covers 8 rows x one chunk, and a
// thread the 8 k of one (row, kk), packed into one 8-byte word per plane.
// Every 64 threads store 2 rows x 256 contiguous bytes per instruction (short
// runs in many rows are slow, mostly from address translation), and read
//   B (TRANS = false, k contiguous): 2 rows x 512 contiguous bytes, with 32
//     threads per row (kk) which store their own words;
//   A (TRANS = true, rows contiguous): 8 rows x 8 k (128-byte runs), with 8
//     threads per k whose words reach the storing threads through shared
//     memory.
// Consecutive blocks advance along the contiguous dimension of the input.
template <bool TRANS, bool IS2M>
__global__ void __launch_bounds__(256)
residue_kernel(const double2 *__restrict__ in, int ld, int rows, int k,
               const unsigned *__restrict__ amax,
               int8_t *__restrict__ out, int rows_pad, int kp, int j0, int G,
               const OzakiParams *__restrict__ pp) {
  // [buffer][s][kk * 9 + row]; the padding avoids bank conflicts
  __shared__ uint2 tile[TRANS ? 2 : 1][2][32 * 9];
  const OzakiParams &P = *pp;
  const int t = threadIdx.x;
  const int r0 = (TRANS ? blockIdx.x : blockIdx.y) * 8;
  const int k0 = (TRANS ? blockIdx.y : blockIdx.x) * 256;
  const int st = ((kp - k0 < 256) ? kp - k0 : 256) / 8;
  const int rl = TRANS ? (t & 7) : (t >> 5), kk = TRANS ? (t >> 3) : (t & 31);
  const int r = r0 + rl;
  const int e = (r < rows) ? oz_scale_exponent(amax[r], P.bits) : 0;

  // Unconditional loads (from a valid address) let all of them be in flight.
  OzFixed x[8];
  const size_t rc = (r < rows) ? r : 0;
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const int kq = k0 + kk + st * q;
    const bool valid = r < rows && kk < st && kq < k;
    const size_t kc = valid ? kq : 0;
    const double2 a = TRANS ? in[kc * ld + rc] : in[rc * ld + kc];
    x[q] = oz_fixed(valid ? a.x : 0.0, valid ? a.y : 0.0, e);
  }

  const size_t plane = (size_t)rows_pad * kp;
  const int srl = t >> 5, skk = t & 31;  // the word this thread stores
  int8_t *dst = out + (size_t)(r0 + srl) * kp + k0 + 8 * skk;
  for (int jl = 0; jl < G; jl++) {
    int s0[8], s1[8];
#pragma unroll
    for (int q = 0; q < 8; q++) oz_residue_pair<IS2M>(x[q], j0 + jl, P, s0[q], s1[q]);
    uint2 w0 = make_uint2(oz_pack4(s0), oz_pack4(s0 + 4));
    uint2 w1 = make_uint2(oz_pack4(s1), oz_pack4(s1 + 4));
    if (TRANS) {
      // Double buffered: a buffer is rewritten two moduli later, after the
      // next barrier, by which time all its reads are done.
      uint2(*b)[32 * 9] = tile[jl & 1];
      b[0][kk * 9 + rl] = w0;
      b[1][kk * 9 + rl] = w1;
      __syncthreads();
      w0 = b[0][skk * 9 + srl];
      w1 = b[1][skk * 9 + srl];
    }
    if (skk < st) {
      *reinterpret_cast<uint2 *>(dst + (2 * jl) * plane) = w0;
      *reinterpret_cast<uint2 *>(dst + (2 * jl + 1) * plane) = w1;
    }
  }
}

// Folds the INT32 products of moduli j0 .. j0+G-1 into the CRT sums and, in
// the last pass, reconstructs and unscales C.
//   2M: products [2*jl + s] of size mp x np (C+ and C-)
//   4M: product [jl] of size 2mp x 2np = [ArBr ArBi; AiBr AiBi]
template <bool IS2M, bool FIRST, bool LAST>
__global__ void __launch_bounds__(256)
accumulate_kernel(const int32_t *__restrict__ Cint, int m, int mp, int np,
                  int j0, int G, const OzakiParams *__restrict__ pp,
                  OzAcc *__restrict__ acc, const unsigned *__restrict__ amaxA,
                  const unsigned *__restrict__ amaxB, double2 *__restrict__ C) {
  const OzakiParams &P = *pp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  if (i >= m) return;
  const size_t idx = (size_t)j * m + i;
  const size_t plane = (size_t)mp * np;

  OzSums a = {};
  for (int jl = 0; jl < G; jl++) {
    if (IS2M) {
      const int32_t *c = Cint + 2 * jl * plane + (size_t)j * mp + i;
      oz_add_2m(a, c[0], c[plane], j0 + jl, P);
    } else {
      const size_t ld = 2 * (size_t)mp;
      const int32_t *c = Cint + 4 * jl * plane;
      oz_add_4m(a, c[(size_t)j * ld + i], c[(size_t)j * ld + mp + i],
                c[(size_t)(np + j) * ld + i], c[(size_t)(np + j) * ld + mp + i],
                j0 + jl, P);
    }
  }
  OzAcc s = {0, 0, 0, 0};
  if (!FIRST) s = acc[idx];
  oz_fold(a, s);
  if (LAST) {
    const int sc = P.shift - oz_scale_exponent(amaxA[i], P.bits) -
                   oz_scale_exponent(amaxB[j], P.bits);
    C[idx] = make_double2(ldexp(oz_reconstruct(s.r1, s.r2, P), sc),
                          ldexp(oz_reconstruct(s.i1, s.i2, P), sc));
  } else {
    acc[idx] = s;
  }
}

struct OzakiWorkspace {
  unsigned *amaxA, *amaxB;  // row (column) maxima of |A| (|B|), oz_abs_hi
  int8_t *resA, *resB;
  int32_t *Cint;
  OzAcc *acc;
  OzakiParams *P;  // device copy, indexed by the kernels
  int G2, G4;      // moduli per pass for 2M and 4M
  int kslice;      // k slice of a row_amax_kernel block
};

template <bool IS2M>
static void launch_accumulate(bool first, bool last, dim3 grid, const int32_t *Cint,
                              int m, int mp, int np, int j0, int G,
                              const OzakiWorkspace &w, double2 *C) {
  if (first && last)
    accumulate_kernel<IS2M, true, true><<<grid, 256>>>(Cint, m, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
  else if (first)
    accumulate_kernel<IS2M, true, false><<<grid, 256>>>(Cint, m, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
  else if (last)
    accumulate_kernel<IS2M, false, true><<<grid, 256>>>(Cint, m, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
  else
    accumulate_kernel<IS2M, false, false><<<grid, 256>>>(Cint, m, mp, np, j0, G, w.P, w.acc, w.amaxA, w.amaxB, C);
}

// C = A * B with Ozaki-II, using the 2M (two real products per modulus) or
// the 4M (four real products per modulus) algorithm.
template <bool IS2M>
static void ozaki_zgemm(cublasHandle_t blas, int m, int n, int k,
                        const OzakiParams &P, const double2 *A,
                        const double2 *B, double2 *C, const OzakiWorkspace &w) {
  const int mp = oz_round_up(m, OZ_MN_ALIGN);
  const int np = oz_round_up(n, OZ_MN_ALIGN);
  const int kp = oz_round_up(k, OZ_K_ALIGN);
  const int G = IS2M ? w.G2 : w.G4;
  const int32_t one = 1, zero = 0;

  CUDA_CHECK(cudaMemsetAsync(w.amaxA, 0, m * sizeof(unsigned)));
  row_amax_kernel<<<dim3((m + 63) / 64, (k + w.kslice - 1) / w.kslice), 256>>>(
      A, m, k, w.kslice, w.amaxA);
  col_amax_kernel<<<n, 256>>>(B, k, w.amaxB);

  for (int j0 = 0; j0 < P.num_moduli; j0 += G) {
    const int g = (P.num_moduli - j0 < G) ? P.num_moduli - j0 : G;
    residue_kernel<true, IS2M><<<dim3(mp / 8, (kp + 255) / 256), 256>>>(
        A, m, m, k, w.amaxA, w.resA, mp, kp, j0, g, w.P);
    residue_kernel<false, IS2M><<<dim3((kp + 255) / 256, np / 8), 256>>>(
        B, k, n, k, w.amaxB, w.resB, np, kp, j0, g, w.P);

    if (IS2M) {
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          blas, CUBLAS_OP_T, CUBLAS_OP_N, mp, np, kp, &one,
          w.resA, CUDA_R_8I, kp, (long long)mp * kp,
          w.resB, CUDA_R_8I, kp, (long long)np * kp, &zero,
          w.Cint, CUDA_R_32I, mp, (long long)mp * np, 2 * g,
          CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    } else {
      // [Ar; Ai] (2mp x kp) times [Br Bi] (kp x 2np) yields all four products.
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * mp, 2 * np, kp, &one,
          w.resA, CUDA_R_8I, kp, 2LL * mp * kp,
          w.resB, CUDA_R_8I, kp, 2LL * np * kp, &zero,
          w.Cint, CUDA_R_32I, 2 * mp, 4LL * mp * np, g,
          CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    }

    launch_accumulate<IS2M>(j0 == 0, j0 + g >= P.num_moduli,
                            dim3((m + 255) / 256, n), w.Cint, m, mp, np, j0, g,
                            w, C);
  }
}

// Time `repeat` invocations of `fn` after one warm-up call; returns ms/call.
template <typename F>
static double time_ms(F fn, int repeat) {
  fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / repeat;
}

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 7) {
    printf("Usage: %s <m> <n> <k> <repeat> [num moduli] [mantissa bits]\n", argv[0]);
    printf("  C = A * B for complex double matrices A (m x k), B (k x n)\n");
    printf("  num moduli: Ozaki-II moduli, 1..%d (default 16)\n", OZ_MAX_MODULI);
    printf("  mantissa bits: cuBLAS emulation; 0 (default) lets cuBLAS choose "
           "dynamically, > 0 fixes the bit budget (53 matches FP64)\n");
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int num_moduli = (argc > 5) ? atoi(argv[5]) : 16;
  const int mantissa_bits = (argc > 6) ? atoi(argv[6]) : 0;

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

  // ---- host data -----------------------------------------------------------
  std::vector<double> hA(2 * mk), hB(2 * kn), hC_native(2 * mn), hC_cublas(2 * mn),
      hC_4m(2 * mn), hC_2m(2 * mn);
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &v : hA) v = dist(gen);
  for (auto &v : hB) v = dist(gen);

  // ---- device data ---------------------------------------------------------
  double2 *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, mk * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&dB, kn * sizeof(double2)));
  CUDA_CHECK(cudaMalloc(&dC, mn * sizeof(double2)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), mk * sizeof(double2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), kn * sizeof(double2), cudaMemcpyHostToDevice));
  const cuDoubleComplex *zA = reinterpret_cast<const cuDoubleComplex *>(dA);
  const cuDoubleComplex *zB = reinterpret_cast<const cuDoubleComplex *>(dB);
  cuDoubleComplex *zC = reinterpret_cast<cuDoubleComplex *>(dC);

  const size_t mp = oz_round_up(m, OZ_MN_ALIGN);
  const size_t np = oz_round_up(n, OZ_MN_ALIGN);
  const size_t kp = oz_round_up(k, OZ_K_ALIGN);
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  const size_t acc_bytes = mn * sizeof(OzAcc);
  const size_t cap = free_mem / 5 * 2;
  const size_t budget = (cap > acc_bytes) ? cap - acc_bytes : 0;

  OzakiWorkspace w;
  w.G2 = ozaki_group_size(P.num_moduli, true, mp, np, kp, budget);
  w.G4 = ozaki_group_size(P.num_moduli, false, mp, np, kp, budget);
  const int Gmax = (w.G2 > w.G4) ? w.G2 : w.G4;
  const size_t cint_elems = ((size_t)w.G2 * 2 > (size_t)w.G4 * 4 ? (size_t)w.G2 * 2
                                                                   : (size_t)w.G4 * 4) * mp * np;
  CUDA_CHECK(cudaMalloc(&w.P, sizeof(OzakiParams)));
  CUDA_CHECK(cudaMemcpy(w.P, &P, sizeof(OzakiParams), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&w.amaxA, mp * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&w.amaxB, np * sizeof(unsigned)));
  {
    // About 8 blocks per multiprocessor for the row maxima.
    int dev, sms;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int slices = (8 * sms + (m + 63) / 64 - 1) / ((m + 63) / 64);
    w.kslice = oz_round_up((k + slices - 1) / slices, 4);
  }
  CUDA_CHECK(cudaMalloc(&w.resA, (size_t)Gmax * 2 * mp * kp));
  CUDA_CHECK(cudaMalloc(&w.resB, (size_t)Gmax * 2 * np * kp));
  CUDA_CHECK(cudaMalloc(&w.Cint, cint_elems * sizeof(int32_t)));
  w.acc = nullptr;
  if (w.G2 < P.num_moduli || w.G4 < P.num_moduli)
    CUDA_CHECK(cudaMalloc(&w.acc, acc_bytes));
  printf("Moduli per pass: 2M %d, 4M %d\n", w.G2, w.G4);

  const cuDoubleComplex zalpha = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zbeta = make_cuDoubleComplex(0.0, 0.0);

  // ---- native ZGEMM --------------------------------------------------------
  cublasHandle_t native;
  CUBLAS_CHECK(cublasCreate(&native));
  CUBLAS_CHECK(cublasSetMathMode(native, CUBLAS_DEFAULT_MATH));

  auto run_native = [&]() {
    CUBLAS_CHECK(cublasZgemm(native, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &zalpha,
                             zA, m, zB, k, &zbeta, zC, m));
  };
  run_native();
  CUDA_CHECK(cudaMemcpy(hC_native.data(), dC, mn * sizeof(double2), cudaMemcpyDeviceToHost));

  if ((size_t)m * n * k <= (size_t)256 * 256 * 256) {
    std::vector<double> hC_host(2 * mn);
    reference_zgemm(m, n, k, hA.data(), hB.data(), hC_host.data());
    double ne, ce;
    compare(hC_native.data(), hC_host.data(), mn, ne, ce);
    printf("Native ZGEMM vs host reference: normwise error = %e, "
           "componentwise error = %e\n", ne, ce);
  }

  const double native_ms = time_ms(run_native, repeat);

  // ---- cuBLAS emulated ZGEMM -----------------------------------------------
  cublasHandle_t emu;
  CUBLAS_CHECK(cublasCreate(&emu));
  // FP64 fixed-point emulation is available from cuBLAS 13.2 (CUDA 13.1) on.
#if CUBLAS_VERSION >= 130200
  cublasStatus_t st = cublasSetMathMode(emu, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
  if (st == CUBLAS_STATUS_SUCCESS)
    st = cublasSetEmulationStrategy(emu, CUBLAS_EMULATION_STRATEGY_EAGER);
  if (st == CUBLAS_STATUS_SUCCESS && mantissa_bits > 0) {
    st = cublasSetFixedPointEmulationMantissaControl(emu, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
    if (st == CUBLAS_STATUS_SUCCESS)
      st = cublasSetFixedPointEmulationMaxMantissaBitCount(emu, mantissa_bits);
  }
#else
  (void)mantissa_bits;
  cublasStatus_t st = CUBLAS_STATUS_NOT_SUPPORTED;
#endif
  const bool have_cublas_emu = (st == CUBLAS_STATUS_SUCCESS);
  double cublas_ms = 0.0, cublas_ne = 0.0, cublas_ce = 0.0;
  if (have_cublas_emu) {
    auto run_emu = [&]() {
      CUBLAS_CHECK(cublasZgemm(emu, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &zalpha,
                               zA, m, zB, k, &zbeta, zC, m));
    };
    run_emu();
    CUDA_CHECK(cudaMemcpy(hC_cublas.data(), dC, mn * sizeof(double2), cudaMemcpyDeviceToHost));
    compare(hC_cublas.data(), hC_native.data(), mn, cublas_ne, cublas_ce);
    cublas_ms = time_ms(run_emu, repeat);
  }

  // ---- Ozaki-II 4M and 2M --------------------------------------------------
  auto run_4m = [&]() { ozaki_zgemm<false>(native, m, n, k, P, dA, dB, dC, w); };
  auto run_2m = [&]() { ozaki_zgemm<true>(native, m, n, k, P, dA, dB, dC, w); };

  run_4m();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(hC_4m.data(), dC, mn * sizeof(double2), cudaMemcpyDeviceToHost));
  const double m4_ms = time_ms(run_4m, repeat);

  run_2m();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(hC_2m.data(), dC, mn * sizeof(double2), cudaMemcpyDeviceToHost));
  const double m2_ms = time_ms(run_2m, repeat);

  double m4_ne, m4_ce, m2_ne, m2_ce;
  compare(hC_4m.data(), hC_native.data(), mn, m4_ne, m4_ce);
  compare(hC_2m.data(), hC_native.data(), mn, m2_ne, m2_ce);
  const bool identical = memcmp(hC_2m.data(), hC_4m.data(), mn * sizeof(double2)) == 0;

  // ---- report --------------------------------------------------------------
  printf("Native ZGEMM:           %10.4f ms  %10.2f GFLOPS\n", native_ms, flops / (native_ms * 1e6));
  if (have_cublas_emu)
    printf("cuBLAS emulated ZGEMM:  %10.4f ms  %10.2f GFLOPS\n", cublas_ms, flops / (cublas_ms * 1e6));
  else
    printf("cuBLAS emulated ZGEMM:  not supported by this cuBLAS (status %d)\n", (int)st);
  printf("Ozaki-II 4M ZGEMM:      %10.4f ms  %10.2f GFLOPS\n", m4_ms, flops / (m4_ms * 1e6));
  printf("Ozaki-II 2M ZGEMM:      %10.4f ms  %10.2f GFLOPS\n", m2_ms, flops / (m2_ms * 1e6));

  if (have_cublas_emu)
    printf("cuBLAS emulated vs native: normwise error = %e, componentwise error = %e\n",
           cublas_ne, cublas_ce);
  printf("Ozaki-II 4M vs native:     normwise error = %e, componentwise error = %e\n",
         m4_ne, m4_ce);
  printf("Ozaki-II 2M vs native:     normwise error = %e, componentwise error = %e\n",
         m2_ne, m2_ce);
  printf("Ozaki-II 2M and 4M results are bitwise identical: %s\n", identical ? "yes" : "no");
  printf("Speedup of Ozaki-II 2M over Ozaki-II 4M:  %.2fx\n", m4_ms / m2_ms);
  printf("Speedup of Ozaki-II 2M over native ZGEMM: %.2fx\n", native_ms / m2_ms);
  printf("Average execution time of Ozaki-II 2M ZGEMM: %f (ms)\n", m2_ms);

  const double tol = 1e-13 + ldexp(32.0, -P.bits);
  printf("%s\n", (identical && m2_ne <= tol) ? "PASS" : "FAIL");

  CUBLAS_CHECK(cublasDestroy(native));
  CUBLAS_CHECK(cublasDestroy(emu));
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(w.P)); CUDA_CHECK(cudaFree(w.amaxA)); CUDA_CHECK(cudaFree(w.amaxB));
  CUDA_CHECK(cudaFree(w.resA)); CUDA_CHECK(cudaFree(w.resB));
  CUDA_CHECK(cudaFree(w.Cint));
  if (w.acc) CUDA_CHECK(cudaFree(w.acc));
  return 0;
}
