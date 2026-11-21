// Kernels from
// https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cross-arch/wavefront-builtins.html
//
// GFX8 and GFX9 take the wave64 path: wave_ror, row_bcast, ballot_w64, and
// mbcnt_hi. Other targets take the wave32 path: row_ror plus ds_bpermute,
// ds_swizzle 0x1e0, ballot_w32, and mbcnt_lo. Those arch macros are set only
// while compiling device code, so the host reference uses the runtime wave size.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__GFX8__) || defined(__GFX9__)
#define HAS_DPP_BROADCAST 1
#endif
constexpr int kBlock = 256;

constexpr int dpp_quad_perm_1032 = 0xb1;
constexpr int dpp_quad_perm_2301 = 0x4e;
constexpr int dpp_row_ror4 = 0x124;
constexpr int dpp_row_ror8 = 0x128;
constexpr int swizzle_swap_groups4 = 0x101F;
#ifdef HAS_DPP_BROADCAST
// wave_ror:1. The handbook constant 0x134 encodes wave_rol:1.
constexpr int dpp_wave_ror1 = 0x13c;
constexpr int dpp_row_bcast15 = 0x142;
constexpr int dpp_row_bcast31 = 0x143;
#else
constexpr int dpp_row_ror1 = 0x121;
constexpr int swizzle_bcast15 = 0x1e0;
#endif

#define CHECK(cmd)                                                                                 \
  do {                                                                                             \
    hipError_t _e = (cmd);                                                                         \
    if (_e != hipSuccess) {                                                                        \
      std::printf("FAIL hip %s: %s\n", #cmd, hipGetErrorString(_e));                               \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

__device__ int warp_rotate_right(int val) {
#ifdef HAS_DPP_BROADCAST
  return __builtin_amdgcn_mov_dpp(val, dpp_wave_ror1, 0xf, 0xf, false);
#else
  int rotated = __builtin_amdgcn_mov_dpp(val, dpp_row_ror1, 0xf, 0xf, false);
  int lane = threadIdx.x % warpSize;
  int addr = (lane == 0) ? 124 : (lane == 16) ? 60 : lane * 4;
  int wrapped = __builtin_amdgcn_ds_bpermute(addr, val);
  return (lane == 0 || lane == 16) ? wrapped : rotated;
#endif
}

__device__ int warp_rotate_right_shfl(int val) {
  int lane = threadIdx.x % warpSize;
  return __shfl(val, (lane - 1 + warpSize) % warpSize);
}

__device__ float shfl_down_reduce(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
  return val;
}

// wave_reduce is missing from older compilers, so those use a butterfly.
__device__ unsigned wave_sum(unsigned val) {
#if __has_builtin(__builtin_amdgcn_wave_reduce_add_u32)
  return __builtin_amdgcn_wave_reduce_add_u32(val, 0);
#else
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_xor(val, offset);
  return val;
#endif
}

__device__ float dpp_warp_reduce(float val) {
  auto as_int = [](float f) { return __float_as_int(f); };
  auto as_float = [](int i) { return __int_as_float(i); };
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_quad_perm_1032, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_quad_perm_2301, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_ror4, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_ror8, 0xf, 0xf, false));
#ifdef HAS_DPP_BROADCAST
  // Full row mask, matching rocPRIM. The sum is in the last lane.
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_bcast15, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_bcast31, 0xf, 0xf, false));
#else
  val += as_float(__builtin_amdgcn_ds_swizzle(as_int(val), swizzle_bcast15));
#endif
  return val;
}

__global__ void broadcast_readlane(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val = (gid < n) ? input[gid] : 0;
  int max_val = val;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    max_val = max(max_val, __shfl_down(max_val, offset));
  int broadcast = __builtin_amdgcn_readlane(max_val, 0);
  if (gid < n)
    output[gid] = broadcast;
}

__global__ void rotate_warp(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val = (gid < n) ? input[gid] : 0;
  for (int i = 0; i < warpSize; ++i) {
    val = warp_rotate_right(val);
    val ^= i;
  }
  if (gid < n)
    output[gid] = val;
}

__global__ void rotate_warp_shfl(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val = (gid < n) ? input[gid] : 0;
  for (int i = 0; i < warpSize; ++i) {
    val = warp_rotate_right_shfl(val);
    val ^= i;
  }
  if (gid < n)
    output[gid] = val;
}

__global__ void swizzle_swap_groups(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val = (gid < n) ? input[gid] : 0;
  int swapped = __builtin_amdgcn_ds_swizzle(val, swizzle_swap_groups4);
  if (gid < n)
    output[gid] = swapped;
}

__global__ void __launch_bounds__(kBlock)
reduce_shfl_down(const float *input, float *output, int n) {
  int gid = blockIdx.x * kBlock + threadIdx.x;
  int lane = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;
  float val = (gid < n) ? input[gid] : 0.f;
  val = shfl_down_reduce(val);
  __shared__ float sdata[kBlock / 32];
  if (lane == 0)
    sdata[warpid] = val;
  __syncthreads();
  if (warpid == 0) {
    int num_warps = kBlock / warpSize;
    val = (lane < num_warps) ? sdata[lane] : 0.f;
    val = shfl_down_reduce(val);
  }
  if (threadIdx.x == 0)
    output[blockIdx.x] = val;
}

__global__ void __launch_bounds__(kBlock)
reduce_dpp(const float *input, float *output, int n) {
  int gid = blockIdx.x * kBlock + threadIdx.x;
  int lane = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;
  float val = (gid < n) ? input[gid] : 0.f;
  val = dpp_warp_reduce(val);
  __shared__ float sdata[kBlock / 32];
  if (lane == warpSize - 1)
    sdata[warpid] = val;
  __syncthreads();
  if (warpid == 0) {
    int num_warps = kBlock / warpSize;
    val = (lane < num_warps) ? sdata[lane] : 0.f;
    val = dpp_warp_reduce(val);
  }
  if (threadIdx.x == warpSize - 1)
    output[blockIdx.x] = val;
}

__global__ void __launch_bounds__(kBlock)
reduce_wave(const unsigned *input, unsigned *output, int n) {
  int gid = blockIdx.x * kBlock + threadIdx.x;
  int lane = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;
  unsigned val = (gid < n) ? input[gid] : 0u;
  unsigned sum = wave_sum(val);
  __shared__ unsigned sdata[kBlock / 32];
  if (lane == 0)
    sdata[warpid] = sum;
  __syncthreads();
  if (warpid == 0) {
    int num_warps = kBlock / warpSize;
    sum = (lane < num_warps) ? sdata[lane] : 0u;
    sum = wave_sum(sum);
  }
  if (threadIdx.x == 0)
    output[blockIdx.x] = sum;
}

__global__ void compaction_index(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;
  int val = (gid < n) ? input[gid] : 0;
  bool keep = (gid < n) && (val > 0);
  int index;
#ifdef HAS_DPP_BROADCAST
  unsigned long long mask = __builtin_amdgcn_ballot_w64(keep);
  unsigned lo = static_cast<unsigned>(mask);
  unsigned hi = static_cast<unsigned>(mask >> 32);
  index = __builtin_amdgcn_mbcnt_lo(lo, 0);
  index = __builtin_amdgcn_mbcnt_hi(hi, index);
#else
  unsigned mask = __builtin_amdgcn_ballot_w32(keep);
  index = __builtin_amdgcn_mbcnt_lo(mask, 0);
#endif
  __shared__ int sdata[kBlock / 32];
  int warp_count = __builtin_amdgcn_readlane(index + (keep ? 1 : 0), warpSize - 1);
  if (lane == 0)
    sdata[warpid] = warp_count;
  __syncthreads();
  int warp_base = 0;
  for (int w = 0; w < warpid; ++w)
    warp_base += sdata[w];
  if (keep)
    output[blockIdx.x * blockDim.x + warp_base + index] = val;
}

static int g_fail = 0;

static void report_i(const char *name, double sec, const int *got, const int *ref, int n) {
  int mismatch = 0;
  for (int i = 0; i < n; ++i)
    mismatch += got[i] != ref[i];
  if (mismatch == 0) {
    std::printf("PASS %-18s time %.6f s\n", name, sec);
    return;
  }
  g_fail++;
  std::printf("FAIL %-18s mismatch=%d/%d time %.6f s", name, mismatch, n, sec);
  int shown = 0;
  for (int i = 0; i < n && shown < 4; ++i) {
    if (got[i] != ref[i]) {
      std::printf(" [%d got=%d ref=%d]", i, got[i], ref[i]);
      ++shown;
    }
  }
  std::printf("\n");
}

static void report_f(const char *name, double sec, const float *got, const float *ref, int n) {
  int mismatch = 0;
  for (int i = 0; i < n; ++i)
    mismatch += got[i] != ref[i];
  if (mismatch == 0) {
    std::printf("PASS %-18s time %.6f s\n", name, sec);
    return;
  }
  g_fail++;
  std::printf("FAIL %-18s mismatch=%d/%d time %.6f s", name, mismatch, n, sec);
  int shown = 0;
  for (int i = 0; i < n && shown < 4; ++i) {
    if (got[i] != ref[i]) {
      std::printf(" [%d got=%g ref=%g]", i, got[i], ref[i]);
      ++shown;
    }
  }
  std::printf("\n");
}

static void report_u(const char *name, double sec, const unsigned *got, const unsigned *ref, int n) {
  int mismatch = 0;
  for (int i = 0; i < n; ++i)
    mismatch += got[i] != ref[i];
  if (mismatch == 0) {
    std::printf("PASS %-18s time %.6f s\n", name, sec);
    return;
  }
  g_fail++;
  std::printf("FAIL %-18s mismatch=%d/%d time %.6f s", name, mismatch, n, sec);
  int shown = 0;
  for (int i = 0; i < n && shown < 4; ++i) {
    if (got[i] != ref[i]) {
      std::printf(" [%d got=%u ref=%u]", i, got[i], ref[i]);
      ++shown;
    }
  }
  std::printf("\n");
}

template <typename F>
static double time_launches(int repeat, F &&launch) {
  launch();
  CHECK(hipDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    launch();
  CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

static void ref_rotate(const std::vector<int> &in, std::vector<int> &out, int warp) {
  out = in;
  std::vector<int> nxt(in.size());
  int nwarps = static_cast<int>(in.size()) / warp;
  for (int s = 0; s < warp; ++s) {
    for (int w = 0; w < nwarps; ++w) {
      for (int lane = 0; lane < warp; ++lane) {
        int src = w * warp + (lane - 1 + warp) % warp;
        nxt[w * warp + lane] = out[src] ^ s;
      }
    }
    out.swap(nxt);
  }
}

static void ref_swizzle(const std::vector<int> &in, std::vector<int> &out, int warp) {
  out.resize(in.size());
  for (int i = 0; i < static_cast<int>(in.size()); ++i) {
    int lane = i % warp;
    int src = (lane & ~31) | ((lane & 31) ^ 4);
    out[i] = in[i - lane + src];
  }
}

static void ref_max(const std::vector<int> &in, std::vector<int> &out, int warp) {
  out.resize(in.size());
  int nwarps = static_cast<int>(in.size()) / warp;
  for (int w = 0; w < nwarps; ++w) {
    int mx = in[w * warp];
    for (int lane = 1; lane < warp; ++lane)
      mx = std::max(mx, in[w * warp + lane]);
    for (int lane = 0; lane < warp; ++lane)
      out[w * warp + lane] = mx;
  }
}

static void ref_sum_f(const std::vector<float> &in, std::vector<float> &out) {
  int nblocks = static_cast<int>(in.size()) / kBlock;
  out.assign(nblocks, 0.f);
  for (int b = 0; b < nblocks; ++b) {
    float s = 0.f;
    for (int i = 0; i < kBlock; ++i)
      s += in[b * kBlock + i];
    out[b] = s;
  }
}

static void ref_sum_u(const std::vector<unsigned> &in, std::vector<unsigned> &out) {
  int nblocks = static_cast<int>(in.size()) / kBlock;
  out.assign(nblocks, 0u);
  for (int b = 0; b < nblocks; ++b) {
    unsigned s = 0u;
    for (int i = 0; i < kBlock; ++i)
      s += in[b * kBlock + i];
    out[b] = s;
  }
}

static void ref_compact(const std::vector<int> &in, std::vector<int> &out, int warp) {
  out.assign(in.size(), 0);
  int nblocks = static_cast<int>(in.size()) / kBlock;
  int warps = kBlock / warp;
  for (int b = 0; b < nblocks; ++b) {
    int warp_base = 0;
    for (int w = 0; w < warps; ++w) {
      int index = 0;
      for (int lane = 0; lane < warp; ++lane) {
        int gid = b * kBlock + w * warp + lane;
        if (in[gid] > 0) {
          out[b * kBlock + warp_base + index] = in[gid];
          ++index;
        }
      }
      warp_base += index;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::printf("Usage: %s <n> <repeat>\n", argv[0]);
    return 1;
  }
  int n = std::atoi(argv[1]);
  int repeat = std::atoi(argv[2]);
  if (n <= 0 || repeat <= 0) {
    std::printf("FAIL n and repeat must be positive\n");
    return 1;
  }
  n = (n + kBlock - 1) / kBlock * kBlock;

  int warp = 0;
  CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, 0));
  if (warp != 32 && warp != 64) {
    std::printf("FAIL warpSize=%d\n", warp);
    return 1;
  }
  if (warp == 64)
    std::printf("INFO warpSize=%d path=wave64 wave_ror row_bcast ballot_w64 mbcnt_hi n=%d\n", warp, n);
  else
    std::printf("INFO warpSize=%d path=wave32 ballot_w32 mbcnt_lo row_ror ds_swizzle n=%d\n", warp, n);

  std::vector<int> hin(n), href(n);
  std::vector<float> hfin(n), hfref;
  std::vector<unsigned> huin(n), huref;
  for (int i = 0; i < n; ++i) {
    hin[i] = ((i * 17 + 3) % 31) - 10;
    hfin[i] = static_cast<float>((i % 13) + 1);
    huin[i] = static_cast<unsigned>((i % 13) + 1);
  }

  int *d_in = nullptr, *d_out = nullptr;
  float *d_fin = nullptr, *d_fout = nullptr;
  unsigned *d_uin = nullptr, *d_uout = nullptr;
  int nblocks = n / kBlock;
  CHECK(hipMalloc(&d_in, n * sizeof(int)));
  CHECK(hipMalloc(&d_out, n * sizeof(int)));
  CHECK(hipMalloc(&d_fin, n * sizeof(float)));
  CHECK(hipMalloc(&d_fout, nblocks * sizeof(float)));
  CHECK(hipMalloc(&d_uin, n * sizeof(unsigned)));
  CHECK(hipMalloc(&d_uout, nblocks * sizeof(unsigned)));
  CHECK(hipMemcpy(d_in, hin.data(), n * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_fin, hfin.data(), n * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_uin, huin.data(), n * sizeof(unsigned), hipMemcpyHostToDevice));

  auto grid = dim3(nblocks);
  auto block = dim3(kBlock);
  std::vector<int> got(n);

  auto run_i = [&](const char *name, auto kernel, auto ref) {
    ref(hin, href, warp);
    double sec = time_launches(repeat, [&] { kernel(); });
    CHECK(hipMemcpy(got.data(), d_out, n * sizeof(int), hipMemcpyDeviceToHost));
    report_i(name, sec, got.data(), href.data(), n);
  };

  run_i("readlane", [&] { broadcast_readlane<<<grid, block>>>(d_in, d_out, n); }, ref_max);
  run_i("rotate-dpp", [&] { rotate_warp<<<grid, block>>>(d_in, d_out, n); }, ref_rotate);
  run_i("rotate-shfl", [&] { rotate_warp_shfl<<<grid, block>>>(d_in, d_out, n); }, ref_rotate);
  run_i("swizzle", [&] { swizzle_swap_groups<<<grid, block>>>(d_in, d_out, n); }, ref_swizzle);

  std::vector<float> fgot(nblocks);
  ref_sum_f(hfin, hfref);
  double sec = time_launches(repeat, [&] { reduce_shfl_down<<<grid, block>>>(d_fin, d_fout, n); });
  CHECK(hipMemcpy(fgot.data(), d_fout, nblocks * sizeof(float), hipMemcpyDeviceToHost));
  report_f("reduce-shfl", sec, fgot.data(), hfref.data(), nblocks);

  sec = time_launches(repeat, [&] { reduce_dpp<<<grid, block>>>(d_fin, d_fout, n); });
  CHECK(hipMemcpy(fgot.data(), d_fout, nblocks * sizeof(float), hipMemcpyDeviceToHost));
  report_f("reduce-dpp", sec, fgot.data(), hfref.data(), nblocks);

  std::vector<unsigned> ugot(nblocks);
  ref_sum_u(huin, huref);
  sec = time_launches(repeat, [&] { reduce_wave<<<grid, block>>>(d_uin, d_uout, n); });
  CHECK(hipMemcpy(ugot.data(), d_uout, nblocks * sizeof(unsigned), hipMemcpyDeviceToHost));
  report_u("reduce-wave", sec, ugot.data(), huref.data(), nblocks);

  ref_compact(hin, href, warp);
  CHECK(hipMemset(d_out, 0, n * sizeof(int)));
  sec = time_launches(repeat, [&] { compaction_index<<<grid, block>>>(d_in, d_out, n); });
  CHECK(hipMemcpy(got.data(), d_out, n * sizeof(int), hipMemcpyDeviceToHost));
  report_i("compact", sec, got.data(), href.data(), n);

  CHECK(hipFree(d_in));
  CHECK(hipFree(d_out));
  CHECK(hipFree(d_fin));
  CHECK(hipFree(d_fout));
  CHECK(hipFree(d_uin));
  CHECK(hipFree(d_uout));

  if (g_fail) {
    std::printf("FAIL kernels=%d\n", g_fail);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
