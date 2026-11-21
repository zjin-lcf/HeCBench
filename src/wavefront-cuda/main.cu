// CUDA port of the handbook wavefront kernels. CUDA warps are 32 lanes and
// have no DPP, ds_swizzle, ballot_w64, or mbcnt. Each kernel uses the warp
// shuffle or vote intrinsic that computes the same result:
//   readlane     __shfl_sync from lane 0
//   rotate       __shfl_sync of (lane - 1) mod 32
//   swizzle      __shfl_xor_sync by 4
//   reduce-shfl  __shfl_down_sync
//   reduce-dpp   the wave32 DPP lane map, expressed with __shfl_sync
//   reduce-wave  xor butterfly, sum in every lane
//   compact      __ballot_sync and __popc

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kWarp = 32;
constexpr int kBlock = 256;
constexpr unsigned kMask = 0xffffffffu;

#define CHECK(cmd)                                                                                 \
  do {                                                                                             \
    cudaError_t _e = (cmd);                                                                        \
    if (_e != cudaSuccess) {                                                                       \
      std::printf("FAIL cuda %s: %s\n", #cmd, cudaGetErrorString(_e));                             \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

__device__ int lane_of() { return threadIdx.x & (kWarp - 1); }

__device__ float shfl_down_reduce(float val) {
  for (int offset = kWarp / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(kMask, val, offset);
  return val;
}

// quad_perm xor1, quad_perm xor2, row_ror 4, row_ror 8, then lane 15.
// Lane 31 holds the wave sum. Earlier lanes hold partials.
__device__ float dpp_lane_reduce(float val) {
  int lane = lane_of();
  val += __shfl_sync(kMask, val, lane ^ 1);
  val += __shfl_sync(kMask, val, lane ^ 2);
  int row = lane & ~15;
  val += __shfl_sync(kMask, val, row + ((lane - 4) & 15));
  val += __shfl_sync(kMask, val, row + ((lane - 8) & 15));
  val += __shfl_sync(kMask, val, 15);
  return val;
}

__device__ unsigned xor_reduce(unsigned val) {
  for (int offset = kWarp / 2; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(kMask, val, offset);
  return val;
}

__global__ void broadcast_readlane(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val = (gid < n) ? input[gid] : 0;
  int max_val = val;
  for (int offset = kWarp / 2; offset > 0; offset >>= 1)
    max_val = max(max_val, __shfl_down_sync(kMask, max_val, offset));
  int broadcast = __shfl_sync(kMask, max_val, 0);
  if (gid < n)
    output[gid] = broadcast;
}

__global__ void rotate_warp(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = lane_of();
  int val = (gid < n) ? input[gid] : 0;
  for (int i = 0; i < kWarp; ++i) {
    val = __shfl_sync(kMask, val, (lane + kWarp - 1) % kWarp);
    val ^= i;
  }
  if (gid < n)
    output[gid] = val;
}

__global__ void swizzle_swap_groups(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val = (gid < n) ? input[gid] : 0;
  int swapped = __shfl_xor_sync(kMask, val, 4);
  if (gid < n)
    output[gid] = swapped;
}

template <int WarpSize>
__global__ void __launch_bounds__(kBlock)
reduce_shfl_down(const float *input, float *output, int n) {
  int gid = blockIdx.x * kBlock + threadIdx.x;
  int lane = threadIdx.x % WarpSize;
  int warpid = threadIdx.x / WarpSize;
  float val = (gid < n) ? input[gid] : 0.f;
  val = shfl_down_reduce(val);
  __shared__ float sdata[kBlock / 32];
  if (lane == 0)
    sdata[warpid] = val;
  __syncthreads();
  if (warpid == 0) {
    int num_warps = kBlock / WarpSize;
    val = (lane < num_warps) ? sdata[lane] : 0.f;
    val = shfl_down_reduce(val);
  }
  if (threadIdx.x == 0)
    output[blockIdx.x] = val;
}

template <int WarpSize>
__global__ void __launch_bounds__(kBlock)
reduce_dpp(const float *input, float *output, int n) {
  int gid = blockIdx.x * kBlock + threadIdx.x;
  int lane = threadIdx.x % WarpSize;
  int warpid = threadIdx.x / WarpSize;
  float val = (gid < n) ? input[gid] : 0.f;
  val = dpp_lane_reduce(val);
  __shared__ float sdata[kBlock / 32];
  if (lane == WarpSize - 1)
    sdata[warpid] = val;
  __syncthreads();
  if (warpid == 0) {
    int num_warps = kBlock / WarpSize;
    val = (lane < num_warps) ? sdata[lane] : 0.f;
    val = dpp_lane_reduce(val);
  }
  if (threadIdx.x == WarpSize - 1)
    output[blockIdx.x] = val;
}

template <int WarpSize>
__global__ void __launch_bounds__(kBlock)
reduce_wave(const unsigned *input, unsigned *output, int n) {
  int gid = blockIdx.x * kBlock + threadIdx.x;
  int lane = threadIdx.x % WarpSize;
  int warpid = threadIdx.x / WarpSize;
  unsigned val = (gid < n) ? input[gid] : 0u;
  unsigned sum = xor_reduce(val);
  __shared__ unsigned sdata[kBlock / 32];
  if (lane == 0)
    sdata[warpid] = sum;
  __syncthreads();
  if (warpid == 0) {
    int num_warps = kBlock / WarpSize;
    sum = (lane < num_warps) ? sdata[lane] : 0u;
    sum = xor_reduce(sum);
  }
  if (threadIdx.x == 0)
    output[blockIdx.x] = sum;
}

__global__ void compaction_index(const int *input, int *output, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = lane_of();
  int warpid = threadIdx.x / kWarp;
  int val = (gid < n) ? input[gid] : 0;
  bool keep = (gid < n) && (val > 0);
  unsigned mask = __ballot_sync(kMask, keep);
  int index = __popc(mask & ((1u << lane) - 1));
  __shared__ int sdata[kBlock / 32];
  int warp_count = __shfl_sync(kMask, index + (keep ? 1 : 0), kWarp - 1);
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
  CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    launch();
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

static void ref_rotate(const std::vector<int> &in, std::vector<int> &out) {
  out = in;
  std::vector<int> nxt(in.size());
  int nwarps = static_cast<int>(in.size()) / kWarp;
  for (int s = 0; s < kWarp; ++s) {
    for (int w = 0; w < nwarps; ++w) {
      for (int lane = 0; lane < kWarp; ++lane) {
        int src = w * kWarp + (lane - 1 + kWarp) % kWarp;
        nxt[w * kWarp + lane] = out[src] ^ s;
      }
    }
    out.swap(nxt);
  }
}

static void ref_swizzle(const std::vector<int> &in, std::vector<int> &out) {
  out.resize(in.size());
  for (int i = 0; i < static_cast<int>(in.size()); ++i) {
    int lane = i % kWarp;
    int src = (lane & ~31) | ((lane & 31) ^ 4);
    out[i] = in[i - lane + src];
  }
}

static void ref_max(const std::vector<int> &in, std::vector<int> &out) {
  out.resize(in.size());
  int nwarps = static_cast<int>(in.size()) / kWarp;
  for (int w = 0; w < nwarps; ++w) {
    int mx = in[w * kWarp];
    for (int lane = 1; lane < kWarp; ++lane)
      mx = std::max(mx, in[w * kWarp + lane]);
    for (int lane = 0; lane < kWarp; ++lane)
      out[w * kWarp + lane] = mx;
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

static void ref_compact(const std::vector<int> &in, std::vector<int> &out) {
  out.assign(in.size(), 0);
  int nblocks = static_cast<int>(in.size()) / kBlock;
  int warps = kBlock / kWarp;
  for (int b = 0; b < nblocks; ++b) {
    int warp_base = 0;
    for (int w = 0; w < warps; ++w) {
      int index = 0;
      for (int lane = 0; lane < kWarp; ++lane) {
        int gid = b * kBlock + w * kWarp + lane;
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

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("INFO warpSize=%d path=cuda shfl ballot n=%d\n", prop.warpSize, n);
  if (prop.warpSize != kWarp) {
    std::printf("FAIL warpSize=%d compiled for %d\n", prop.warpSize, kWarp);
    return 1;
  }

  std::vector<int> hin(n), href(n), got(n);
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
  CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CHECK(cudaMalloc(&d_fin, n * sizeof(float)));
  CHECK(cudaMalloc(&d_fout, nblocks * sizeof(float)));
  CHECK(cudaMalloc(&d_uin, n * sizeof(unsigned)));
  CHECK(cudaMalloc(&d_uout, nblocks * sizeof(unsigned)));
  CHECK(cudaMemcpy(d_in, hin.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_fin, hfin.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_uin, huin.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));

  auto grid = dim3(nblocks);
  auto block = dim3(kBlock);

  auto run_i = [&](const char *name, auto kernel, auto ref) {
    ref(hin, href);
    double sec = time_launches(repeat, [&] { kernel(); });
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    report_i(name, sec, got.data(), href.data(), n);
  };

  run_i("readlane", [&] { broadcast_readlane<<<grid, block>>>(d_in, d_out, n); }, ref_max);
  run_i("rotate-shfl", [&] { rotate_warp<<<grid, block>>>(d_in, d_out, n); }, ref_rotate);
  run_i("swizzle", [&] { swizzle_swap_groups<<<grid, block>>>(d_in, d_out, n); }, ref_swizzle);

  std::vector<float> fgot(nblocks);
  ref_sum_f(hfin, hfref);
  double sec = time_launches(repeat, [&] {
    reduce_shfl_down<kWarp><<<grid, block>>>(d_fin, d_fout, n);
  });
  CHECK(cudaMemcpy(fgot.data(), d_fout, nblocks * sizeof(float), cudaMemcpyDeviceToHost));
  report_f("reduce-shfl", sec, fgot.data(), hfref.data(), nblocks);

  sec = time_launches(repeat, [&] { reduce_dpp<kWarp><<<grid, block>>>(d_fin, d_fout, n); });
  CHECK(cudaMemcpy(fgot.data(), d_fout, nblocks * sizeof(float), cudaMemcpyDeviceToHost));
  report_f("reduce-dpp", sec, fgot.data(), hfref.data(), nblocks);

  std::vector<unsigned> ugot(nblocks);
  ref_sum_u(huin, huref);
  sec = time_launches(repeat, [&] { reduce_wave<kWarp><<<grid, block>>>(d_uin, d_uout, n); });
  CHECK(cudaMemcpy(ugot.data(), d_uout, nblocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
  report_u("reduce-wave", sec, ugot.data(), huref.data(), nblocks);

  ref_compact(hin, href);
  CHECK(cudaMemset(d_out, 0, n * sizeof(int)));
  sec = time_launches(repeat, [&] { compaction_index<<<grid, block>>>(d_in, d_out, n); });
  CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  report_i("compact", sec, got.data(), href.data(), n);

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_fin));
  CHECK(cudaFree(d_fout));
  CHECK(cudaFree(d_uin));
  CHECK(cudaFree(d_uout));

  if (g_fail) {
    std::printf("FAIL kernels=%d\n", g_fail);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
