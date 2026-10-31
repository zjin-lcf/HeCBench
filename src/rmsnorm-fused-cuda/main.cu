// Fused per-head Gemma RMSNorm, NeoX RoPE, and Q/gate deinterleave.
//
// This is the CUDA source. hipcc compiles the same file through
// src/rmsnorm-fused-hip/main.cu. The arithmetic follows SGLang
// fused_qk_gemma_rmsnorm_rope_gate
// (python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py):
//
//   y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w))
//   NeoX on the leading rotary_dim elements of y, then store as bf16
//   elements [rotary_dim, head_dim) are stored as y, without rotation
//
// The bf16 round happens after the norm and again after the rotation, matching
// that kernel. Gate values are copied and are not normalized.
//
// Layout, all row-major bf16 except the position and axis tables:
//   q_gate    [tokens, q_heads, 2, head_dim]   per head: q then gate
//   k         [tokens, kv_heads, head_dim]
//   q_out     [tokens, q_heads, head_dim]
//   k_out     [tokens, kv_heads, head_dim]
//   gate_out  [tokens, q_heads, head_dim]
//   q_weight, k_weight                         [head_dim], shared by every head
//   cos_sin_cache [max_pos, rotary_dim]        first half cos, second half sin
//   positions [tokens]                         mode 0, one position per token
//             [3, tokens]                      modes 1 and 2, rows T, H, W
//   axis_map  [rotary_dim / 2]                 which of the 3 rows owns a lane
//
// mode 0 is ordinary 1D RoPE. mode 1 assigns contiguous rotary lanes to T, then
// H, then W. mode 2 is the Qwen interleaved map: lane 0 stays T, H overwrites
// 1, 4, 7, ... and W overwrites 2, 5, 8, ... . head_dim is 64, 128, or 256.
// Supported rotary widths are even and no larger than head_dim. The fast kernel
// requires rotary_dim to be a multiple of 16; other widths use the fallback.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__HIP__)
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
using bf16 = __hip_bfloat16;
using gpuError_t = hipError_t;
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess
// HIP shuffle width is the subgroup size. It must stay inside one wavefront.
#define shfl_xor(v, m, w) __shfl_xor(v, m, w)
#define shfl_idx(v, s, w) __shfl(v, s, w)
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using bf16 = __nv_bfloat16;
using gpuError_t = cudaError_t;
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuSuccess cudaSuccess
// The mask covers one 32-lane CUDA warp. Callers only use widths 8, 16, and 32.
#define shfl_xor(v, m, w) __shfl_xor_sync(0xffffffffu, v, m, w)
#define shfl_idx(v, s, w) __shfl_sync(0xffffffffu, v, s, w)
#endif

#include "reference.h"

static void gpu_check(gpuError_t error, const char* file, int line) {
  if (error != gpuSuccess) {
    printf("[GPU ERROR] at file %s:%d:\n%s\n", file, line, gpuGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define gpuCheck(err) gpu_check(err, __FILE__, __LINE__)

// section[0, 1, 2] are the T, H, W lane counts. They sum to rotary_dim / 2.
// 32 and 64 match the Qwen3.6 (11, 11, 10) and Qwen3 (24, 20, 20) sections.
// Any other width is split as evenly as possible, with the remainder on T.
static void build_sections(int half, int section[3]) {
  if (half == 32) {
    section[0] = 11;
    section[1] = 11;
    section[2] = 10;
    return;
  }
  if (half == 64) {
    section[0] = 24;
    section[1] = 20;
    section[2] = 20;
    return;
  }
  section[1] = half / 3;
  section[2] = half / 3;
  section[0] = half - section[1] - section[2];
}

// axis[lane] is 0, 1, or 2. Sectioned layout is [TTT...HHH...WWW]. Interleaved
// layout starts as all T, then writes H at 1, 4, 7, ... and W at 2, 5, 8, ...
// stopping at 3 * section[axis]. This is the non-GLM map from SGLang
// MRotaryEmbedding._build_axis_map.
static void build_axis_map(const int section[3], int half, bool interleaved, int* axis) {
  if (!interleaved) {
    int lane = 0;
    for (int a = 0; a < 3; ++a)
      for (int n = 0; n < section[a]; ++n)
        axis[lane++] = a;
    return;
  }
  for (int lane = 0; lane < half; ++lane)
    axis[lane] = 0;
  for (int a = 1; a <= 2; ++a) {
    const int limit = 3 * section[a] < half ? 3 * section[a] : half;
    for (int lane = a; lane < limit; lane += 3)
      axis[lane] = a;
  }
}

// A uint4 load of bf16 is eight elements, stored as four little-endian pairs.
// Shifting the low 16 bits into the float exponent field recovers the value
// without a convert intrinsic. Packing rounds each float to bf16.
__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(a))) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(b))) << 16);
}

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  f[0] = bf16_lo(v.x); f[1] = bf16_hi(v.x);
  f[2] = bf16_lo(v.y); f[3] = bf16_hi(v.y);
  f[4] = bf16_lo(v.z); f[5] = bf16_hi(v.z);
  f[6] = bf16_lo(v.w); f[7] = bf16_hi(v.w);
}

// Fast path. A block has 256 lanes and covers 256 / (head_dim / 8) heads.
// Lane g of a head owns elements [8g, 8g + 8) and loads them as one uint4.
//
// NeoX pairs element i with element i + rotary_dim/2. Those two live on lanes
// g and g + (rotary_dim/2)/8, so rotary_dim must be a multiple of 16 and the
// partner is one indexed shuffle inside the head's lane group. Group widths
// are 8, 16, or 32, which fit in a CUDA warp and in one AMD wavefront.
//
// Sum of squares is reduced with xor shuffles across that same group, so the
// kernel takes no block barrier. Lanes past the last head still run the
// shuffles: they read head 0 and skip the stores. Dropping them would make
// the shuffle source undefined for the last partial group.
//
// mRoPE reads the T, H, and W cos/sin rows for this token, then selects one
// row per element with axis_map. The rows are loaded before the reduction
// because they do not depend on x. 1D RoPE reads positions[token] only.
template <int HEAD_DIM, bool MROPE>
__global__ void __launch_bounds__(256)
fused_qk_rmsnorm_rope_gate_vec(const bf16* __restrict__ q_gate,
                               const bf16* __restrict__ k,
                               bf16* __restrict__ q_out,
                               bf16* __restrict__ k_out,
                               bf16* __restrict__ gate_out,
                               const bf16* __restrict__ q_weight,
                               const bf16* __restrict__ k_weight,
                               const bf16* __restrict__ cos_sin_cache,
                               const int64_t* __restrict__ positions,
                               const int* __restrict__ axis_map,
                               int tokens, int q_heads, int kv_heads, int rotary_dim,
                               int pos_stride, float eps) {
  constexpr int VEC = 8;
  constexpr int G = HEAD_DIM / VEC;
  constexpr int HEADS_PER_BLOCK = 256 / G;

  const int g = threadIdx.x % G;
  const int nheads = q_heads + kv_heads;
  const long long total = static_cast<long long>(tokens) * nheads;
  const long long gh = static_cast<long long>(blockIdx.x) * HEADS_PER_BLOCK + threadIdx.x / G;
  const bool valid = gh < total;
  const long long ghc = valid ? gh : 0;
  const int token = static_cast<int>(ghc / nheads);
  const int head = static_cast<int>(ghc % nheads);
  const bool is_k = head >= q_heads;
  const int local = is_k ? head - q_heads : head;
  const int d0 = g * VEC;

  const bf16* in_base;
  const bf16* weight;
  bf16* out_base;
  if (is_k) {
    in_base = k + (static_cast<long long>(token) * kv_heads + local) * HEAD_DIM;
    weight = k_weight;
    out_base = k_out + (static_cast<long long>(token) * kv_heads + local) * HEAD_DIM;
  } else {
    in_base = q_gate + (static_cast<long long>(token) * q_heads + local) * 2 * HEAD_DIM;
    weight = q_weight;
    out_base = q_out + (static_cast<long long>(token) * q_heads + local) * HEAD_DIM;
  }

  const uint4 xv = *reinterpret_cast<const uint4*>(in_base + d0);
  const uint4 wv = *reinterpret_cast<const uint4*>(weight + d0);
  uint4 gv;
  if (!is_k)
    gv = *reinterpret_cast<const uint4*>(in_base + HEAD_DIM + d0);

  const int half = rotary_dim >> 1;
  const int h8 = half / VEC;
  const bool rot = d0 < rotary_dim;
  const bool first = g < h8;
  const int c0 = first ? d0 : d0 - half;

  constexpr int ROWS = MROPE ? 3 : 1;
  uint4 cv[ROWS], sv[ROWS];
  int4 a_lo, a_hi;
  if (rot) {
#pragma unroll
    for (int a = 0; a < ROWS; ++a) {
      const bf16* row = cos_sin_cache + positions[a * pos_stride + token] * rotary_dim;
      cv[a] = *reinterpret_cast<const uint4*>(row + c0);
      sv[a] = *reinterpret_cast<const uint4*>(row + half + c0);
    }
    if constexpr (MROPE) {
      a_lo = *reinterpret_cast<const int4*>(axis_map + c0);
      a_hi = *reinterpret_cast<const int4*>(axis_map + c0 + 4);
    }
  }

  float x[VEC], w[VEC];
  unpack8(xv, x);
  unpack8(wv, w);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < VEC; ++i)
    ss += x[i] * x[i];
#pragma unroll
  for (int m = G / 2; m > 0; m >>= 1)
    ss += shfl_xor(ss, m, G);
  const float inv_rms = rsqrtf(ss / static_cast<float>(HEAD_DIM) + eps);

  uint32_t n[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i)
    n[i] = pack_bf16x2(x[2 * i] * inv_rms * (w[2 * i] + 1.f),
                       x[2 * i + 1] * inv_rms * (w[2 * i + 1] + 1.f));

  const int src = rot ? (first ? g + h8 : g - h8) : g;
  uint32_t p[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i)
    p[i] = shfl_idx(n[i], src, G);

  uint4 ov = make_uint4(n[0], n[1], n[2], n[3]);
  if (rot) {
    float c[VEC], s[VEC];
    if constexpr (!MROPE) {
      unpack8(cv[0], c);
      unpack8(sv[0], s);
    } else {
      const int axes[VEC] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
      float cr[3][VEC], sr[3][VEC];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        unpack8(cv[a], cr[a]);
        unpack8(sv[a], sr[a]);
      }
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        c[i] = axes[i] == 0 ? cr[0][i] : (axes[i] == 1 ? cr[1][i] : cr[2][i]);
        s[i] = axes[i] == 0 ? sr[0][i] : (axes[i] == 1 ? sr[1][i] : sr[2][i]);
      }
    }
    uint32_t r[VEC / 2];
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
      const float me0 = bf16_lo(n[i]), me1 = bf16_hi(n[i]);
      const float pa0 = bf16_lo(p[i]), pa1 = bf16_hi(p[i]);
      float o0, o1;
      if (first) {
        o0 = me0 * c[2 * i] - pa0 * s[2 * i];
        o1 = me1 * c[2 * i + 1] - pa1 * s[2 * i + 1];
      } else {
        o0 = me0 * c[2 * i] + pa0 * s[2 * i];
        o1 = me1 * c[2 * i + 1] + pa1 * s[2 * i + 1];
      }
      r[i] = pack_bf16x2(o0, o1);
    }
    ov = make_uint4(r[0], r[1], r[2], r[3]);
  }

  if (valid) {
    *reinterpret_cast<uint4*>(out_base + d0) = ov;
    if (!is_k) {
      bf16* gate_base = gate_out + (static_cast<long long>(token) * q_heads + local) * HEAD_DIM;
      *reinterpret_cast<uint4*>(gate_base + d0) = gv;
    }
  }
}

// One thread per element, one block per (token, head). Used when rotary_dim is
// not a multiple of 16, so the partner lane is not 8-element aligned. The
// reduction is a shared-memory tree and the normalized head is staged in
// shared memory so either half of a NeoX pair can be read by any thread.
template <int HEAD_DIM, bool MROPE>
__global__ void fused_qk_rmsnorm_rope_gate(const bf16* __restrict__ q_gate,
                                           const bf16* __restrict__ k,
                                           bf16* __restrict__ q_out,
                                           bf16* __restrict__ k_out,
                                           bf16* __restrict__ gate_out,
                                           const bf16* __restrict__ q_weight,
                                           const bf16* __restrict__ k_weight,
                                           const bf16* __restrict__ cos_sin_cache,
                                           const int64_t* __restrict__ positions,
                                           const int* __restrict__ axis_map,
                                           int q_heads, int kv_heads, int rotary_dim,
                                           int pos_stride, float eps) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int idx = threadIdx.x;
  const bool is_k = head >= q_heads;
  const int local = is_k ? head - q_heads : head;

  const bf16* in_base;
  const bf16* weight;
  bf16* out_base;
  if (is_k) {
    in_base = k + (static_cast<long long>(token) * kv_heads + local) * HEAD_DIM;
    weight = k_weight;
    out_base = k_out + (static_cast<long long>(token) * kv_heads + local) * HEAD_DIM;
  } else {
    in_base = q_gate + (static_cast<long long>(token) * q_heads + local) * 2 * HEAD_DIM;
    weight = q_weight;
    out_base = q_out + (static_cast<long long>(token) * q_heads + local) * HEAD_DIM;
  }

  __shared__ float red[HEAD_DIM];
  __shared__ float normed[HEAD_DIM];

  const float x = __bfloat162float(in_base[idx]);
  red[idx] = x * x;
  __syncthreads();
  for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
    if (idx < stride)
      red[idx] += red[idx + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(red[0] / static_cast<float>(HEAD_DIM) + eps);

  const float w = __bfloat162float(weight[idx]);
  normed[idx] = __bfloat162float(__float2bfloat16(x * inv_rms * (w + 1.f)));
  __syncthreads();

  const int half = rotary_dim >> 1;
  float result = normed[idx];
  if (idx < rotary_dim) {
    const int lane = (idx < half) ? idx : (idx - half);
    const int axis = MROPE ? axis_map[lane] : 0;
    const int64_t pos = MROPE ? positions[axis * pos_stride + token] : positions[token];
    const bf16* row = cos_sin_cache + pos * rotary_dim;
    const float cos = __bfloat162float(row[lane]);
    const float sin = __bfloat162float(row[half + lane]);
    const float x1 = normed[lane];
    const float x2 = normed[half + lane];
    result = (idx < half) ? (x1 * cos - x2 * sin) : (x2 * cos + x1 * sin);
  }
  out_base[idx] = __float2bfloat16(result);

  if (!is_k) {
    bf16* gate_base = gate_out + (static_cast<long long>(token) * q_heads + local) * HEAD_DIM;
    gate_base[idx] = in_base[HEAD_DIM + idx];
  }
}

// rotary_dim % 16 == 0 selects the vector kernel. Every other even width uses
// the scalar fallback. Both specialize MROPE so the 1D path does not load an
// axis map.
template <int HEAD_DIM, bool MROPE>
static void launch(const bf16* q_gate, const bf16* k, bf16* q_out, bf16* k_out, bf16* gate_out,
                   const bf16* q_weight, const bf16* k_weight, const bf16* cache,
                   const int64_t* positions, const int* axis_map, int tokens, int q_heads,
                   int kv_heads, int rotary_dim, int pos_stride, float eps) {
  if (rotary_dim % 16 == 0) {
    constexpr int heads_per_block = 256 / (HEAD_DIM / 8);
    const long long total = static_cast<long long>(tokens) * (q_heads + kv_heads);
    const unsigned blocks = static_cast<unsigned>((total + heads_per_block - 1) / heads_per_block);
    fused_qk_rmsnorm_rope_gate_vec<HEAD_DIM, MROPE><<<blocks, 256>>>(
        q_gate, k, q_out, k_out, gate_out, q_weight, k_weight, cache, positions, axis_map,
        tokens, q_heads, kv_heads, rotary_dim, pos_stride, eps);
    return;
  }
  dim3 grid(tokens, q_heads + kv_heads);
  fused_qk_rmsnorm_rope_gate<HEAD_DIM, MROPE><<<grid, HEAD_DIM>>>(
      q_gate, k, q_out, k_out, gate_out, q_weight, k_weight, cache, positions, axis_map,
      q_heads, kv_heads, rotary_dim, pos_stride, eps);
}

// Device rsqrtf and the host libm sqrt do not match bit for bit, and the two
// reductions add the squares in a different order. 2e-2 covers that after the
// bf16 round. The gate copy is compared exactly.
static bool close_enough(float got, float ref) {
  const float atol = 2e-2f;
  const float rtol = 2e-2f;
  return fabsf(got - ref) <= atol + rtol * fabsf(ref);
}

static int run(int tokens, int q_heads, int kv_heads, int head_dim, int rotary_dim, int mode,
               int repeat) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || repeat <= 0) {
    printf("tokens, heads, and repeat must be positive\n");
    return 1;
  }
  if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
    printf("head_dim must be 64, 128, or 256\n");
    return 1;
  }
  if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) {
    printf("rotary_dim must be even and in [2, head_dim]\n");
    return 1;
  }
  if (mode < 0 || mode > 2) {
    printf("mode must be 0 (1D), 1 (sectioned mRoPE), or 2 (interleaved mRoPE)\n");
    return 1;
  }

  const bool mrope = mode != 0;
  const bool interleaved = mode == 2;
  const int half = rotary_dim / 2;
  const float eps = 1e-6f;
  const int max_pos = 512;
  const int q_pack = 2;

  int section[3] = {half, 0, 0};
  std::vector<int> axis_map(half, 0);
  if (mrope) {
    build_sections(half, section);
    if (section[0] + section[1] + section[2] != half) {
      printf("mrope sections %d %d %d do not sum to %d\n", section[0], section[1], section[2],
             half);
      return 1;
    }
    build_axis_map(section, half, interleaved, axis_map.data());
  }

  const size_t q_in_n = static_cast<size_t>(tokens) * q_heads * q_pack * head_dim;
  const size_t k_n = static_cast<size_t>(tokens) * kv_heads * head_dim;
  const size_t q_n = static_cast<size_t>(tokens) * q_heads * head_dim;
  const size_t pos_n = mrope ? static_cast<size_t>(3) * tokens : static_cast<size_t>(tokens);
  const size_t cache_n = static_cast<size_t>(max_pos) * rotary_dim;

  std::vector<bf16> h_q(q_in_n), h_k(k_n), h_qw(head_dim), h_kw(head_dim), h_cache(cache_n);
  std::vector<bf16> h_qo(q_n), h_ko(k_n), h_gate(q_n);
  std::vector<int64_t> h_pos(pos_n);

  srand(0);
  auto rnd = []() { return rand() / static_cast<float>(RAND_MAX) * 2.f - 1.f; };
  for (size_t i = 0; i < q_in_n; ++i)
    h_q[i] = __float2bfloat16(rnd());
  for (size_t i = 0; i < k_n; ++i)
    h_k[i] = __float2bfloat16(rnd());
  for (int i = 0; i < head_dim; ++i) {
    h_qw[i] = __float2bfloat16(rnd());
    h_kw[i] = __float2bfloat16(rnd());
  }
  for (int p = 0; p < max_pos; ++p) {
    for (int lane = 0; lane < half; ++lane) {
      const float angle = (p + 1) * (lane + 1) * 0.01f;
      h_cache[p * rotary_dim + lane] = __float2bfloat16(cosf(angle));
      h_cache[p * rotary_dim + half + lane] = __float2bfloat16(sinf(angle));
    }
  }
  for (int t = 0; t < tokens; ++t) {
    if (!mrope) {
      h_pos[t] = (t * 3 + 1) % max_pos;
    } else {
      h_pos[t] = (t * 3 + 1) % max_pos;
      h_pos[tokens + t] = (t * 5 + 4) % max_pos;
      h_pos[2 * tokens + t] = (t * 7 + 9) % max_pos;
    }
  }

  bf16 *d_q, *d_k, *d_qw, *d_kw, *d_cache, *d_qo, *d_ko, *d_gate;
  int64_t* d_pos;
  int* d_axis;
  gpuCheck(gpuMalloc(&d_q, q_in_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_k, k_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_qw, head_dim * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_kw, head_dim * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_cache, cache_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_qo, q_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_ko, k_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_gate, q_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_pos, pos_n * sizeof(int64_t)));
  gpuCheck(gpuMalloc(&d_axis, half * sizeof(int)));
  gpuCheck(gpuMemcpy(d_q, h_q.data(), q_in_n * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_k, h_k.data(), k_n * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_qw, h_qw.data(), head_dim * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_kw, h_kw.data(), head_dim * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_cache, h_cache.data(), cache_n * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_pos, h_pos.data(), pos_n * sizeof(int64_t), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_axis, axis_map.data(), half * sizeof(int), gpuMemcpyHostToDevice));

  const int pos_stride = tokens;
  auto launch_mode = [&]() {
    if (head_dim == 64 && !mrope)
      launch<64, false>(d_q, d_k, d_qo, d_ko, d_gate, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                        q_heads, kv_heads, rotary_dim, pos_stride, eps);
    else if (head_dim == 64 && mrope)
      launch<64, true>(d_q, d_k, d_qo, d_ko, d_gate, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                       q_heads, kv_heads, rotary_dim, pos_stride, eps);
    else if (head_dim == 128 && !mrope)
      launch<128, false>(d_q, d_k, d_qo, d_ko, d_gate, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                         q_heads, kv_heads, rotary_dim, pos_stride, eps);
    else if (head_dim == 128 && mrope)
      launch<128, true>(d_q, d_k, d_qo, d_ko, d_gate, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                        q_heads, kv_heads, rotary_dim, pos_stride, eps);
    else if (head_dim == 256 && !mrope)
      launch<256, false>(d_q, d_k, d_qo, d_ko, d_gate, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                         q_heads, kv_heads, rotary_dim, pos_stride, eps);
    else
      launch<256, true>(d_q, d_k, d_qo, d_ko, d_gate, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                        q_heads, kv_heads, rotary_dim, pos_stride, eps);
  };

  launch_mode();
  gpuCheck(gpuGetLastError());
  gpuCheck(gpuDeviceSynchronize());
  gpuCheck(gpuMemcpy(h_qo.data(), d_qo, q_n * sizeof(bf16), gpuMemcpyDeviceToHost));
  gpuCheck(gpuMemcpy(h_ko.data(), d_ko, k_n * sizeof(bf16), gpuMemcpyDeviceToHost));
  gpuCheck(gpuMemcpy(h_gate.data(), d_gate, q_n * sizeof(bf16), gpuMemcpyDeviceToHost));

  int nfaults = 0;
  std::vector<float> ref(head_dim);
  for (int t = 0; t < tokens && nfaults < 8; ++t) {
    for (int h = 0; h < q_heads && nfaults < 8; ++h) {
      const bf16* in = h_q.data() + (static_cast<long long>(t) * q_heads + h) * q_pack * head_dim;
      gemma_rmsnorm_neox_host(ref.data(), in, h_qw.data(), head_dim, rotary_dim, h_cache.data(),
                              h_pos.data(), axis_map.data(), pos_stride, t, mrope, eps);
      const bf16* got = h_qo.data() + (static_cast<long long>(t) * q_heads + h) * head_dim;
      const bf16* gate = h_gate.data() + (static_cast<long long>(t) * q_heads + h) * head_dim;
      for (int i = 0; i < head_dim; ++i) {
        const float gv = __bfloat162float(got[i]);
        if (!close_enough(gv, ref[i])) {
          printf("Mismatch of q at token %d head %d dim %d: CPU_ref %f vs GPU %f\n", t, h, i,
                 ref[i], gv);
          if (++nfaults >= 8)
            break;
        }
        const float gate_v = __bfloat162float(gate[i]);
        const float gate_ref = __bfloat162float(in[head_dim + i]);
        if (gate_v != gate_ref) {
          printf("Mismatch of gate at token %d head %d dim %d: CPU_ref %f vs GPU %f\n", t, h, i,
                 gate_ref, gate_v);
          if (++nfaults >= 8)
            break;
        }
      }
    }
    for (int h = 0; h < kv_heads && nfaults < 8; ++h) {
      const bf16* in = h_k.data() + (static_cast<long long>(t) * kv_heads + h) * head_dim;
      gemma_rmsnorm_neox_host(ref.data(), in, h_kw.data(), head_dim, rotary_dim, h_cache.data(),
                              h_pos.data(), axis_map.data(), pos_stride, t, mrope, eps);
      const bf16* got = h_ko.data() + (static_cast<long long>(t) * kv_heads + h) * head_dim;
      for (int i = 0; i < head_dim; ++i) {
        const float gv = __bfloat162float(got[i]);
        if (!close_enough(gv, ref[i])) {
          printf("Mismatch of k at token %d head %d dim %d: CPU_ref %f vs GPU %f\n", t, h, i,
                 ref[i], gv);
          if (++nfaults >= 8)
            break;
        }
      }
    }
  }
  if (nfaults) {
    printf("Verification failed\n");
    return 1;
  }
  printf("All results match.\n");

  gpuCheck(gpuDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    launch_mode();
  gpuCheck(gpuGetLastError());
  gpuCheck(gpuDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(stop - start).count() / repeat;

  // Bytes that must move once per token: packed Q+gate and K in, then Q, K,
  // and gate out. Weights are one head vector reused by every head, and each
  // cos/sin row is reused by every head of the token, so neither is counted.
  const size_t bytes = (q_in_n + k_n + q_n + k_n + q_n) * sizeof(bf16);
  const double gbps = (bytes / ms) / 1e6;

  const char* mode_name = mode == 0 ? "1d" : mode == 1 ? "sectioned" : "interleaved";
  printf("tokens %d q_heads %d kv_heads %d head_dim %d rotary_dim %d mode %s", tokens, q_heads,
         kv_heads, head_dim, rotary_dim, mode_name);
  if (mrope)
    printf(" sections %d %d %d", section[0], section[1], section[2]);
  printf(" | bandwidth %.2f GB/s\n", gbps);
  printf("Average kernel execution time: %f (ms)\n", ms);

  gpuCheck(gpuFree(d_q));
  gpuCheck(gpuFree(d_k));
  gpuCheck(gpuFree(d_qw));
  gpuCheck(gpuFree(d_kw));
  gpuCheck(gpuFree(d_cache));
  gpuCheck(gpuFree(d_qo));
  gpuCheck(gpuFree(d_ko));
  gpuCheck(gpuFree(d_gate));
  gpuCheck(gpuFree(d_pos));
  gpuCheck(gpuFree(d_axis));
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    printf("Usage: %s <tokens> <q_heads> <kv_heads> <head_dim> <rotary_dim> <mode> <repeat>\n",
           argv[0]);
    printf("  mode 0: 1D RoPE\n");
    printf("  mode 1: sectioned mRoPE (T then H then W)\n");
    printf("  mode 2: interleaved mRoPE\n");
    return 1;
  }
  return run(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
             atoi(argv[6]), atoi(argv[7]));
}
