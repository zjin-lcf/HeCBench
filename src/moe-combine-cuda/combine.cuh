// Intra-node MoE combine, bf16 expert outputs, fp32 accumulation.
// The peer path reads expert outputs through IPC-mapped pointers (the MORI
// P2P-read combine). A grid barrier, then a system-scope flag per peer, makes
// every rank wait until the others have entered the kernel before those reads.
// The MPI path reads local replicas and skips that barrier; the host exchange
// is the synchronization.

#pragma once

#include <cstdint>
#include "reference.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#if defined(__HIPCC__)
#define MC_HD __host__ __device__
__device__ inline void mc_store_u64(unsigned long long *p, unsigned long long v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline unsigned long long mc_load_u64(const unsigned long long *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void mc_pause() { __builtin_amdgcn_s_sleep(4); }
#else
#define MC_HD __host__ __device__
__device__ inline void mc_store_u64(unsigned long long *p, unsigned long long v) {
  atomicExch_system(p, v);
}
__device__ inline unsigned long long mc_load_u64(const unsigned long long *p) {
  return atomicAdd_system(const_cast<unsigned long long *>(p), 0ull);
}
__device__ inline void mc_pause() { __nanosleep(64); }
#endif

MC_HD inline float mc_bf16_to_f32(std::uint16_t b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(static_cast<unsigned>(b) << 16);
#else
  return bf16_to_f32(b);
#endif
}

MC_HD inline std::uint16_t mc_f32_to_bf16(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const unsigned bits = __float_as_uint(x);
  const unsigned lsb = (bits >> 16) & 1u;
  return static_cast<std::uint16_t>((bits + 0x7fffu + lsb) >> 16);
#else
  return f32_to_bf16(x);
#endif
}

__device__ inline void load8(const std::uint16_t *p, float *o) {
  const uint4 v = *reinterpret_cast<const uint4 *>(p);
  const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    o[2 * i] = mc_bf16_to_f32(static_cast<std::uint16_t>(w[i]));
    o[2 * i + 1] = mc_bf16_to_f32(static_cast<std::uint16_t>(w[i] >> 16));
  }
}

__device__ inline void store8(std::uint16_t *p, const float *a) {
  unsigned w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned lo = mc_f32_to_bf16(a[2 * i]);
    const unsigned hi = mc_f32_to_bf16(a[2 * i + 1]);
    w[i] = lo | (hi << 16);
  }
  *reinterpret_cast<uint4 *>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

// arrival[0] counts blocks, arrival[1] publishes the phase they should observe.
__device__ void grid_sync(unsigned *arrival, unsigned phase) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned ticket = atomicAdd(arrival, 1u);
    if (ticket + 1u == static_cast<unsigned>(gridDim.x)) {
      atomicExch(arrival, 0u);
      atomicExch(arrival + 1, phase);
    } else {
      while (atomicAdd(arrival + 1, 0u) != phase)
        mc_pause();
    }
  }
  __syncthreads();
}

// One thread talks to the other GPUs. Every block waits on the local grid sync
// so the rest of the kernel cannot issue remote reads early, and so the flag
// spin does not flood the interconnect.
__device__ void cross_device_barrier(unsigned *arrival,
                                    unsigned long long *local_flags,
                                    unsigned long long *const *peer_flags,
                                    int rank, int world,
                                    unsigned long long epoch) {
  const unsigned phase = static_cast<unsigned>(epoch) * 2u;
  __threadfence_system();
  grid_sync(arrival, phase);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int peer = 0; peer < world; ++peer)
      mc_store_u64(peer_flags[peer] + rank, epoch);
    __threadfence_system();
    for (int peer = 0; peer < world; ++peer)
      while (mc_load_u64(local_flags + peer) != epoch)
        mc_pause();
    __threadfence_system();
  }
  grid_sync(arrival, phase + 1u);
}

template <int kTopk, bool kPeer>
__global__ void __launch_bounds__(256)
    combine_kernel(int tokens, int hidden, int world, int rank,
                   const int *__restrict__ dest, const int *__restrict__ slot,
                   const std::uint16_t *const *__restrict__ stage,
                   std::uint16_t *__restrict__ out, unsigned *arrival,
                   unsigned long long *local_flags,
                   unsigned long long *const *peer_flags,
                   unsigned long long epoch) {
  if constexpr (kPeer)
    cross_device_barrier(arrival, local_flags, peer_flags, rank, world, epoch);

  const int nvec = hidden >> 3;
  for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
    for (int vec = threadIdx.x; vec < nvec; vec += blockDim.x) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const int *td = dest + static_cast<std::size_t>(token) * kTopk;
      const int *ts = slot + static_cast<std::size_t>(token) * kTopk;
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        const std::uint16_t *src =
            stage[td[k]] +
            (static_cast<std::size_t>(ts[k]) * hidden +
             (static_cast<std::size_t>(vec) << 3));
        float x[8];
        load8(src, x);
        const float w = weight_of(token, k, kTopk);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[j] += w * x[j];
      }
      store8(out + static_cast<std::size_t>(token) * hidden +
                 (static_cast<std::size_t>(vec) << 3),
             acc);
    }
  }
}

__global__ void fill_stage(std::uint16_t *stage, int rank, int slots, int hidden) {
  const std::size_t n = static_cast<std::size_t>(slots) * hidden;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const int slot = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<std::size_t>(slot) * hidden);
    stage[i] = mc_f32_to_bf16(stage_unit(rank, slot, h));
  }
}

// Same increment the pingpong benchmark uses to prove a device buffer survived
// a GPU-aware MPI round trip.
__global__ void mpi_check_add(double *d, long n) {
  for (long i = static_cast<long>(blockDim.x) * blockIdx.x + threadIdx.x; i < n;
       i += static_cast<long>(blockDim.x) * gridDim.x)
    d[i] = d[i] + 1.0;
}
