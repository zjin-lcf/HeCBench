/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted from Meta Superintelligence Labs Kernels (MSLK) index_shuffling:
 * https://github.com/meta-pytorch/MSLK/blob/main/csrc/moe/index_shuffling.cu
 *
 * Cooperative top-k + histogram + expert-order index placement for MoE routing
 * (MetaShuffling IndexShuffling).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d '%s': %s\n", __FILE__, __LINE__,      \
              #call, cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int kNumThreadsPerWarp = 32;

__host__ __device__ __forceinline__ constexpr int ceil_of_ratio(int a, int b) {
  return (a + b - 1) / b;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
__device__ __forceinline__ int atomic_add_relaxed(int *addr, int inc) {
  int val;
  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(val)
               : "l"(addr), "r"(inc));
  return val;
}

__device__ __forceinline__ int atomic_add_release(int *addr, int inc) {
  int val;
  asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(val)
               : "l"(addr), "r"(inc));
  return val;
}

__device__ __forceinline__ int load_acquire(int *addr) {
  int val;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(val)
               : "l"(addr));
  return val;
}
#else
__device__ __forceinline__ int atomic_add_relaxed(int *addr, int inc) {
  return atomicAdd(addr, inc);
}

__device__ __forceinline__ int atomic_add_release(int *addr, int inc) {
  int val = atomicAdd(addr, inc);
  __threadfence();
  return val;
}

__device__ __forceinline__ int load_acquire(int *addr) {
  __threadfence();
  return *addr;
}
#endif

template <class DataType, class IndexType, int NumExperts, int NumTokensPerTile,
          int TopK>
struct SharedStorage {
  DataType routing_scores[NumTokensPerTile * NumExperts * TopK];
  IndexType expert_indices[NumTokensPerTile * NumExperts * TopK];
  IndexType token_count_cumsums[NumExperts];
};

template <class DataType, class IndexType>
struct Params {
  const DataType *routing_scores;
  int stride_t;
  int stride_e;
  int expert_index_start;
  int expert_index_end;
  const IndexType *valid_token_count;
  int num_tokens;
  int num_tokens_per_cta;

  IndexType *buffered_expert_indices;
  IndexType *buffered_token_indices;

  IndexType *token_count_per_expert;
  IndexType *shuffled_expert_indices;
  IndexType *shuffled_token_indices;
};

template <class DataType, class IndexType>
__device__ __forceinline__ void merge_top1(DataType *routing_scores,
                                           IndexType *expert_indices,
                                           int lhs_smem_index,
                                           int rhs_smem_index) {
  auto lhs_score = routing_scores[lhs_smem_index];
  auto rhs_score = routing_scores[rhs_smem_index];
  auto lhs_expert_index = expert_indices[lhs_smem_index];
  auto rhs_expert_index = expert_indices[rhs_smem_index];

  bool lhs_larger = lhs_score >= rhs_score;
  routing_scores[lhs_smem_index] = lhs_larger ? lhs_score : rhs_score;
  expert_indices[lhs_smem_index] =
      lhs_larger ? lhs_expert_index : rhs_expert_index;
}

template <class DataType, class IndexType>
__device__ __forceinline__ void
merge_top2(DataType *routing_scores, IndexType *expert_indices,
           int lhs_smem_index, int rhs_smem_index, bool skip_duplicates) {
  auto lhs_score0 = routing_scores[lhs_smem_index];
  auto lhs_score1 = routing_scores[lhs_smem_index + 1];
  auto rhs_score0 = routing_scores[rhs_smem_index];
  auto rhs_score1 = routing_scores[rhs_smem_index + 1];
  auto lhs_expert_index0 = expert_indices[lhs_smem_index];
  auto lhs_expert_index1 = expert_indices[lhs_smem_index + 1];
  auto rhs_expert_index0 = expert_indices[rhs_smem_index];
  auto rhs_expert_index1 = expert_indices[rhs_smem_index + 1];

  if (lhs_score0 >= rhs_score0) {
    routing_scores[lhs_smem_index] = lhs_score0;
    expert_indices[lhs_smem_index] = lhs_expert_index0;
    if ((lhs_score1 >= rhs_score0) && !skip_duplicates) {
      routing_scores[lhs_smem_index + 1] = lhs_score1;
      expert_indices[lhs_smem_index + 1] = lhs_expert_index1;
    } else {
      routing_scores[lhs_smem_index + 1] = rhs_score0;
      expert_indices[lhs_smem_index + 1] = rhs_expert_index0;
    }
  } else {
    routing_scores[lhs_smem_index] = rhs_score0;
    expert_indices[lhs_smem_index] = rhs_expert_index0;
    if ((rhs_score1 > lhs_score0) && !skip_duplicates) {
      routing_scores[lhs_smem_index + 1] = rhs_score1;
      expert_indices[lhs_smem_index + 1] = rhs_expert_index1;
    } else {
      routing_scores[lhs_smem_index + 1] = lhs_score0;
      expert_indices[lhs_smem_index + 1] = lhs_expert_index0;
    }
  }
}

template <class DataType, class IndexType, int K>
__device__ __forceinline__ void
merge_topk(DataType *routing_scores, IndexType *expert_indices,
           int lhs_smem_index, int rhs_smem_index, int num_valid_values) {
  DataType merged_scores[K];
  IndexType merged_indices[K];

  DataType *lhs_scores = &routing_scores[lhs_smem_index];
  DataType *rhs_scores = &routing_scores[rhs_smem_index];
  IndexType *lhs_indices = &expert_indices[lhs_smem_index];
  IndexType *rhs_indices = &expert_indices[rhs_smem_index];

  int lhs_idx = 0;
  int rhs_idx = 0;
  int merged_idx = 0;
  int max_from_one_array = min(num_valid_values, K);

  while (merged_idx < K) {
    if (lhs_idx >= max_from_one_array) {
      merged_scores[merged_idx] = rhs_scores[rhs_idx];
      merged_indices[merged_idx] = rhs_indices[rhs_idx];
      rhs_idx++;
      merged_idx++;
      continue;
    }
    if (rhs_idx >= max_from_one_array) {
      merged_scores[merged_idx] = lhs_scores[lhs_idx];
      merged_indices[merged_idx] = lhs_indices[lhs_idx];
      lhs_idx++;
      merged_idx++;
      continue;
    }
    if (lhs_scores[lhs_idx] >= rhs_scores[rhs_idx]) {
      merged_scores[merged_idx] = lhs_scores[lhs_idx];
      merged_indices[merged_idx] = lhs_indices[lhs_idx];
      lhs_idx++;
    } else {
      merged_scores[merged_idx] = rhs_scores[rhs_idx];
      merged_indices[merged_idx] = rhs_indices[rhs_idx];
      rhs_idx++;
    }
    merged_idx++;
  }

  for (int i = 0; i < K; i++) {
    routing_scores[lhs_smem_index + i] = merged_scores[i];
    expert_indices[lhs_smem_index + i] = merged_indices[i];
  }
}

template <class DataType, class IndexType, int NumExperts, int NumTokensPerTile,
          int TopK>
__global__ void index_shuffling_kernel(Params<DataType, IndexType> params) {
  constexpr int kNumWarps = (NumExperts > 128) ? 8 : 4;
  constexpr int kNumThreads = kNumThreadsPerWarp * kNumWarps;

  extern __shared__ char shared_memory[];
  auto &smem = *reinterpret_cast<
      SharedStorage<DataType, IndexType, NumExperts, NumTokensPerTile, TopK> *>(
      shared_memory);

  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;

  const int num_total_tokens = params.num_tokens;
  const int token_index_start = bidx * params.num_tokens_per_cta;
  if (token_index_start >= num_total_tokens) {
    return;
  }

  const int token_index_end =
      min(token_index_start + params.num_tokens_per_cta, num_total_tokens);
  const int num_valid_tokens =
      params.valid_token_count ? *params.valid_token_count : num_total_tokens;

  const int expert_index_start = params.expert_index_start;
  const int expert_index_end = params.expert_index_end;

  const int stride_t = params.stride_t;
  const int stride_e = params.stride_e;

  const DataType zero = static_cast<DataType>(-INFINITY);
  for (int token_index_offset = token_index_start;
       token_index_offset < token_index_end;
       token_index_offset += NumTokensPerTile) {
#pragma unroll
    for (int i = tidx; i < NumTokensPerTile * NumExperts; i += kNumThreads) {
      const int token_index = token_index_offset + i / NumExperts;
      const int expert_index = i % NumExperts;
#pragma unroll
      for (int j = 0; j < TopK; j++) {
        smem.routing_scores[i * TopK + j] =
            (token_index < num_valid_tokens)
                ? params.routing_scores[token_index * stride_t +
                                        expert_index * stride_e]
                : zero;
        smem.expert_indices[i * TopK + j] = expert_index;
      }
    }
    __syncthreads();

    static_assert(NumExperts % 2 == 0, "");
    constexpr int kNumParallelReductionThreads =
        (NumExperts > 128) ? 256 : (NumExperts / 2);
    static_assert(kNumThreads % kNumParallelReductionThreads == 0, "");
    constexpr int kNumParallelReductionGroups =
        kNumThreads / kNumParallelReductionThreads;
    static_assert(NumTokensPerTile % kNumParallelReductionGroups == 0, "");

    for (int num_reduced_threads = 1;
         num_reduced_threads <= kNumParallelReductionThreads;
         num_reduced_threads <<= 1) {
#pragma unroll
      for (int local_token_offset = 0; local_token_offset < NumTokensPerTile;
           local_token_offset += kNumParallelReductionGroups) {
        if (!(tidx & (num_reduced_threads - 1))) {
          int local_token_index =
              local_token_offset + tidx / kNumParallelReductionThreads;
          int lhs_expert_offset = (tidx % kNumParallelReductionThreads) * 2;
          int lhs_smem_index =
              (local_token_index * NumExperts + lhs_expert_offset) * TopK;
          int rhs_smem_index = lhs_smem_index + num_reduced_threads * TopK;
          if (lhs_expert_offset + num_reduced_threads < NumExperts) {
            if (TopK == 1) {
              merge_top1(smem.routing_scores, smem.expert_indices,
                         lhs_smem_index, rhs_smem_index);
            } else if (TopK == 2) {
              merge_top2(smem.routing_scores, smem.expert_indices,
                         lhs_smem_index, rhs_smem_index,
                         num_reduced_threads == 1);
            } else {
              merge_topk<DataType, IndexType, 4>(
                  smem.routing_scores, smem.expert_indices, lhs_smem_index,
                  rhs_smem_index, num_reduced_threads);
            }
          }
        }
        if (TopK > 1) {
          __syncthreads();
        }
      }
      if constexpr (kNumParallelReductionThreads <= kNumThreadsPerWarp) {
        __syncwarp();
      } else {
        __syncthreads();
      }
    }
    if constexpr (kNumParallelReductionThreads > kNumThreadsPerWarp) {
      __syncthreads();
    }

#pragma unroll
    for (int i = tidx; i < NumTokensPerTile; i += kNumThreads) {
      int local_token_index = i;
      int token_index = token_index_offset + i;
      if (token_index < num_valid_tokens) {
#pragma unroll
        for (int j = 0; j < TopK; j++) {
          auto expert_index =
              smem.expert_indices[local_token_index * NumExperts * TopK + j];
          params.buffered_expert_indices[token_index * TopK + j] = expert_index;
          if (expert_index >= expert_index_start &&
              expert_index < expert_index_end) {
            auto token_index_in_expert = atomic_add_relaxed(
                &params.token_count_per_expert[expert_index], 1);
            params.buffered_token_indices[token_index * TopK + j] =
                token_index_in_expert;
          }
        }
      }
    }
    __syncthreads();
  }

  if (tidx == 0) {
    int processed_tokens = 0;
    int *processed_tokens_addr = &params.token_count_per_expert[NumExperts];
    int inc = token_index_end - token_index_start;
    atomic_add_release(processed_tokens_addr, inc);
    do {
      processed_tokens = load_acquire(processed_tokens_addr);
    } while (processed_tokens != num_total_tokens);
  }
  __syncthreads();

  for (int i = tidx; i < NumExperts; i += kNumThreads) {
    smem.token_count_cumsums[i] = params.token_count_per_expert[i];
  }
  __syncthreads();

  if (tidx == 0) {
#pragma unroll
    for (int i = 1; i < NumExperts; ++i) {
      smem.token_count_cumsums[i] += smem.token_count_cumsums[i - 1];
    }
  }
  __syncthreads();

  auto get_token_count_cumsum = [&smem](int index) {
    return index == 0 ? 0 : smem.token_count_cumsums[index - 1];
  };

  const int token_count_cumsum_start =
      get_token_count_cumsum(expert_index_start);
  const int token_count_cumsum_end = get_token_count_cumsum(expert_index_end);
  const int num_selected_tokens =
      token_count_cumsum_end - token_count_cumsum_start;

  for (int global_token_offset = bidx * params.num_tokens_per_cta;
       global_token_offset < (bidx + 1) * params.num_tokens_per_cta;
       global_token_offset += kNumThreads) {
    int token_index = global_token_offset + tidx;
    if (token_index < num_valid_tokens) {
#pragma unroll
      for (int j = 0; j < TopK; j++) {
        int expert_index =
            params.buffered_expert_indices[token_index * TopK + j];
        if (expert_index >= expert_index_start &&
            expert_index < expert_index_end) {
          int new_token_index_in_expert =
              params.buffered_token_indices[token_index * TopK + j];
          int new_token_index = get_token_count_cumsum(expert_index) -
                                token_count_cumsum_start +
                                new_token_index_in_expert;
          params.shuffled_expert_indices[new_token_index] =
              expert_index - expert_index_start;
          params.shuffled_token_indices[new_token_index] = token_index;
        }
      }
    }
  }

  if (tidx == 0 && bidx == 0) {
    params.token_count_per_expert[NumExperts + 1] = num_selected_tokens;
  }
}

template <int NumExperts, int NumTokensPerTile, int TopK>
void launch_index_shuffling(const float *d_scores, int num_tokens,
                            int *d_token_count_per_expert,
                            int *d_shuffled_expert_indices,
                            int *d_shuffled_token_indices,
                            int *d_buffered_expert_indices,
                            int *d_buffered_token_indices, int num_sms,
                            cudaStream_t stream) {
  using DataType = float;
  using IndexType = int;

  constexpr int kNumWarps = (NumExperts > 128) ? 8 : 4;
  constexpr int kNumThreads = kNumThreadsPerWarp * kNumWarps;
  const int smem_size = (int)sizeof(
      SharedStorage<DataType, IndexType, NumExperts, NumTokensPerTile, TopK>);

  const int num_tiles = ceil_of_ratio(num_tokens, NumTokensPerTile);
  const int num_ctas = std::min(num_tiles, std::max(num_sms, 1));
  const int num_tiles_per_cta =
      ceil_of_ratio(ceil_of_ratio(num_tokens, num_ctas), NumTokensPerTile);
  const int num_tokens_per_cta = num_tiles_per_cta * NumTokensPerTile;

  Params<DataType, IndexType> params;
  params.routing_scores = d_scores;
  params.stride_t = NumExperts;
  params.stride_e = 1;
  params.expert_index_start = 0;
  params.expert_index_end = NumExperts;
  params.valid_token_count = nullptr;
  params.num_tokens = num_tokens;
  params.num_tokens_per_cta = num_tokens_per_cta;
  params.buffered_expert_indices = d_buffered_expert_indices;
  params.buffered_token_indices = d_buffered_token_indices;
  params.token_count_per_expert = d_token_count_per_expert;
  params.shuffled_expert_indices = d_shuffled_expert_indices;
  params.shuffled_token_indices = d_shuffled_token_indices;

  void *kernel = (void *)index_shuffling_kernel<DataType, IndexType, NumExperts,
                                                NumTokensPerTile, TopK>;
  if (smem_size >= 48 * 1024) {
    CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  dim3 grids(num_ctas);
  dim3 blocks(kNumThreads);
  void *args[] = {(void *)&params};

  cudaError_t launch_status = cudaLaunchCooperativeKernel(
      kernel, grids, blocks, args, smem_size, stream);
  if (launch_status == cudaErrorCooperativeLaunchTooLarge ||
      launch_status == cudaErrorNotSupported) {
    CHECK(cudaLaunchKernel(kernel, grids, blocks, args, smem_size, stream));
  } else if (launch_status != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d cudaLaunchCooperativeKernel: %s\n",
            __FILE__, __LINE__, cudaGetErrorString(launch_status));
    exit(EXIT_FAILURE);
  }
}

inline int tokens_per_tile_for(int num_experts, int top_k) {
  constexpr int kBase = 16;
  const int storage_factor = top_k * num_experts;
  if (storage_factor <= 128)
    return kBase;
  if (storage_factor <= 256)
    return kBase / 2;
  if (storage_factor <= 512)
    return kBase / 4;
  return kBase / 8;
}

inline void dispatch_index_shuffling(const float *d_scores, int num_tokens,
                                     int num_experts, int top_k,
                                     int *d_token_count_per_expert,
                                     int *d_shuffled_expert_indices,
                                     int *d_shuffled_token_indices,
                                     int *d_buffered_expert_indices,
                                     int *d_buffered_token_indices, int num_sms,
                                     cudaStream_t stream) {
  if (!(top_k == 1 || top_k == 2 || top_k == 4)) {
    fprintf(stderr, "Unsupported top_k=%d (supported: 1, 2, 4)\n", top_k);
    exit(EXIT_FAILURE);
  }
  if (!(num_experts == 16 || num_experts == 32 || num_experts == 128)) {
    fprintf(stderr,
            "Unsupported num_experts=%d (supported: 16, 32, 128)\n",
            num_experts);
    exit(EXIT_FAILURE);
  }

  const int tile = tokens_per_tile_for(num_experts, top_k);

#define LAUNCH(E, B, K)                                                        \
  launch_index_shuffling<E, B, K>(                                             \
      d_scores, num_tokens, d_token_count_per_expert,                          \
      d_shuffled_expert_indices, d_shuffled_token_indices,                     \
      d_buffered_expert_indices, d_buffered_token_indices, num_sms, stream)

  if (num_experts == 16) {
    if (top_k == 1)
      LAUNCH(16, 16, 1);
    else if (top_k == 2)
      LAUNCH(16, 16, 2);
    else
      LAUNCH(16, 16, 4);
  } else if (num_experts == 32) {
    if (top_k == 1)
      LAUNCH(32, 16, 1);
    else if (top_k == 2)
      LAUNCH(32, 16, 2);
    else
      LAUNCH(32, 16, 4);
  } else { // 128
    if (top_k == 1) {
      if (tile == 16)
        LAUNCH(128, 16, 1);
      else if (tile == 8)
        LAUNCH(128, 8, 1);
      else if (tile == 4)
        LAUNCH(128, 4, 1);
      else
        LAUNCH(128, 2, 1);
    } else if (top_k == 2) {
      if (tile == 16)
        LAUNCH(128, 16, 2);
      else if (tile == 8)
        LAUNCH(128, 8, 2);
      else if (tile == 4)
        LAUNCH(128, 4, 2);
      else
        LAUNCH(128, 2, 2);
    } else {
      if (tile == 16)
        LAUNCH(128, 16, 4);
      else if (tile == 8)
        LAUNCH(128, 8, 4);
      else if (tile == 4)
        LAUNCH(128, 4, 4);
      else
        LAUNCH(128, 2, 4);
    }
  }
#undef LAUNCH
}
