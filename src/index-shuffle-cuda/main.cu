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
 * Benchmark for MetaShuffling IndexShuffling: fused top-k, per-expert
 * histogram, and expert-order token-index placement used by Llama 4 MoE.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.h"
#include "reference.h"

static void run_case(int num_tokens, int num_experts, int top_k, int repeat) {
  const int total = num_tokens * top_k;
  const size_t scores_bytes = (size_t)num_tokens * num_experts * sizeof(float);
  const size_t index_bytes = (size_t)total * sizeof(int);
  const size_t counts_bytes = (size_t)(num_experts + 2) * sizeof(int);

  std::vector<float> h_scores(num_tokens * num_experts);
  std::mt19937 gen(19937);
  // Unique scores per token avoid argmax ties so host top-k matches the device
  // reduction on equal values.
  for (int t = 0; t < num_tokens; ++t) {
    std::vector<float> row(num_experts);
    for (int e = 0; e < num_experts; ++e)
      row[e] = (float)e;
    std::shuffle(row.begin(), row.end(), gen);
    for (int e = 0; e < num_experts; ++e)
      h_scores[t * num_experts + e] = row[e] + 0.01f * (float)t;
  }

  float *d_scores = nullptr;
  int *d_counts = nullptr;
  int *d_shuffled_experts = nullptr;
  int *d_shuffled_tokens = nullptr;
  int *d_buf_experts = nullptr;
  int *d_buf_tokens = nullptr;

  CHECK(cudaMalloc(&d_scores, scores_bytes));
  CHECK(cudaMalloc(&d_counts, counts_bytes));
  CHECK(cudaMalloc(&d_shuffled_experts, index_bytes));
  CHECK(cudaMalloc(&d_shuffled_tokens, index_bytes));
  CHECK(cudaMalloc(&d_buf_experts, index_bytes));
  CHECK(cudaMalloc(&d_buf_tokens, index_bytes));
  CHECK(cudaMemcpy(d_scores, h_scores.data(), scores_bytes,
                   cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;
  cudaStream_t stream = 0;

  CHECK(cudaMemset(d_counts, 0, counts_bytes));
  dispatch_index_shuffling(d_scores, num_tokens, num_experts, top_k, d_counts,
                           d_shuffled_experts, d_shuffled_tokens, d_buf_experts,
                           d_buf_tokens, num_sms, stream);
  CHECK(cudaDeviceSynchronize());

  std::vector<int> h_counts(num_experts + 2);
  std::vector<int> h_shuffled_experts(total);
  std::vector<int> h_shuffled_tokens(total);
  CHECK(cudaMemcpy(h_counts.data(), d_counts, counts_bytes,
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_shuffled_experts.data(), d_shuffled_experts, index_bytes,
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_shuffled_tokens.data(), d_shuffled_tokens, index_bytes,
                   cudaMemcpyDeviceToHost));

  const int error =
      verify_index_shuffling(h_scores.data(), num_tokens, num_experts, top_k,
                             h_counts.data(), h_shuffled_experts.data(),
                             h_shuffled_tokens.data());
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) {
    cudaFree(d_scores);
    cudaFree(d_counts);
    cudaFree(d_shuffled_experts);
    cudaFree(d_shuffled_tokens);
    cudaFree(d_buf_experts);
    cudaFree(d_buf_tokens);
    exit(EXIT_FAILURE);
  }

  // Warmup
  for (int i = 0; i < 10; ++i) {
    CHECK(cudaMemsetAsync(d_counts, 0, counts_bytes, stream));
    dispatch_index_shuffling(d_scores, num_tokens, num_experts, top_k, d_counts,
                             d_shuffled_experts, d_shuffled_tokens,
                             d_buf_experts, d_buf_tokens, num_sms, stream);
  }
  CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    CHECK(cudaMemsetAsync(d_counts, 0, counts_bytes, stream));
    dispatch_index_shuffling(d_scores, num_tokens, num_experts, top_k, d_counts,
                             d_shuffled_experts, d_shuffled_tokens,
                             d_buf_experts, d_buf_tokens, num_sms, stream);
  }
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the index shuffle kernel "
         "(tokens %d, experts %d, topk %d): %f (us)\n",
         num_tokens, num_experts, top_k, (time * 1e-3f) / repeat);

  cudaFree(d_scores);
  cudaFree(d_counts);
  cudaFree(d_shuffled_experts);
  cudaFree(d_shuffled_tokens);
  cudaFree(d_buf_experts);
  cudaFree(d_buf_tokens);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of tokens> <number of experts> <top K> <repeat>\n",
           argv[0]);
    printf("  experts must be 16, 32, or 128; top K must be 1, 2, or 4\n");
    return 1;
  }
  const int num_tokens = atoi(argv[1]);
  const int num_experts = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  if (num_tokens <= 0 || repeat <= 0) {
    printf("tokens and repeat must be positive\n");
    return 1;
  }
  run_case(num_tokens, num_experts, top_k, repeat);
  return 0;
}
