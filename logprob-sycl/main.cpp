/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

#ifdef WAVE64
#define SG 64
#else
#define SG 32
#endif

static const float HALF_FLT_MAX = 65504.F;

#ifndef USE_SYCL_GROUP_REDUCE
  #include "reduce.h"
#endif

template<typename T>
void log_probs_kernel(
    sycl::nd_item<2>   &item,
#ifndef USE_SYCL_GROUP_REDUCE
    float        *shared,
#endif
    float        &s_max_logit,
    float*       log_probs,
    const T*     logits,
    const int*   ids,
    const int*   lengths,
    const int    max_input_length,
    const int    batch_size,
    const int    vocab_size,
    const int    vocab_size_padded,
    bool         batch_first)
{
  // Calculate the log probability from logits.
  //   log_probs[t, :] = log(softmax(logits))[ids[t + 1, :]]
  //
  // log_probs: [max_length - 1, batch_size] or [batch_size, max_length -1],
  //     log probabilities of each token.
  // logits: [max_length, batch_size, vocab_size_padded] or [batch_size, max_length, vocab_size_padded]
  // lengths: [batch_size], sequence lengths
  // ids: [max_length, batch_size], token ids.
  // batch_size: [1], batch_size. in case of beam > 1, batch x beam.
  // vocab_size: [1], vocab_size,
  // vocab_size: [1], vocab_size_padded, padded vocab size.

  const bool IS_FP16 = std::is_same<T, sycl::half>::value;
  const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  int tidx = item.get_local_id(1); // vocab dim
  int bidx = batch_first ? item.get_group(1) : item.get_group(0); // batch dim
  int step = batch_first ? item.get_group(0) : item.get_group(1); // step dim

  if (bidx < batch_size && step < lengths[bidx] - 1) {
    // Compute the address of logits to data for the current batch
    int step_offset  = batch_first ? step * vocab_size_padded : step * batch_size * vocab_size_padded;
    int batch_offset = batch_first ? bidx * max_input_length * vocab_size_padded : bidx * vocab_size_padded;
    logits += step_offset + batch_offset;

    // Find max(logits)
    float local_max = -MAX_T_VAL;
    float val       = -MAX_T_VAL;
    for (int i = tidx; i < vocab_size; i += item.get_local_range(1)) {
      val       = static_cast<float>(logits[i]);
      local_max = sycl::fmax(local_max, val);
    }

#ifdef USE_SYCL_GROUP_REDUCE
    float max_val = reduce_over_group(item.get_group(), local_max, maximum<>());
#else
    float max_val = blockReduceMax<float>(local_max, item, shared);
#endif
    if (tidx == 0) {
      s_max_logit = max_val;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Calculate the denominator: sum_i exp(logits[i])
    float local_sum_exp = 0.0f;
    for (int i = tidx; i < vocab_size; i += item.get_local_range(1)) {
      val = sycl::exp(static_cast<float>(logits[i]) - s_max_logit);
      local_sum_exp += val;
    }

#ifdef USE_SYCL_GROUP_REDUCE
    float sum_exp  = reduce_over_group(item.get_group(), local_sum_exp, plus<>());
#else
    float sum_exp = blockReduceSum<float>(local_sum_exp, item, shared);
#endif
    if (tidx == 0) {
      int idx = batch_first ? step + bidx * (max_input_length - 1) : step * batch_size + bidx;
      // log_probs[step, ...] is the log probability of a token at step t + 1.
      int token_idx = batch_first ? step + 1 + bidx * max_input_length : (step + 1) * batch_size + bidx;
      log_probs[idx] = static_cast<float>(logits[ids[token_idx]]) -
                       s_max_logit - sycl::log(sum_exp + 1e-9f);
    }
  }
}

void accumulate_log_probs(
    sycl::nd_item<2>   &item,
#ifndef USE_SYCL_GROUP_REDUCE
          float *shared,
#endif
          float* cum_log_probs,
    const float* log_probs,
    const int*   lengths,
    const int    max_input_length,
    const int    batch_size,
    const bool   batch_first)
{
  // Accumulate the log probability along the sequence dimension.
  //   cum_log_probs[j] = sum_i log(softmax(logits))[ids[i,j]]
  //
  // cum_log_probs: [batch_size], cumulative log probability
  // log_probs: [max_length - 1, batch_size] or [batch_size, max_length - 1],
  //   log probability of each token
  // lengths: [batch_size], sequence lengths
  // batch_size: [1], batch_size. in case of beam > 1, batch x beam.

  int bidx = item.get_group(1);    // batch dim
  int tidx = item.get_local_id(1); // step dim

  int length = lengths[bidx];
  // reposition logits to data for the current batch.
  log_probs += batch_first ? bidx * (max_input_length - 1) : bidx;
  int stride = batch_first ? 1 : batch_size;  // stride along with seq dim.
  float local_accum = 0.0f;
  for (int step = tidx; step < length - 1;
       step += item.get_local_range(1)) {
    local_accum += static_cast<float>(log_probs[step * stride]);
  }
#ifdef USE_SYCL_GROUP_REDUCE
  float accum  = reduce_over_group(item.get_group(), local_accum, plus<>());
#else
  float accum = blockReduceSum<float>(local_accum, item, shared);
#endif
  if (tidx == 0) {
    cum_log_probs[bidx] = accum;
  }
}


int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <maximum sequence length> <batch size> <vocabulary size> <repeat>\n", argv[0]);
    return 1;
  }
  const int max_length = atoi(argv[1]);  // max input length
  const int batch_size = atoi(argv[2]);
  const int vocab_size = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int vocab_size_padded = (vocab_size + 31) / 32 * 32;

  size_t logits_size = (size_t)batch_size * max_length * vocab_size_padded;
  size_t logits_size_bytes = logits_size * sizeof(float);

  size_t log_probs_size = (size_t)batch_size * (max_length - 1);
  size_t log_probs_size_bytes = log_probs_size * sizeof(float);

  size_t batch_size_bytes = batch_size * sizeof(float);
  size_t length_size_bytes = batch_size * sizeof(int);

  float *h_logits = (float*) malloc (logits_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-6.f, 6.f);

  for (size_t i = 0; i < logits_size; i++)
    h_logits[i] = distr(g);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_logits = sycl::malloc_device<float>(logits_size, q);
  q.memcpy(d_logits, h_logits, logits_size_bytes);

  float *h_log_probs = (float*) malloc (log_probs_size_bytes);
  float *d_log_probs = sycl::malloc_device<float>(log_probs_size, q);

  float *h_cum_log_probs = (float*) malloc (batch_size_bytes);
  float *d_cum_log_probs = sycl::malloc_device<float>(batch_size, q);

  int *h_lengths = (int*) malloc (length_size_bytes);

  srand(123);
  for (int i = 0; i < batch_size; i++)
    h_lengths[i] = rand() % max_length + max_length / 2;

  int *d_lengths = sycl::malloc_device<int>(batch_size, q);
  q.memcpy(d_lengths, h_lengths, length_size_bytes);

  size_t ids_size = batch_size * max_length;
  size_t ids_size_bytes = ids_size * sizeof(int);

  int *h_ids = (int*) malloc (ids_size_bytes);
  for (size_t i = 0; i < ids_size; i++)
    h_ids[i] = rand() % vocab_size;

  int *d_ids = sycl::malloc_device<int>(ids_size, q);
  q.memcpy(d_ids, h_ids, ids_size_bytes);

  // A batched version of log prob computation.
  //
  // cum_log_probs: [batch_size]
  // logits: [max_input_length, batch_size, vocab_size] or [batch_size, max_input_length, vocab_size]
  // input_ids: [max_input_length, batch_size] or [max_input_length, batch_size]
  // input_lengths: [batch_size]

  // TODO: batch_first = 0 does not produce the same results for each run
  bool batch_first = 1;

  const int block_size = vocab_size < 1024 ? (vocab_size + 31) / 32 * 32 : 1024;
  const int gx = batch_first ? batch_size : max_length - 1;
  const int gy = batch_first ? max_length - 1 : batch_size;

  sycl::range<2> gws (gy, gx * block_size);
  sycl::range<2> gws2 (1, batch_size * block_size);
  sycl::range<2> lws (1, block_size);

  q.wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    q.submit([&](sycl::handler &cgh) {
#ifndef USE_SYCL_GROUP_REDUCE
      sycl::local_accessor<float, 1> sm (sycl::range<1>(SG), cgh);
#endif
      sycl::local_accessor<float, 0> s_max_logit (cgh);
      cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item)
        [[intel::reqd_sub_group_size(SG)]] {
        log_probs_kernel<float>(
            item,
#ifndef USE_SYCL_GROUP_REDUCE
            sm.get_pointer(),
#endif
            s_max_logit,
            d_log_probs, d_logits, d_ids, d_lengths, max_length, batch_size,
            vocab_size, vocab_size_padded, batch_first);
      });
    });

    q.submit([&](sycl::handler &cgh) {
#ifndef USE_SYCL_GROUP_REDUCE
      sycl::local_accessor<float, 1> sm (sycl::range<1>(SG), cgh);
#endif
      cgh.parallel_for(sycl::nd_range<2>(gws2, lws), [=](sycl::nd_item<2> item)
        [[intel::reqd_sub_group_size(SG)]] {
        accumulate_log_probs(
           item,
#ifndef USE_SYCL_GROUP_REDUCE
           sm.get_pointer(),
#endif
           d_cum_log_probs, d_log_probs, d_lengths,
           max_length, batch_size, batch_first);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_log_probs, d_log_probs, log_probs_size_bytes).wait();
  q.memcpy(h_cum_log_probs, d_cum_log_probs, batch_size_bytes).wait();

  float checksum = 0, checkmax = FLT_MIN, checkmin = FLT_MAX;
  for (int i = 0; i < batch_size; i++) {
    checksum += h_cum_log_probs[i];
    checkmax = fmax(checkmax, h_cum_log_probs[i]);
    checkmin = fmin(checkmin, h_cum_log_probs[i]);
  }
  printf("Checksum = %f\n", checksum / batch_size);
  printf("Max cumulative log probs = %f\n", checkmax);
  printf("Min cumulative log probs = %f\n", checkmin);

  free(d_cum_log_probs, q);
  free(d_log_probs, q);
  free(d_logits, q);
  free(d_ids, q);
  free(d_lengths, q);

  free(h_cum_log_probs);
  free(h_log_probs);
  free(h_logits);
  free(h_ids);
  free(h_lengths);

  return 0;
}
