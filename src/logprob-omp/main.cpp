#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

// ---------------------------------------------------------------------------
// log_probs kernel
// ---------------------------------------------------------------------------
template<typename T>
void log_probs_kernel(
    float*     log_probs,
    const T*   logits,
    const int* ids,
    const int* lengths,
    const int  max_input_length,
    const int  batch_size,
    const int  vocab_size,
    const int  vocab_size_padded,
    const int    block_size)
{
  const float MAX_T_VAL = FLT_MAX;

  #pragma omp target teams distribute parallel for collapse(2) \
    num_teams(batch_size * (max_input_length-1)) num_threads(block_size)
  for (int bidx = 0; bidx < batch_size; bidx++) {
    for (int step = 0; step < max_input_length - 1; step++) {
      if (step < lengths[bidx] - 1) {
        int batch_offset = bidx * max_input_length * vocab_size_padded;
        int step_offset  = step * vocab_size_padded;
        const T* logits_ptr = logits + batch_offset + step_offset;

        // Step 1: find max(logits) over vocab
        float max_val = -MAX_T_VAL;
        #pragma omp parallel reduction(max: max_val)
        for (int i = 0; i < vocab_size; i++) {
          float v = static_cast<float>(logits_ptr[i]);
          if (v > max_val) max_val = v;
        }

        // Step 2: sum_i exp(logits[i] - max_val)
        float sum_exp = 0.0f;
        #pragma omp parallel reduction(+: sum_exp)
        for (int i = 0; i < vocab_size; i++) {
          sum_exp += expf(static_cast<float>(logits_ptr[i]) - max_val);
        }

        // Step 3: log prob of the next token
        int idx       = step + bidx * (max_input_length - 1);
        int token_idx = (step + 1) + bidx * max_input_length;
        log_probs[idx] =
          static_cast<float>(logits_ptr[ids[token_idx]])
          - max_val
          - logf(sum_exp + 1e-9f);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// accumulate_log_probs kernel
// ---------------------------------------------------------------------------
void accumulate_log_probs(
    float* cum_log_probs,
    const float* log_probs,
    const int*   lengths,
    const int    max_input_length,
    const int    batch_size,
    const int    block_size)
{
  #pragma omp target teams distribute num_teams(batch_size)
  for (int bidx = 0; bidx < batch_size; bidx++) {
    int length = lengths[bidx];
    const float* lp = log_probs + bidx * (max_input_length - 1);

    float accum = 0.0f;
    #pragma omp parallel for reduction(+: accum) num_threads(block_size)
    for (int step = 0; step < length - 1; step++) {
      accum += lp[step];
    }
    cum_log_probs[bidx] = accum;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <max_seq_len> <batch_size> <vocab_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int max_length  = atoi(argv[1]);
  const int batch_size  = atoi(argv[2]);
  const int vocab_size  = atoi(argv[3]);
  const int repeat      = atoi(argv[4]);

  const int vocab_size_padded = (vocab_size + 31) / 32 * 32;

  size_t logits_size       = (size_t)batch_size * max_length * vocab_size_padded;
  size_t log_probs_size    = (size_t)batch_size * (max_length - 1);

  float* h_logits           = (float*) malloc(logits_size       * sizeof(float));
  float* h_log_probs        = (float*) malloc(log_probs_size    * sizeof(float));
  float* h_log_probs_ref    = (float*) malloc(log_probs_size    * sizeof(float));
  float* h_cum_log_probs    = (float*) malloc(batch_size        * sizeof(float));
  float* h_cum_log_probs_ref= (float*) malloc(batch_size        * sizeof(float));
  int*   h_lengths          = (int*)   malloc(batch_size        * sizeof(int));
  int*   h_ids              = (int*)   malloc(batch_size * max_length * sizeof(int));

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-6.f, 6.f);
  for (size_t i = 0; i < logits_size; i++)
    h_logits[i] = distr(g);

  srand(123);
  for (int i = 0; i < batch_size; i++)
    h_lengths[i] = max_length;
  for (int i = 0; i < batch_size * max_length; i++)
    h_ids[i] = rand() % vocab_size;

  const int block_size = vocab_size < 1024 ? (vocab_size + 31) / 32 * 32 : 1024;

  #pragma omp target data map(to:   h_logits[0:logits_size], \
                                    h_ids[0:batch_size*max_length], \
                                    h_lengths[0:batch_size]) \
                          map(alloc: h_log_probs[0:log_probs_size], \
                                     h_cum_log_probs[0:batch_size])
  {
    // warmup + verify
    log_probs_kernel<float>(
        h_log_probs, h_logits, h_ids, h_lengths,
        max_length, batch_size, vocab_size, vocab_size_padded, block_size);

    accumulate_log_probs(
        h_cum_log_probs, h_log_probs, h_lengths,
        max_length, batch_size, block_size);

    #pragma omp target update from(h_log_probs[0:log_probs_size])
    #pragma omp target update from(h_cum_log_probs[0:batch_size])

    // reference
    log_probs_cpu<float>(
        h_log_probs_ref, h_logits, h_ids, h_lengths,
        max_length, batch_size, vocab_size, vocab_size_padded,
        h_cum_log_probs_ref);

    bool error = false;
    for (size_t i = 0; i < log_probs_size; i++) {
      if (fabsf(h_log_probs[i] - h_log_probs_ref[i]) > 1e-3f) {
        printf("log_probs mismatch @%zu: %f != %f\n",
            i, h_log_probs[i], h_log_probs_ref[i]);
        error = true;
        break;
      }
    }
    for (int i = 0; i < batch_size; i++) {
      if (fabsf(h_cum_log_probs[i] - h_cum_log_probs_ref[i]) > 1e-1f) {
        printf("cum_log_probs mismatch @%d: %f != %f\n",
            i, h_cum_log_probs[i], h_cum_log_probs_ref[i]);
        error = true;
      }
    }
    printf("%s\n", error ? "FAIL" : "PASS");

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      log_probs_kernel<float>(
          h_log_probs, h_logits, h_ids, h_lengths,
          max_length, batch_size, vocab_size, vocab_size_padded, block_size);

      accumulate_log_probs(
          h_cum_log_probs, h_log_probs, h_lengths,
          max_length, batch_size, block_size);
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels: %f (us)\n",
        (time * 1e-3f) / repeat);

  } // end omp target data

  free(h_logits);
  free(h_log_probs);
  free(h_log_probs_ref);
  free(h_cum_log_probs);
  free(h_cum_log_probs_ref);
  free(h_ids);
  free(h_lengths);

  return 0;
}
