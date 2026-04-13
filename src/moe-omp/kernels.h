#include <assert.h>
#include <float.h>

template <int TPB>
void moeSoftmax(
    const float* input,
    const bool* finished,
    float* output,
    const int num_tokens,
    const int num_cols)
{
  #pragma omp target teams distribute parallel for \
   num_teams(num_tokens) num_threads(TPB)
  for (int token = 0; token < num_tokens; token++) {

    if (finished && finished[token]) {
      continue;
    }

    const int row_offset = token * num_cols;

    float max_val = -FLT_MAX;

    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < num_cols; i++) {
      float val = input[row_offset + i];
      if (val > max_val) max_val = val;
    }

    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_cols; i++) {
      sum += expf(input[row_offset + i] - max_val);
    }

    float inv_sum = 1.0f / sum;

    #pragma omp parallel for
    for (int i = 0; i < num_cols; i++) {
      output[row_offset + i] =
          expf(input[row_offset + i] - max_val) * inv_sum;
    }
  }
}

struct kv_t { int key; float value; };

#pragma omp declare target
inline kv_t arg_max(const kv_t &a, const kv_t &b) {
  if (a.value > b.value)
    return a;
  else if (a.value == b.value)
    return {a.key < b.key ? a.key : b.key, a.value};
  else
    return b;
}
#pragma omp end declare target

template <int TPB>
void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* weights,
    int* indices,
    int* source_rows,
    const int num_tokens,
    const int num_experts,
    const int K,
    const int start_expert,
    const int end_expert)
{
  #pragma omp target teams distribute parallel for \
   num_teams(num_tokens) num_threads(TPB)
  for (int token = 0; token < num_tokens; token++) {
    const bool row_is_active = finished ? !finished[token] : true;
    const int thread_read_offset = token * num_experts;
    for (int k_idx = 0; k_idx < K; ++k_idx) {
      kv_t thread_kvp = {0, -1.f}; // This is OK because inputs are probabilities
      for (int expert = 0; expert < num_experts; expert++) {
        const int idx = thread_read_offset + expert;
        kv_t inp_kvp = {expert, inputs_after_softmax[idx]};

        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          const int prior_winning_expert = indices[K * token + prior_k];
          if (prior_winning_expert == expert) {
            inp_kvp = thread_kvp;
          }
        }
        thread_kvp = arg_max(inp_kvp, thread_kvp);
      }

      const int expert = thread_kvp.key;
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = token * K + k_idx;
      weights[idx] = thread_kvp.value;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = k_idx * num_tokens + token;
    }
  }
}
