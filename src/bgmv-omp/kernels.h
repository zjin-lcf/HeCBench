#include <omp.h>

using half = _Float16;

// ============================================================
// BGMV-Shrink kernel
//
//   output[token, r] += scaling * sum_k( input[token, k] * W[lora_id, r, k] )
//   where lora_id = lora_indices[token]
//
//   input  : [num_tokens, hidden_size]   (fp16)
//   weight : [num_loras,  lora_rank, hidden_size]  (fp16)
//   output : [num_tokens, lora_rank]     (fp32, accumulated)
//   lora_indices : [num_tokens]          (int32)
// ============================================================
void bgmv_shrink_kernel(
    float*        output,          // [num_tokens, lora_rank]
    const half* __restrict__ input,        // [num_tokens, hidden_size]
    const half* __restrict__ weights,      // [num_loras, lora_rank, hidden_size]
    const int*    __restrict__ lora_indices, // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    float         scaling,
    int           block_size)
{
    // Each team computes one output element
    #pragma omp target teams distribute collapse(2) num_teams(lora_rank * num_tokens)
    for (int rank_idx = 0; rank_idx < lora_rank; rank_idx++) {
      for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
            int lora_id = lora_indices[token_idx];
            
            // inputs: [num_tokens, hidden_size] -> inputs[token_idx, :]
            const half* inp = input + token_idx * hidden_size;
            
            // weights: [num_loras, lora_rank, hidden_size] -> weights[lora_id, rank_idx, :]
            const half* wt = weights + lora_id * lora_rank * hidden_size
                                        + rank_idx * hidden_size;
            
            float acc = 0.0f;

            #pragma omp parallel for reduction(+:acc) num_threads(block_size)
            for (int k = 0; k < hidden_size; k++) {
                acc += float(inp[k]) * float(wt[k]);
            }
            
            output[token_idx * lora_rank + rank_idx] += scaling * acc;
        }
    }
}

// ============================================================
// BGMV-Expand kernel
//
//   output[token, n] = sum_r(input[token, r] * W[lora_id, n, r])
//                      + (add_to_output ? outputs[token, n] : 0)
//   where lora_id = lora_indices[token]
//
//   input  : [num_tokens, lora_rank]     (fp32)
//   weight : [num_loras,  hidden_size, lora_rank]  (fp16)
//   output : [num_tokens, hidden_size]   (fp16)
// ============================================================
void bgmv_expand_kernel(
    half*       output,          // [num_tokens, hidden_size]
    const float*  input,           // [num_tokens, lora_rank]
    const half* weights,         // [num_loras, hidden_size, lora_rank]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    bool          add_to_output,
    int           block_size)
{
    #pragma omp target teams distribute collapse(2) num_teams(hidden_size * num_tokens)
    for (int hidden_idx = 0; hidden_idx < hidden_size; hidden_idx++) {
      for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
            int lora_id = lora_indices[token_idx];
            
            // inputs: [num_tokens, lora_rank] -> inputs[token_idx, :]
            const float* inp = input + token_idx * lora_rank;
            
            // weights: [num_loras, hidden_size, lora_rank] -> weights[lora_id, hidden_idx, :]
            const half* wt = weights + lora_id * hidden_size * lora_rank
                                         + hidden_idx * lora_rank;
            
            // Each thread processes a chunk of the dot product
            float acc = 0.0f;
            
            #pragma omp parallel for reduction(+:acc) num_threads(block_size)
            for (int r = 0; r < lora_rank; r++) {
                acc += inp[r] * float(wt[r]);
            }
            
            int out_idx = token_idx * hidden_size + hidden_idx;
            float prev = add_to_output ? float(output[out_idx]) : 0.f;
            
            output[out_idx] = half(prev + acc);
        }
    }
}

// Host-side launchers
void launch_bgmv_shrink(float* output,
                        const half* input,
                        const half* weights,
                        const int*    lora_indices,
                        int num_tokens, int hidden_size, int lora_rank, float scaling,
                        bool vectorize, int block_size)
{
  bgmv_shrink_kernel(
    output, input, weights, lora_indices,
    num_tokens, hidden_size, lora_rank, scaling, block_size);
}

void launch_bgmv_expand(half* output,
                        const float*  input,
                        const half* weights,
                        const int*    lora_indices,
                        int num_tokens, int hidden_size, int lora_rank,
                        bool add_to_output, int block_size)
{
  bgmv_expand_kernel(
      output, input, weights, lora_indices,
      num_tokens, hidden_size, lora_rank, add_to_output, block_size);
}
