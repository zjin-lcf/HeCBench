#include <hipcub/hipcub.hpp>

// ============================================================
// BGMV-Shrink kernel
//
//   output[token, r] += scaling * sum_k( input[token, k] * W[lora_id, r, k] )
//   where lora_id = lora_indices[token]
//
//   input  : [num_tokens, hidden_size]   (fp16)
//   weight : [num_loras,  lora_rank, hidden_size]  (fp16, col-major => row-major here)
//   output : [num_tokens, lora_rank]     (fp32, accumulated)
//   lora_indices : [num_tokens]          (int32)  -- per-token LoRA index
// ============================================================
template<int BLOCK_SIZE = 128>
__global__ void bgmv_shrink_kernel(
    float*        output,          // [num_tokens, lora_rank]
    const __half* input,           // [num_tokens, hidden_size]
    const __half* weights,         // [num_loras, lora_rank, hidden_size]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    float         scaling)
{
    // grid : (num_tokens, lora_rank)
    int token_idx = blockIdx.x;
    int rank_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || rank_idx >= lora_rank) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const __half* inp = input   + token_idx * hidden_size;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const __half* wt  = weights + lora_id * lora_rank * hidden_size
                                + rank_idx * hidden_size;

    // each thread processes partial dot product along hidden dimension 
    float acc = 0.0f;
    for (int k = tid; k < hidden_size; k += BLOCK_SIZE) {
        acc += __half2float(inp[k]) * __half2float(wt[k]);
    }

    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    acc = BlockReduce(temp_storage1).Sum(acc);
    if (tid == 0)
      output[token_idx * lora_rank + rank_idx] += scaling * acc;
}

__device__ 
inline void unpack(const half* __restrict__ in,
                         half* __restrict__ out)
{
  auto v4 = *reinterpret_cast<const float4*>(in);
  auto h2 = reinterpret_cast<const __half2*>(&v4);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      out[2*i]     = __low2half(h2[i]);
      out[2*i + 1] = __high2half(h2[i]);
  }
}

template<int BLOCK_SIZE>
__global__ void bgmv_shrink_kernel_vec(
    float*        output,          // [num_tokens, lora_rank]
    const __half* input,           // [num_tokens, hidden_size]
    const __half* weights,         // [num_loras, lora_rank, hidden_size]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    float         scaling)
{
    // grid : (num_tokens, lora_rank)
    int token_idx = blockIdx.x;
    int rank_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || rank_idx >= lora_rank) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const __half* inp = input   + token_idx * hidden_size;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const __half* wt  = weights + lora_id * lora_rank * hidden_size
                                + rank_idx * hidden_size;

    int offset = tid * 8;

    float acc = 0.f;

    while (offset < hidden_size) {
      const auto remain = hidden_size - offset;
      if (remain >= 8) {
        __half inp_vec[8], wt_vec[8];
        unpack(inp + offset, inp_vec);
        unpack(wt + offset, wt_vec);

        for (int i = 0; i < 8; i++) 
          acc += __half2float(inp_vec[i]) * __half2float(wt_vec[i]);
        
      } else {
        // Partial vector processing: handle remaining elements less than vec_size
        for (int k = 0; k < remain; k++) {
          acc += __half2float(inp[offset + k]) * __half2float(wt[offset + k]);
        }
      }
      offset += BLOCK_SIZE * 8;  // next position handled by current thread
    }

    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    acc = BlockReduce(temp_storage1).Sum(acc);
    if (tid == 0)
      output[token_idx * lora_rank + rank_idx] += scaling * acc;
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
//   add_to_output
// ============================================================
template<int BLOCK_SIZE>
__global__ void bgmv_expand_kernel(
    __half*       output,          // [num_tokens, hidden_size]
    const float*  input,           // [num_tokens, lora_rank]
    const __half* weights,         // [num_loras, hidden_size, lora_rank]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    bool          add_to_output)
{
    // grid : (num_tokens, hidden_size)
    int token_idx  = blockIdx.x;
    int hidden_idx = blockIdx.y;
    int tid        = threadIdx.x;

    if (token_idx >= num_tokens || hidden_idx >= hidden_size) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, lora_rank] -> inputs[batch_id, :]
    const float*  inp = input   + token_idx  * lora_rank;

    // weights: [num_loras, hidden_size, rank] -> weights[lora_idx, rank_id, :]
    const __half* wt  = weights + lora_id    * hidden_size * lora_rank
                                + hidden_idx * lora_rank;

    // each thread processes partial dot product along rank dimension 
    float acc = 0.0f;
    for (int r = tid; r < lora_rank; r += BLOCK_SIZE) {
        acc += inp[r] * __half2float(wt[r]);
    }

    using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    acc = BlockReduce(temp_storage1).Sum(acc);
    if (tid == 0) {
      int out_idx = token_idx * hidden_size + hidden_idx;
      float prev  = add_to_output ? __half2float(output[out_idx]) : 0.f;
      output[out_idx] = __float2half(prev + acc);
    }
}

// Host-side launchers
template <int block_size>
void launch_bgmv_shrink(float*        d_output,
                        const __half* d_input,
                        const __half* d_weights,
                        const int*    d_lora_indices,
                        int num_tokens, int hidden_size, int lora_rank, float scaling,
                        bool vectorize,
                        hipStream_t stream = 0)
{
  dim3 grid(num_tokens, lora_rank);
  dim3 block(block_size);
  if (vectorize)
    bgmv_shrink_kernel_vec<block_size><<<grid, block, 0, stream>>>(
      d_output, d_input, d_weights, d_lora_indices,
      num_tokens, hidden_size, lora_rank, scaling);
  else
    bgmv_shrink_kernel<block_size><<<grid, block, 0, stream>>>(
      d_output, d_input, d_weights, d_lora_indices,
      num_tokens, hidden_size, lora_rank, scaling);
}

template <int block_size>
void launch_bgmv_expand(__half*       d_output,
                        const float*  d_input,
                        const __half* d_weights,
                        const int*    d_lora_indices,
                        int num_tokens, int hidden_size, int lora_rank,
                        bool add_to_output,
                        hipStream_t stream = 0)
{
  dim3 grid(num_tokens, hidden_size);
  dim3 block(block_size);
  bgmv_expand_kernel<block_size><<<grid, block, 0, stream>>>(
      d_output, d_input, d_weights, d_lora_indices,
      num_tokens, hidden_size, lora_rank, add_to_output);
}
