#include <sycl/sycl.hpp>

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
template <int BLOCK_SIZE = 128>
void bgmv_shrink_kernel(
    float *output,             // [num_tokens, lora_rank]
    const sycl::half *input,   // [num_tokens, hidden_size]
    const sycl::half *weights, // [num_loras, lora_rank, hidden_size]
    const int *lora_indices,   // [num_tokens]
    int num_tokens, int hidden_size, int lora_rank, float scaling,
    sycl::nd_item<3> &item)
{
    // grid : (num_tokens, lora_rank)
    auto g = item.get_group();
    int token_idx = item.get_group(2);
    int rank_idx = item.get_group(1);
    int tid = item.get_local_id(2);

    if (token_idx >= num_tokens || rank_idx >= lora_rank) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const sycl::half *inp = input + token_idx * hidden_size;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const sycl::half *wt =
        weights + lora_id * lora_rank * hidden_size + rank_idx * hidden_size;

    // each thread processes partial dot product along hidden dimension 
    float acc = 0.0f;
    for (int k = tid; k < hidden_size; k += BLOCK_SIZE) {
        acc += sycl::vec<sycl::half, 1>(inp[k])
                   .convert<float, sycl::rounding_mode::automatic>()[0] *
               sycl::vec<sycl::half, 1>(wt[k])
                   .convert<float, sycl::rounding_mode::automatic>()[0];
    }

    acc = sycl::reduce_over_group(g, acc, sycl::plus<float>());
    if (tid == 0)
      output[token_idx * lora_rank + rank_idx] += scaling * acc;
}

inline void unpack(const sycl::half *__restrict__ in,
                         sycl::half *__restrict__ out)
{
  auto v4 = *reinterpret_cast<const sycl::float4 *>(in);
  auto h2 = reinterpret_cast<const sycl::half2 *>(&v4);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      out[2 * i] = h2[i][0];
      out[2 * i + 1] = h2[i][1];
  }
}

template <int BLOCK_SIZE>
void bgmv_shrink_kernel_vec(
    float *output,             // [num_tokens, lora_rank]
    const sycl::half *input,   // [num_tokens, hidden_size]
    const sycl::half *weights, // [num_loras, lora_rank, hidden_size]
    const int *lora_indices,   // [num_tokens]
    int num_tokens, int hidden_size, int lora_rank, float scaling,
    sycl::nd_item<3> &item)
{
    // grid : (num_tokens, lora_rank)
    auto g = item.get_group();
    int token_idx = item.get_group(2);
    int rank_idx = item.get_group(1);
    int tid = item.get_local_id(2);

    if (token_idx >= num_tokens || rank_idx >= lora_rank) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const sycl::half *inp = input + token_idx * hidden_size;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const sycl::half *wt =
        weights + lora_id * lora_rank * hidden_size + rank_idx * hidden_size;

    int offset = tid * 8;

    float acc = 0.f;

    while (offset < hidden_size) {
      const auto remain = hidden_size - offset;
      if (remain >= 8) {
        sycl::half inp_vec[8], wt_vec[8];
        unpack(inp + offset, inp_vec);
        unpack(wt + offset, wt_vec);

        for (int i = 0; i < 8; i++)
          acc += sycl::vec<sycl::half, 1>(inp_vec[i])
                     .convert<float, sycl::rounding_mode::automatic>()[0] *
                 sycl::vec<sycl::half, 1>(wt_vec[i])
                     .convert<float, sycl::rounding_mode::automatic>()[0];

      } else {
        // Partial vector processing: handle remaining elements less than vec_size
        for (int k = 0; k < remain; k++) {
          acc += sycl::vec<sycl::half, 1>(inp[offset + k])
                     .convert<float, sycl::rounding_mode::automatic>()[0] *
                 sycl::vec<sycl::half, 1>(wt[offset + k])
                     .convert<float, sycl::rounding_mode::automatic>()[0];
        }
      }
      offset += BLOCK_SIZE * 8;  // next position handled by current thread
    }

    acc = sycl::reduce_over_group(g, acc, sycl::plus<float>());
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
template <int BLOCK_SIZE>
void bgmv_expand_kernel(
    sycl::half *output,        // [num_tokens, hidden_size]
    const float *input,        // [num_tokens, lora_rank]
    const sycl::half *weights, // [num_loras, hidden_size, lora_rank]
    const int *lora_indices,   // [num_tokens]
    int num_tokens, int hidden_size, int lora_rank, bool add_to_output,
    sycl::nd_item<3> &item)
{
    // grid : (num_tokens, hidden_size)
    auto g = item.get_group();
    int token_idx = item.get_group(2);
    int hidden_idx = item.get_group(1);
    int tid = item.get_local_id(2);

    if (token_idx >= num_tokens || hidden_idx >= hidden_size) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, lora_rank] -> inputs[batch_id, :]
    const float*  inp = input   + token_idx  * lora_rank;

    // weights: [num_loras, hidden_size, rank] -> weights[lora_idx, rank_id, :]
    const sycl::half *wt =
        weights + lora_id * hidden_size * lora_rank + hidden_idx * lora_rank;

    // each thread processes partial dot product along rank dimension 
    float acc = 0.0f;
    for (int r = tid; r < lora_rank; r += BLOCK_SIZE) {
        acc += inp[r] * sycl::vec<sycl::half, 1>(wt[r])
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    }

    acc = sycl::reduce_over_group(g, acc, sycl::plus<float>());
    if (tid == 0) {
      int out_idx = token_idx * hidden_size + hidden_idx;
      float prev = add_to_output
              ? sycl::vec<sycl::half, 1>(output[out_idx])
                    .convert<float, sycl::rounding_mode::automatic>()[0]
              : 0.f;
      output[out_idx] =
          sycl::vec<float, 1>(prev + acc)
              .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
}

// Host-side launchers
template <int block_size>
void launch_bgmv_shrink(sycl::queue &stream,
                        float *d_output, const sycl::half *d_input,
                        const sycl::half *d_weights, const int *d_lora_indices,
                        int num_tokens, int hidden_size, int lora_rank,
                        float scaling, bool vectorize)
{
  sycl::range<3> gws (1, lora_rank, num_tokens * block_size);
  sycl::range<3> lws (1, 1, block_size);

  if (vectorize) {
    stream.parallel_for(
      sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
        bgmv_shrink_kernel_vec<block_size>(d_output, d_input, d_weights,
                                           d_lora_indices, num_tokens,
                                           hidden_size, lora_rank, scaling,
                                           item);
      });
  } else {
    stream.parallel_for(
      sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
        bgmv_shrink_kernel<block_size>(d_output, d_input, d_weights,
                                       d_lora_indices, num_tokens,
                                       hidden_size, lora_rank, scaling,
                                       item);
    });
  }
}

template <int block_size>
void launch_bgmv_expand(sycl::queue &stream,
                        sycl::half *d_output, const float *d_input,
                        const sycl::half *d_weights, const int *d_lora_indices,
                        int num_tokens, int hidden_size, int lora_rank,
                        bool add_to_output)
{
  sycl::range<3> gws (1, hidden_size, num_tokens * block_size);
  sycl::range<3> lws (1, 1, block_size);

  stream.parallel_for(
    sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      bgmv_expand_kernel<block_size>(d_output, d_input, d_weights,
                                     d_lora_indices, num_tokens,
                                     hidden_size, lora_rank, add_to_output,
                                     item);
  });
}
