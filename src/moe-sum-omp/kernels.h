
template <typename scalar_t, int TOPK>
void moe_sum_kernel(
    scalar_t* out,
    const scalar_t* input,
    const int d,
    const int num_blocks,
    const int block_size)
{
  #pragma omp target teams distribute parallel for collapse(2) num_threads(block_size)
  for (int block = 0; block < num_blocks; block++) {
    for (int idx = 0; idx < d; idx++) {

      const int64_t output_base = (int64_t)block * d;
      const int64_t input_base  = output_base * TOPK;

      scalar_t x = 0;

      #pragma unroll
      for (int k = 0; k < TOPK; k++) {
        x += input[input_base + k * d + idx];
      }

      out[output_base + idx] = x;
    }
  }
}

template <typename scalar_t, int TOPK, int VL=4>
void moe_sum_kernel_vec(
          float* __restrict__ out,
    const float* __restrict__ input,
    int d,
    int num_blocks,
    const int block_size)
{
  int dv = d / VL;

  #pragma omp target teams distribute parallel for collapse(2) num_threads(block_size)
  for (int block = 0; block < num_blocks; block++) {
    for (int idx = 0; idx < dv; idx++) {

      const int64_t output_base = (int64_t)block * d;
      const int64_t input_base  = output_base * TOPK;

      scalar_t acc[VL];
      scalar_t v[VL];

      #pragma unroll
      for (int i = 0; i < VL; i++)
        acc[i] = 0;

      #pragma unroll
      for (int k = 0; k < TOPK; ++k) {

        #pragma unroll
        for (int i = 0; i < VL; i++) {
          v[i] = input[input_base + k * d + idx * VL + i];
          acc[i] += v[i];
        }
      }

      #pragma unroll
      for (int i = 0; i < VL; i++)
        out[output_base + idx * VL + i] = acc[i];
    }
  }
}
