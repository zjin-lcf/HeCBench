
template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d)
{
  const int64_t output_base = blockIdx.x * d; 
  const int64_t input_base = output_base * TOPK;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += __ldg(&input[input_base + k * d + idx]);
    }
    out[output_base + idx] = x;
  }
}

template <int TOPK>
__global__ void moe_sum_kernel_vec4(
    float* __restrict__ out,          // [..., d]
    const float* __restrict__ input,  // [..., topk, d]
    int d)
{
  int d4 = d >> 2; // divisible by 4
  int output_base4 = (blockIdx.x * d) >> 2;
  int input_base4  = (blockIdx.x * d * TOPK) >> 2;

  const float4* input4 = reinterpret_cast<const float4*>(input);
  float4* out4 = reinterpret_cast<float4*>(out);

  for (int idx = threadIdx.x; idx < d4; idx += blockDim.x) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      float4 v = input4[input_base4 + k * d4 + idx];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
    out4[output_base4 + idx] = acc;
  }
}
