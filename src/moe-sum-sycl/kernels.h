template <typename scalar_t, int TOPK>
void moe_sum_kernel(
    scalar_t* __restrict out,          // [..., d]
    const scalar_t* __restrict input,  // [..., topk, d]
    const int d,
    sycl::nd_item<1> &item)
{
  const int64_t output_base = item.get_group(1) * d;
  const int64_t input_base = output_base * TOPK;
  for (int64_t idx = item.get_local_id(1); idx < d;
       idx += item.get_local_range(1)) {
    scalar_t x = 0.0;
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += input[input_base + k * d + idx];
    }
    out[output_base + idx] = x;
  }
}

template <int TOPK>
void moe_sum_kernel_vec4(
    float* __restrict out,          // [..., d]
    const float* __restrict input,  // [..., topk, d]
    int d,
    sycl::nd_item<1> &item)
{
  int d4 = d >> 2; // divisible by 4
  int output_base4 = (item.get_group(1) * d) >> 2;
  int input_base4 = (item.get_group(1) * d * TOPK) >> 2;

  const sycl::float4 *input4 = reinterpret_cast<const sycl::float4 *>(input);
  sycl::float4 *out4 = reinterpret_cast<sycl::float4 *>(out);

  for (int idx = item.get_local_id(1); idx < d4;
       idx += item.get_local_range(1)) {
    sycl::float4 acc = sycl::float4(0.f);
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      sycl::float4 v = input4[input_base4 + k * d4 + idx];
      acc += v;
    }
    out4[output_base4 + idx] = acc;
  }
}
