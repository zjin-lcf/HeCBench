template <typename scalar_t>
void moe_sum_ref(
    int TOPK,
    scalar_t* out,          // [num_blocks, d]
    const scalar_t* input,  // [num_blocks, TOPK, d]
    int num_blocks,
    int d)
{
  for (int block = 0; block < num_blocks; ++block) {
    const int64_t output_base = static_cast<int64_t>(block) * d;
    const int64_t input_base  = output_base * TOPK;
    for (int idx = 0; idx < d; ++idx) {
      scalar_t x = 0;
      for (int k = 0; k < TOPK; ++k) {
        x += input[input_base + k * d + idx];
      }
      out[output_base + idx] = x;
    }
  }
}
