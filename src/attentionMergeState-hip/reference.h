#include <algorithm>
#include <cmath>
#include <limits>

// overloaded functions
inline float to_float32(float u) {
  return u;
}
inline float to_float32(half u) {
  return __half2float(u);
}
inline float to_float32(__hip_bfloat16 u) {
  return __bfloat162float(u);
}
inline void from_float32(float& d, float s) {
  d = s;
}
inline void from_float32(half& d, float s) {
  d = __float2half(s);
}
inline void from_float32(__hip_bfloat16& d, float s) {
  d = __float2bfloat16(s);
}

template <typename scalar_t>
void reference (
          scalar_t* __restrict__ output,        // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const scalar_t* __restrict__ prefix_output, // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const scalar_t* __restrict__ suffix_output, // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]

          float* __restrict__ lse,           // [NUM_TOKENS,NUM_HEADS] or nullptr
    const float* __restrict__ prefix_lse,    // [NUM_TOKENS,NUM_HEADS]
    const float* __restrict__ suffix_lse,    // [NUM_TOKENS,NUM_HEADS]
    int num_tokens,
    int num_heads,
    int head_size)
{
    // compute per-element scale factors p_scale and s_scale.
    for (int t = 0; t < num_tokens; ++t) {
      for (int h = 0; h < num_heads; ++h) {
            const int lse_idx = t * num_heads + h;

            // exp(-inf)=0, exp(+inf)=NaN
            float p_lse = prefix_lse[lse_idx];
            float s_lse = suffix_lse[lse_idx];

            const float inf = std::numeric_limits<float>::infinity();
            if (p_lse == inf) p_lse = -inf;
            if (s_lse == inf) s_lse = -inf;

            // safe-softmax: subtract the element-wise maximum so that
            // numerical range under control.
            const float max_lse = std::max(p_lse, s_lse);

            const float p_lse_shifted = p_lse - max_lse;
            const float s_lse_shifted = s_lse - max_lse;

            const float p_exp = std::exp(p_lse_shifted); // prefix contribution
            const float s_exp = std::exp(s_lse_shifted); // suffix contribution
            const float out_se = p_exp + s_exp;          // normaliser

            // write merged log-sum-exp
            if (lse != nullptr) {
                lse[lse_idx] = std::log(out_se) + max_lse;
            }

            // compute the scales
            const float p_scale = p_exp / out_se;
            const float s_scale = s_exp / out_se;

            // write output
            const int base = t * num_heads * head_size + h * head_size;
            for (int d = 0; d < head_size; ++d) {
                float sum = to_float32(prefix_output[base + d]) * p_scale +
                            to_float32(suffix_output[base + d]) * s_scale;
                from_float32(output[base + d], sum);
            }
        }
    }
}
