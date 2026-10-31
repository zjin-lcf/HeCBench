#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// Host reference for Ideogram v4 per-head RMSNorm plus rotate-half RoPE.
// Include a bf16 header before this file so bf16 and the convert intrinsics
// are declared.
//
// RMSNorm is PyTorch F.rms_norm: y = x * rsqrt(mean(x^2) + eps) * weight.
// The weight is initialized to one in the model, so the scale is the weight
// itself. The norm result is rounded to bf16 before rotation, which is the
// dtype the fused kernel stores between the two stages. The square sum is a
// sequential power-of-two tree and the reciprocal square root is libm, so it
// is not bit-identical to the GPU reduction. out[] is float; the caller
// rounds it when comparing.

inline float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

inline void ideogram_qk_rmsnorm_rope_host(float* out, const bf16* in, const bf16* weight,
                                          int head_dim, const bf16* cache,
                                          const int64_t* positions, const int* axis_map,
                                          int pos_stride, int token, bool mrope, float eps) {
  std::vector<float> red(head_dim);
  std::vector<float> normed(head_dim);
  for (int i = 0; i < head_dim; ++i) {
    float x = __bfloat162float(in[i]);
    red[i] = x * x;
  }
  for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
    for (int i = 0; i < stride; ++i)
      red[i] += red[i + stride];
  }
  const float inv_rms = 1.f / sqrtf(red[0] / static_cast<float>(head_dim) + eps);

  for (int i = 0; i < head_dim; ++i) {
    float x = __bfloat162float(in[i]);
    float w = __bfloat162float(weight[i]);
    normed[i] = round_bf16(x * inv_rms * w);
  }

  const int half = head_dim / 2;
  for (int i = 0; i < head_dim; ++i) {
    const int lane = (i < half) ? i : (i - half);
    const int axis = mrope ? axis_map[lane] : 0;
    const int64_t pos = mrope ? positions[axis * pos_stride + token] : positions[token];
    const bf16* row = cache + pos * head_dim;
    const float cos = __bfloat162float(row[lane]);
    const float sin = __bfloat162float(row[half + lane]);
    const float x1 = normed[lane];
    const float x2 = normed[half + lane];
    // rotate_half pairs lane with lane + half and applies
    // (x * cos) + rotate_half(x) * sin. Ideogram duplicates the frequencies,
    // so both halves of a pair share one cos and one sin.
    out[i] = (i < half) ? (x1 * cos - x2 * sin) : (x2 * cos + x1 * sin);
  }
}
