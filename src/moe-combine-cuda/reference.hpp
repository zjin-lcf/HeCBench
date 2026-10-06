// Host routing and reference for the intra-node MoE combine.
// Round-robin dispatch matches MORI's benchmark initializer: expert slot
// (token * topk + k) is sent to rank (token * topk + k) % world, and each
// destination packs arrivals in source-rank order. Combine then reads those
// expert outputs back and applies the router's weights:
//   out[token] = sum_k weight[token, k] * expert_output[dest_rank, slot]
//
// Bandwidth follows the MORI EP benchmark. Algo bytes are
// recv_tokens * hidden * sizeof(bf16), including expert outputs that land on
// the same rank. Fabric bytes count only reads from other ranks.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COMBINE_HD __host__ __device__
#else
#define COMBINE_HD
#endif

struct CombineRoute {
  int world = 1;
  int tokens = 0;
  int topk = 1;
  int hidden = 1;
  int experts = 1;
  // dest/slot are indexed [source_rank, token, k].
  std::vector<int> dest;
  std::vector<int> slot;
  std::vector<int> recv;

  std::size_t index(int src, int token, int k) const {
    return (static_cast<std::size_t>(src) * tokens + token) * topk + k;
  }
};

COMBINE_HD inline int stage_mod(int rank, int slot, int h) {
  const std::int64_t m = static_cast<std::int64_t>(rank + 1) * 17 +
                         static_cast<std::int64_t>(slot) * 3 + (h % 251);
  return static_cast<int>(m % 251);
}

COMBINE_HD inline float stage_unit(int rank, int slot, int h) {
  return static_cast<float>(stage_mod(rank, slot, h)) / 251.0f;
}

COMBINE_HD inline float weight_of(int token, int k, int topk) {
  return static_cast<float>((token * topk + k) % 7 + 1) / 8.0f;
}

// Round-to-nearest-even bf16, matching the CUDA/HIP conversion for finite values.
inline std::uint16_t f32_to_bf16(float x) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

inline float bf16_to_f32(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float x = 0.f;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

inline CombineRoute build_route(int world, int tokens, int topk, int hidden,
                                int experts) {
  CombineRoute route;
  route.world = world;
  route.tokens = tokens;
  route.topk = topk;
  route.hidden = hidden;
  route.experts = experts;
  const std::size_t n =
      static_cast<std::size_t>(world) * tokens * topk;
  route.dest.assign(n, 0);
  route.slot.assign(n, 0);
  route.recv.assign(world, 0);
  for (int src = 0; src < world; ++src) {
    for (int token = 0; token < tokens; ++token) {
      for (int k = 0; k < topk; ++k) {
        const int disp = token * topk + k;
        const int dest = disp % world;
        const std::size_t id = route.index(src, token, k);
        route.dest[id] = dest;
        route.slot[id] = route.recv[dest]++;
      }
    }
  }
  return route;
}

inline std::size_t remote_slots(const CombineRoute &route, int rank) {
  std::size_t n = 0;
  for (int token = 0; token < route.tokens; ++token)
    for (int k = 0; k < route.topk; ++k)
      if (route.dest[route.index(rank, token, k)] != rank)
        ++n;
  return n;
}

// Largest absolute error over the first `checked` tokens and the last token.
inline double max_abs_error(const CombineRoute &route, int rank,
                            const std::uint16_t *got, int checked) {
  double error = 0.0;
  auto consider = [&](int token) {
    for (int h = 0; h < route.hidden; ++h) {
      float acc = 0.f;
      for (int k = 0; k < route.topk; ++k) {
        const std::size_t id = route.index(rank, token, k);
        const float x = bf16_to_f32(f32_to_bf16(
            stage_unit(route.dest[id], route.slot[id], h)));
        acc += weight_of(token, k, route.topk) * x;
      }
      const float expect = bf16_to_f32(f32_to_bf16(acc));
      const float actual =
          bf16_to_f32(got[static_cast<std::size_t>(token) * route.hidden + h]);
      const double diff = std::fabs(static_cast<double>(actual) -
                                    static_cast<double>(expect));
      if (!std::isfinite(diff))
        return diff;
      error = std::max(error, diff);
    }
    return 0.0;
  };
  const int n = std::min(checked, route.tokens);
  for (int token = 0; token < n; ++token) {
    const double bad = consider(token);
    if (bad != 0.0)
      return bad;
  }
  if (route.tokens > n) {
    const double bad = consider(route.tokens - 1);
    if (bad != 0.0)
      return bad;
  }
  return error;
}
