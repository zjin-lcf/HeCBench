/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#define FULL_MASK 0xffffffff

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint8 {
    sycl::uint4 u;
    sycl::uint4 v;
};

template<int BYTES> struct BytesToType {};

template<>
struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
};

template<> struct BytesToType<16> {
    using Type = sycl::uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// https://stackoverflow.com/questions/35311711/whats-the-right-way-to-compute-integral-base-2-logarithms-at-compile-time
constexpr int cilog2(int val) { return val > 0 ? 1 + cilog2(val >> 1) : -1; }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kLogN, int kNChunks>
inline void hadamard_mult_thread(float x[kNChunks][1 << kLogN]) {
    constexpr int N = 1 << kLogN;
    #pragma unroll
    for (int i = 0; i < kLogN; ++i) {
        const int stride = 1 << i;
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int c = 0; c < kNChunks; ++c) {
                const float a = x[c][idx];
                const float b = x[c][idx + stride];
                x[c][idx] = a + b;
                x[c][idx + stride] = a - b;
            }
        }
    }
}

template <int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
inline void hadamard_mult_warp(float x[kNChunks][kNItems], sycl::nd_item<3> &item) {
    //constexpr int N = 1 << kLogWarpSize;
    //int lane_id = item.get_local_id(2) % N;
    auto sg = item.get_sub_group();
    int lane_id = sg.get_local_linear_id();
#pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int lane_mask = 1 << step;
        const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float x_val_other = sycl::permute_group_by_xor(sg, x[c][i], lane_mask);
                x[c][i] = sign * x[c][i] + x_val_other;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNChunks, int kNElts, typename input_t>
inline void load_input(input_t *x, float x_vals[kNChunks][kNElts], int dim, sycl::nd_item<3> &item) {
    using vec_t = typename BytesToType<sizeof(input_t) * kNElts>::Type;
    input_t x_vals_load[kNChunks][kNElts] = {{0}};
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * item.get_local_range(2) + item.get_local_id(2)) * kNElts < dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[c] =
                reinterpret_cast<const vec_t *>(x)[c * item.get_local_range(2) +
                       item.get_local_id(2)];
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = float(x_vals_load[c][i]); }
    }
}

template <int kNChunks, int kNElts, typename output_t>
inline void store_output(output_t *out, float out_vals[kNChunks][kNElts],
                         int dim, float scale, sycl::nd_item<3> &item) {
    using vec_t = typename BytesToType<sizeof(output_t) * kNElts>::Type;
    output_t out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { out_vals_store[c][i] = out_vals[c][i] * scale; }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * item.get_local_range(2) + item.get_local_id(2)) *
                kNElts < dim) {
            reinterpret_cast<vec_t *>(out)[c * item.get_local_range(2) + item.get_local_id(2)] =
                reinterpret_cast<const vec_t *>(out_vals_store)[c];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Pre=true means the exchange before the hadamard_mult_warp, Pre=false means after.
template <int kNChunks, int kChunksPerExchange, int kNElts, int kWarpSize, int kNWarps, bool Pre, typename vec_t>
inline void exchange_smem_pre(float x_vals[kNChunks][kNElts], vec_t *smem, sycl::nd_item<3> &item) {
    constexpr int kNThreads = kWarpSize * kNWarps;
    constexpr int kNExchangePerVec = kNElts / (sizeof(vec_t) / sizeof(float));
    const int warp_id = item.get_local_id(2) / kWarpSize;
    const int lane_id = item.get_local_id(2) % kWarpSize;
    const int row_t = item.get_local_id(2) % kNWarps;
    const int col_t = item.get_local_id(2) / kNWarps;
    // We use the XOR swizzle trick (new_col = col ^ row) to avoid / reduce smem bank conflicts.
    #pragma unroll
    for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            #pragma unroll
            for (int r = 0; r < kNExchangePerVec; ++r) {
                smem[(c1 * kNExchangePerVec + r) * kNThreads + (Pre ? warp_id * kWarpSize + lane_id ^ warp_id : row_t * kWarpSize + col_t ^ row_t)] = reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            #pragma unroll
            for (int r = 0; r < kNExchangePerVec; ++r) {
                reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] = smem[(c1 * kNExchangePerVec + r) * kNThreads + (Pre ? row_t * kWarpSize + col_t ^ row_t : warp_id * kWarpSize + lane_id ^ warp_id)];
            }
        }
    }
}
