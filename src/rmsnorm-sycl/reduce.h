// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#define MAX_THREADS_PER_BLOCK 1024

// Sum
template <typename T> struct SumOp {
    static T init() { return T(0); }
    static T op(const T &x, const T &y) { return x + y; }
};


/**
 * Warp Reduce and Block Reduce
 */
template <template <class> class Func, typename T, int THREADS_PER_WARP>
T WarpReduce(T val, sycl::nd_item<3> &item) {
    auto sg = item.get_sub_group();
#pragma unroll
    for (int offset = THREADS_PER_WARP >> 1; offset > 0; offset >>= 1) {
        T tmp = sycl::permute_group_by_xor(sg, val, offset);
        val   = Func<T>::op(tmp, val);
    }
    return val;
}

template <template <class> class Func, typename T, int THREADS_PER_WARP>
T BlockReduce(const T &val, T *smem, sycl::nd_item<3> &item) {
    const int num_warps =
        (item.get_local_range(2) + THREADS_PER_WARP - 1) / THREADS_PER_WARP;

    const int warp_id = item.get_local_id(2) / THREADS_PER_WARP;
    const int lane_id = item.get_local_id(2) % THREADS_PER_WARP;

    T val_reg = Func<T>::init();
    val_reg   = Func<T>::op(val_reg, val);
    val_reg   = WarpReduce<Func, T, THREADS_PER_WARP>(val_reg, item);
    if (lane_id == 0) {
        smem[warp_id] = val_reg;
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (warp_id == 0) {
        val_reg = (lane_id < num_warps) ? smem[lane_id] : Func<T>::init();
        val_reg = WarpReduce<Func, T, THREADS_PER_WARP>(val_reg, item);
        if (lane_id == 0)
            smem[0] = val_reg;
    }
    item.barrier(sycl::access::fence_space::local_space);
    return smem[0];
}
