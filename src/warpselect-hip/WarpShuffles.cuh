/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include "DeviceDefs.cuh"

namespace faiss {
namespace gpu {

// defines to simplify the SASS assembly structure file/line in the profiler
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) __shfl(VAL, SRC_LANE, WIDTH)

template <typename T>
inline __device__ T shfl(const T val, int srcLane, int width = kWarpSize) {
    return __shfl(val, srcLane, width);
}

template <typename T>
inline __device__ T* shfl(T* const val, int srcLane, int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T
shfl_up(const T val, unsigned int delta, int width = kWarpSize) {
    return __shfl_up(val, delta, width);
}

template <typename T>
inline __device__ T* shfl_up(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl_up(v, delta, width);
}

template <typename T>
inline __device__ T
shfl_down(const T val, unsigned int delta, int width = kWarpSize) {
    return __shfl_down(val, delta, width);
}

template <typename T>
inline __device__ T* shfl_down(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
    return __shfl_xor(val, laneMask, width);
}

template <typename T>
inline __device__ T* shfl_xor(
        T* const val,
        int laneMask,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_xor(v, laneMask, width);
}

} // namespace gpu
} // namespace faiss
