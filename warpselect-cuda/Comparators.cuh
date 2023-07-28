/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace faiss {
namespace gpu {

template <typename T>
struct Comparator {
    __device__ static inline bool lt(T a, T b) {
        return a < b;
    }

    __device__ static inline bool gt(T a, T b) {
        return a > b;
    }
};

} // namespace gpu
} // namespace faiss
