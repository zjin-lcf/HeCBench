/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

namespace faiss {
namespace gpu {

// We validate this against the actual architecture in device initialization
constexpr int kWarpSize = 32;

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {
    // For the time being, assume synchronicity.
     __threadfence_block();
}

#define GPU_MAX_SELECTION_K 2048

} // namespace gpu
} // namespace faiss
