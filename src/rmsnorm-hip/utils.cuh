// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

/**
 * Load & Store data utils
 */

// TODO: ASM
template <typename T, const int N>
__device__ void load_data(const T *src, T *dst) {
    constexpr int BYTES = N * sizeof(T);
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16,
                  "Only 1/2/4/8/16 bytes are supported.");
    if constexpr (BYTES == 1) {
        *reinterpret_cast<uint8_t *>(dst) = *(reinterpret_cast<const uint8_t *>(src));
    } else if constexpr (BYTES == 2) {
        *reinterpret_cast<uint16_t *>(dst) = *(reinterpret_cast<const uint16_t *>(src));
    } else if constexpr (BYTES == 4) {
        *reinterpret_cast<uint32_t *>(dst) = *(reinterpret_cast<const uint32_t *>(src));
    } else if constexpr (BYTES == 8) {
        *reinterpret_cast<uint64_t *>(dst) = *(reinterpret_cast<const uint64_t *>(src));
    } else if constexpr (BYTES == 16) {
        *reinterpret_cast<uint4 *>(dst) = *(reinterpret_cast<const uint4 *>(src));
    }
}

template <typename T, const int N>
__device__ void store_data(T *dst, const T *src) {
    constexpr int BYTES = N * sizeof(T);
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16,
                  "Only 1/2/4/8/16 bytes are supported.");

    if constexpr (BYTES == 1) {
        *reinterpret_cast<uint8_t *>(dst) = *reinterpret_cast<const uint8_t *>(src);
    } else if constexpr (BYTES == 2) {
        *reinterpret_cast<uint16_t *>(dst) = *reinterpret_cast<const uint16_t *>(src);
    } else if constexpr (BYTES == 4) {
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
    } else if constexpr (BYTES == 8) {
        *reinterpret_cast<uint64_t *>(dst) = *reinterpret_cast<const uint64_t *>(src);
    } else if constexpr (BYTES == 16) {
        *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
    }
}
