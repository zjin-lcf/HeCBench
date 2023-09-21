/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ConversionOperators.cuh"

//
// Templated wrappers to express math for different scalar and vector
// types, so kernels can have the same written form but can operate
// over floating-point types, and on vector types transparently
//

namespace faiss {
namespace gpu {

template <typename T>
struct Math {
    typedef T ScalarType;

    static inline __device__ T add(T a, T b) {
        return a + b;
    }

    static inline __device__ T sub(T a, T b) {
        return a - b;
    }

    static inline __device__ T mul(T a, T b) {
        return a * b;
    }

    static inline __device__ T neg(T v) {
        return -v;
    }

    /// For a vector type, this is a horizontal add, returning sum(v_i)
    static inline __device__ float reduceAdd(T v) {
        return ConvertTo<float>::to(v);
    }

    static inline __device__ bool lt(T a, T b) {
        return a < b;
    }

    static inline __device__ bool gt(T a, T b) {
        return a > b;
    }

    static inline __device__ bool eq(T a, T b) {
        return a == b;
    }

    static inline __device__ T zero() {
        return (T)0;
    }
};

template <>
struct Math<float2> {
    typedef float ScalarType;

    static inline __device__ float2 add(float2 a, float2 b) {
        float2 v;
        v.x = a.x + b.x;
        v.y = a.y + b.y;
        return v;
    }

    static inline __device__ float2 sub(float2 a, float2 b) {
        float2 v;
        v.x = a.x - b.x;
        v.y = a.y - b.y;
        return v;
    }

    static inline __device__ float2 add(float2 a, float b) {
        float2 v;
        v.x = a.x + b;
        v.y = a.y + b;
        return v;
    }

    static inline __device__ float2 sub(float2 a, float b) {
        float2 v;
        v.x = a.x - b;
        v.y = a.y - b;
        return v;
    }

    static inline __device__ float2 mul(float2 a, float2 b) {
        float2 v;
        v.x = a.x * b.x;
        v.y = a.y * b.y;
        return v;
    }

    static inline __device__ float2 mul(float2 a, float b) {
        float2 v;
        v.x = a.x * b;
        v.y = a.y * b;
        return v;
    }

    static inline __device__ float2 neg(float2 v) {
        v.x = -v.x;
        v.y = -v.y;
        return v;
    }

    /// For a vector type, this is a horizontal add, returning sum(v_i)
    static inline __device__ float reduceAdd(float2 v) {
        return v.x + v.y;
    }

    // not implemented for vector types
    // static inline __device__ bool lt(float2 a, float2 b);
    // static inline __device__ bool gt(float2 a, float2 b);
    // static inline __device__ bool eq(float2 a, float2 b);

    static inline __device__ float2 zero() {
        float2 v;
        v.x = 0.0f;
        v.y = 0.0f;
        return v;
    }
};

template <>
struct Math<float4> {
    typedef float ScalarType;

    static inline __device__ float4 add(float4 a, float4 b) {
        float4 v;
        v.x = a.x + b.x;
        v.y = a.y + b.y;
        v.z = a.z + b.z;
        v.w = a.w + b.w;
        return v;
    }

    static inline __device__ float4 sub(float4 a, float4 b) {
        float4 v;
        v.x = a.x - b.x;
        v.y = a.y - b.y;
        v.z = a.z - b.z;
        v.w = a.w - b.w;
        return v;
    }

    static inline __device__ float4 add(float4 a, float b) {
        float4 v;
        v.x = a.x + b;
        v.y = a.y + b;
        v.z = a.z + b;
        v.w = a.w + b;
        return v;
    }

    static inline __device__ float4 sub(float4 a, float b) {
        float4 v;
        v.x = a.x - b;
        v.y = a.y - b;
        v.z = a.z - b;
        v.w = a.w - b;
        return v;
    }

    static inline __device__ float4 mul(float4 a, float4 b) {
        float4 v;
        v.x = a.x * b.x;
        v.y = a.y * b.y;
        v.z = a.z * b.z;
        v.w = a.w * b.w;
        return v;
    }

    static inline __device__ float4 mul(float4 a, float b) {
        float4 v;
        v.x = a.x * b;
        v.y = a.y * b;
        v.z = a.z * b;
        v.w = a.w * b;
        return v;
    }

    static inline __device__ float4 neg(float4 v) {
        v.x = -v.x;
        v.y = -v.y;
        v.z = -v.z;
        v.w = -v.w;
        return v;
    }

    /// For a vector type, this is a horizontal add, returning sum(v_i)
    static inline __device__ float reduceAdd(float4 v) {
        return v.x + v.y + v.z + v.w;
    }

    // not implemented for vector types
    // static inline __device__ bool lt(float4 a, float4 b);
    // static inline __device__ bool gt(float4 a, float4 b);
    // static inline __device__ bool eq(float4 a, float4 b);

    static inline __device__ float4 zero() {
        float4 v;
        v.x = 0.0f;
        v.y = 0.0f;
        v.z = 0.0f;
        v.w = 0.0f;
        return v;
    }
};

} // namespace gpu
} // namespace faiss
