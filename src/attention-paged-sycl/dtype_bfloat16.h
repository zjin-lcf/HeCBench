/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "attention_generic.h"

#include <stdint.h>

//namespace vllm {

using Bfloat16 = sycl::ext::oneapi::bfloat16;

// Define custom BF16 vector data types.
struct bf16_4_t {
  sycl::vec<Bfloat16, 2> x;
  sycl::vec<Bfloat16, 2> y;
};

struct bf16_8_t {
  sycl::vec<Bfloat16, 2> x;
  sycl::vec<Bfloat16, 2> y;
  sycl::vec<Bfloat16, 2> z;
  sycl::vec<Bfloat16, 2> w;
};

// BF16 vector types for Q, K, V.
template <> struct Vec<Bfloat16, 1> {
  using Type = Bfloat16;
};
template <> struct Vec<Bfloat16, 2> {
  using Type = sycl::vec<Bfloat16, 2>;
};
template <> struct Vec<Bfloat16, 4> {
  using Type = bf16_4_t;
};
template <> struct Vec<Bfloat16, 8> {
  using Type = bf16_8_t;
};

// FP32 accumulator vector types corresponding to Vec.
template <> struct FloatVec<Bfloat16> {
  using Type = float;
};
template <> struct FloatVec<sycl::vec<Bfloat16, 2>> {
  using Type = sycl::float2;
};
template <>
struct FloatVec<bf16_4_t> {
  using Type = Float4_;
};
template <>
struct FloatVec<bf16_8_t> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline sycl::float2
bf1622float2(const sycl::vec<Bfloat16, 2> val) {
  return sycl::float2(val.x(), val.y());
}

inline sycl::vec<Bfloat16, 2>
bf162bf162(const Bfloat16 val) {
  return sycl::vec<Bfloat16, 2>(val, val);
}

// Vector addition.
inline Bfloat16 add(Bfloat16 a,
                                       Bfloat16 b) {
  return a + b;
}

inline sycl::vec<Bfloat16, 2>
add(sycl::vec<Bfloat16, 2> a,
    sycl::vec<Bfloat16, 2> b) {
  return a + b;
}

inline bf16_4_t add(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline bf16_8_t add(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline sycl::float2 add(sycl::vec<Bfloat16, 2> a,
                        sycl::float2 fb) {
  sycl::float2 fa = bf1622float2(a);
  return add(fa, fb);
}

inline Float4_ add(bf16_4_t a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline Float8_ add(bf16_8_t a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
template <>
inline Bfloat16 mul(Bfloat16 a, Bfloat16 b) {
  return a * b;
}

template <>
inline sycl::vec<Bfloat16, 2>
mul(sycl::vec<Bfloat16, 2> a,
    sycl::vec<Bfloat16, 2> b) {
  return a * b;
}

template <>
inline sycl::vec<Bfloat16, 2>
mul(Bfloat16 a, sycl::vec<Bfloat16, 2> b) {
  return mul<sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(bf162bf162(a), b);
}

template <>
inline bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(a.x, b.x);
  c.y = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(a.y, b.y);
  return c;
}

template <> inline bf16_4_t mul(Bfloat16 a, bf16_4_t b) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  bf16_4_t c;
  c.x = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(s, b.x);
  c.y = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(s, b.y);
  return c;
}

template <>
inline bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(a.x, b.x);
  c.y = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(a.y, b.y);
  c.z = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(a.z, b.z);
  c.w = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(a.w, b.w);
  return c;
}

template <> inline bf16_8_t mul(Bfloat16 a, bf16_8_t b) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  bf16_8_t c;
  c.x = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(s, b.x);
  c.y = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(s, b.y);
  c.z = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(s, b.z);
  c.w = mul<sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>,
            sycl::vec<Bfloat16, 2>>(s, b.w);
  return c;
}

template <>
inline float mul(Bfloat16 a, Bfloat16 b) {
  float fa = static_cast<float>(a);
  float fb = static_cast<float>(b);
  return fa * fb;
}

template <>
inline sycl::float2 mul(sycl::vec<Bfloat16, 2> a,
                        sycl::vec<Bfloat16, 2> b) {
  sycl::float2 fa = bf1622float2(a);
  sycl::float2 fb = bf1622float2(b);
  return mul<sycl::float2, sycl::float2, sycl::float2>(fa, fb);
}

template <>
inline sycl::float2 mul(Bfloat16 a,
                        sycl::vec<Bfloat16, 2> b) {
  return mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(bf162bf162(a), b);
}

template <>
inline Float4_ mul(bf16_4_t a, bf16_4_t b) {
  Float4_ fc;
  fc.x = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(a.x, b.x);
  fc.y = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(a.y, b.y);
  return fc;
}

template <> inline Float4_ mul(Bfloat16 a, bf16_4_t b) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  Float4_ fc;
  fc.x = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(s, b.x);
  fc.y = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(s, b.y);
  return fc;
}

template <>
inline Float8_ mul(bf16_8_t a, bf16_8_t b) {
  Float8_ fc;
  fc.x = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(a.x, b.x);
  fc.y = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(a.y, b.y);
  fc.z = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(a.z, b.z);
  fc.w = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(a.w, b.w);
  return fc;
}

template <> inline Float8_ mul(Bfloat16 a, bf16_8_t b) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  Float8_ fc;
  fc.x = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(s, b.x);
  fc.y = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(s, b.y);
  fc.z = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(s, b.z);
  fc.w = mul<sycl::float2, sycl::vec<Bfloat16, 2>,
             sycl::vec<Bfloat16, 2>>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline sycl::vec<Bfloat16, 2>
vfma(sycl::vec<Bfloat16, 2> a,
     sycl::vec<Bfloat16, 2> b,
     sycl::vec<Bfloat16, 2> c) {
  return a * b + c;
}

inline sycl::vec<Bfloat16, 2>
vfma(Bfloat16 a, sycl::vec<Bfloat16, 2> b,
     sycl::vec<Bfloat16, 2> c) {
  return bf162bf162(a) * b + c;
}

inline bf16_4_t vfma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
  bf16_4_t d;
  d.x = vfma(a.x, b.x, c.x);
  d.y = vfma(a.y, b.y, c.y);
  return d;
}

inline bf16_4_t vfma(Bfloat16 a, bf16_4_t b, bf16_4_t c) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  bf16_4_t d;
  d.x = vfma(s, b.x, c.x);
  d.y = vfma(s, b.y, c.y);
  return d;
}

inline bf16_8_t vfma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
  bf16_8_t d;
  d.x = vfma(a.x, b.x, c.x);
  d.y = vfma(a.y, b.y, c.y);
  d.z = vfma(a.z, b.z, c.z);
  d.w = vfma(a.w, b.w, c.w);
  return d;
}

inline bf16_8_t vfma(Bfloat16 a, bf16_8_t b, bf16_8_t c) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  bf16_8_t d;
  d.x = vfma(s, b.x, c.x);
  d.y = vfma(s, b.y, c.y);
  d.z = vfma(s, b.z, c.z);
  d.w = vfma(s, b.w, c.w);
  return d;
}

inline float vfma(Bfloat16 a, Bfloat16 b,
                  float fc) {
  return static_cast<float>(a) * static_cast<float>(b) + fc;
}

inline sycl::float2 vfma(sycl::vec<Bfloat16, 2> a,
                         sycl::vec<Bfloat16, 2> b,
                         sycl::float2 fc) {
  sycl::float2 fa = bf1622float2(a);
  sycl::float2 fb = bf1622float2(b);
  return vfma(fa, fb, fc);
}

inline sycl::float2 vfma(Bfloat16 a,
                         sycl::vec<Bfloat16, 2> b,
                         sycl::float2 fc) {
  return vfma(bf162bf162(a), b, fc);
}

inline Float4_ vfma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
  Float4_ fd;
  fd.x = vfma(a.x, b.x, fc.x);
  fd.y = vfma(a.y, b.y, fc.y);
  return fd;
}

inline Float4_ vfma(Bfloat16 a, bf16_4_t b, Float4_ fc) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  Float4_ fd;
  fd.x = vfma(s, b.x, fc.x);
  fd.y = vfma(s, b.y, fc.y);
  return fd;
}

inline Float8_ vfma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
  Float8_ fd;
  fd.x = vfma(a.x, b.x, fc.x);
  fd.y = vfma(a.y, b.y, fc.y);
  fd.z = vfma(a.z, b.z, fc.z);
  fd.w = vfma(a.w, b.w, fc.w);
  return fd;
}

inline Float8_ vfma(Bfloat16 a, bf16_8_t b, Float8_ fc) {
  sycl::vec<Bfloat16, 2> s = bf162bf162(a);
  Float8_ fd;
  fd.x = vfma(s, b.x, fc.x);
  fd.y = vfma(s, b.y, fc.y);
  fd.z = vfma(s, b.z, fc.z);
  fd.w = vfma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template <> inline float sum(Bfloat16 v) {
  return static_cast<float>(v);
}

template <> inline float sum(sycl::vec<Bfloat16, 2> v) {
  sycl::float2 vf = bf1622float2(v);
  return vf.x() + vf.y();
}

template <>
inline float sum(bf16_4_t v) {
  return sum(v.x) + sum(v.y);
}

template <>
inline float sum(bf16_8_t v) {
  return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
}

// From float32 to bfloat16.
inline void from_float(Bfloat16 &dst, float src) {
  dst = Bfloat16(src);
}

inline void from_float(sycl::vec<Bfloat16, 2> &dst,
                       sycl::float2 src) {
  dst = sycl::vec<Bfloat16, 2>(src.x(), src.y());
}

inline void from_float(bf16_4_t& dst, Float4_ src) {
  dst.x = sycl::vec<Bfloat16, 2>((src.x).x(), (src.x).y());
  dst.y = sycl::vec<Bfloat16, 2>((src.y).x(), (src.y).y());
}

inline void from_float(bf16_8_t& dst, Float8_ src) {
  dst.x = sycl::vec<Bfloat16, 2>((src.x).x(), (src.x).y());
  dst.y = sycl::vec<Bfloat16, 2>((src.y).x(), (src.y).y());
  dst.z = sycl::vec<Bfloat16, 2>((src.z).x(), (src.z).y());
  dst.w = sycl::vec<Bfloat16, 2>((src.w).x(), (src.w).y());
}

// From bfloat16 to float32.
inline float to_float(Bfloat16 u) {
  return static_cast<float>(u);
}

// Zero-out a variable.
inline void zero(Bfloat16 &dst) {
  // Same as CUDART_ZERO_BF16 introduced in CUDA 12.2.
  //dst = __ushort_as_bfloat16((unsigned short)0x0000U);
  dst = 0;
}

//}  // namespace vllm
