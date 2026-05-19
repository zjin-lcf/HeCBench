/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#include <sycl/sycl.hpp>

#include <type_traits>

namespace flashinfer {

#define FLASHINFER_INLINE inline __attribute__((always_inline)) 


/******************* vec_t type cast *******************/

template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(dst_t* dst, const src_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = (dst_t)src[i];
    }
  }
};

template <> struct vec_cast<float, sycl::half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float *dst, const sycl::half *src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::float2 *)dst)[i] =
            ((sycl::half2 *)src)[i]
                .convert<float, sycl::rounding_mode::automatic>();
      }
    }
  }
};

template <> struct vec_cast<sycl::half, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::half *dst, const float *src) {
    if constexpr (vec_size == 1) {
      dst[0] = sycl::vec<float, 1>(src[0])
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::half2 *)dst)[i] =
            ((sycl::float2 *)src)[i]
                .convert<sycl::half, sycl::rounding_mode::rte>();
      }
    }
  }
};

template <typename T>
constexpr FLASHINFER_INLINE int get_exponent_bits() {
  if constexpr (std::is_same_v<T, sycl::half>) {
    return 5;
  } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
    return 8;
  }
}

template <typename T>
constexpr FLASHINFER_INLINE int get_mantissa_bits() {
  if constexpr (std::is_same_v<T, sycl::half>) {
    return 11;
  } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
    return 7;
  }
}

template <> struct vec_cast<float, sycl::ext::oneapi::bfloat16> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float *dst,
                                     const sycl::ext::oneapi::bfloat16 *src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::float2 *)dst)[i] = sycl::float2(
            ((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)src)[i][0],
            ((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)src)[i][1]);
      }
    }
  }
};

template <> struct vec_cast<sycl::ext::oneapi::bfloat16, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(sycl::ext::oneapi::bfloat16 *dst,
                                     const float *src) {
    if constexpr (vec_size == 1) {
      dst[0] = sycl::ext::oneapi::bfloat16(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)dst)[i] =
            sycl::marray<sycl::ext::oneapi::bfloat16, 2>(
                ((sycl::float2 *)src)[i][0], ((sycl::float2 *)src)[i][1]);
      }
    }
  }
};

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE void fill(float_t val);
  FLASHINFER_INLINE void load(const float_t* ptr);
  FLASHINFER_INLINE void store(float_t* ptr) const;
  //FLASHINFER_INLINE void load_global_acquire(float* addr);
  //FLASHINFER_INLINE void store_global_release(float* addr) const;
  //FLASHINFER_INLINE void load_global_volatile(float* addr);
  //FLASHINFER_INLINE void store_global_volatile(float* addr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr);
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const;
  FLASHINFER_INLINE static void memcpy(float_t* dst, const float_t* src);
  FLASHINFER_INLINE float_t* ptr();
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const vec_t<src_float_t, vec_size>& src) {
  // ./vec_dtypes.dp.hpp:199:3: error: missing 'template' keyword prior to dependent template name 'cast'
  vec_cast<tgt_float_t, src_float_t>::template cast<vec_size>(
      dst.ptr(), const_cast<vec_t<src_float_t, vec_size>*>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const src_float_t* src_ptr) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(tgt_float_t* dst_ptr,
                                       const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}


/******************* vec_t<half> *******************/

// half x 1
template <> struct vec_t<sycl::half, 1> {
  sycl::half data;

  FLASHINFER_INLINE sycl::half &operator[](size_t i) {
    return ((sycl::half *)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::half &operator[](size_t i) const {
    return ((const sycl::half *)(&data))[i];
  }
  FLASHINFER_INLINE sycl::half *ptr() {
    return reinterpret_cast<sycl::half *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::half val);
  FLASHINFER_INLINE void load(const sycl::half *ptr);
  FLASHINFER_INLINE void store(sycl::half *ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(sycl::half *dst, const sycl::half *src);
};

FLASHINFER_INLINE void vec_t<sycl::half, 1>::fill(sycl::half val) {
  data = val;
}

FLASHINFER_INLINE void vec_t<sycl::half, 1>::load(const sycl::half *ptr) {
  data = *ptr;
}

FLASHINFER_INLINE void vec_t<sycl::half, 1>::store(sycl::half *ptr) const {
      *ptr = data;
}

FLASHINFER_INLINE void vec_t<sycl::half, 1>::memcpy(sycl::half *dst,
                                                    const sycl::half *src) {
      *dst = *src;
}

// half x 2
template <> struct vec_t<sycl::half, 2> {
  sycl::half2 data;

  FLASHINFER_INLINE sycl::half &operator[](size_t i) {
    return ((sycl::half *)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::half &operator[](size_t i) const {
    return ((const sycl::half *)(&data))[i];
  }
  FLASHINFER_INLINE sycl::half *ptr() {
    return reinterpret_cast<sycl::half *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::half val);
  FLASHINFER_INLINE void load(const sycl::half *ptr);
  FLASHINFER_INLINE void store(sycl::half *ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(sycl::half *dst, const sycl::half *src);
};

FLASHINFER_INLINE void vec_t<sycl::half, 2>::fill(sycl::half val) {
  data = sycl::half2(val, val);
}

FLASHINFER_INLINE void vec_t<sycl::half, 2>::load(const sycl::half *ptr) {
  data = *((sycl::half2 *)ptr);
}

FLASHINFER_INLINE void vec_t<sycl::half, 2>::store(sycl::half *ptr) const {
  *((sycl::half2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<sycl::half, 2>::memcpy(sycl::half *dst,
                                                    const sycl::half *src) {
  *((sycl::half2 *)dst) = *((sycl::half2 *)src);
}

// half x 4

template <> struct vec_t<sycl::half, 4> {
  sycl::uint2 data;

  FLASHINFER_INLINE sycl::half &operator[](size_t i) {
    return ((sycl::half *)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::half &operator[](size_t i) const {
    return ((const sycl::half *)(&data))[i];
  }
  FLASHINFER_INLINE sycl::half *ptr() {
    return reinterpret_cast<sycl::half *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::half val);
  FLASHINFER_INLINE void load(const sycl::half *ptr);
  FLASHINFER_INLINE void store(sycl::half *ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::half *dst, const sycl::half *src);
};

FLASHINFER_INLINE void vec_t<sycl::half, 4>::fill(sycl::half val) {
  *(sycl::half2 *)(&data.x()) = sycl::half2(val, val);
  *(sycl::half2 *)(&data.y()) = sycl::half2(val, val);
}

FLASHINFER_INLINE void vec_t<sycl::half, 4>::load(const sycl::half *ptr) {
  data = *((sycl::uint2 *)ptr);
}

FLASHINFER_INLINE void vec_t<sycl::half, 4>::store(sycl::half *ptr) const {
  *((sycl::uint2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<sycl::half, 4>::memcpy(sycl::half *dst,
                                                    const sycl::half *src) {
  *((sycl::uint2 *)dst) = *((sycl::uint2 *)src);
}

// half x 8 or more

template <size_t vec_size> struct vec_t<sycl::half, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  sycl::int4 data[vec_size / 8];
  FLASHINFER_INLINE sycl::half &operator[](size_t i) {
    return ((sycl::half *)data)[i];
  }
  FLASHINFER_INLINE const sycl::half &operator[](size_t i) const {
    return ((const sycl::half *)data)[i];
  }
  FLASHINFER_INLINE sycl::half *ptr() {
    return reinterpret_cast<sycl::half *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(sycl::half2 *)(&(data[i].x())) = sycl::half2(val, val);
      *(sycl::half2 *)(&(data[i].y())) = sycl::half2(val, val);
      *(sycl::half2 *)(&(data[i].z())) = sycl::half2(val, val);
      *(sycl::half2 *)(&(data[i].w())) = sycl::half2(val, val);
    }
  }
  FLASHINFER_INLINE void load(const sycl::half *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((sycl::int4 *)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(sycl::half *ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::int4 *)ptr)[i] = data[i];
    }
  }
/*
  FLASHINFER_INLINE void load_global_acquire(half* addr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ld_global_acquire((int4*)(addr + i * 8));
    }
  }
  FLASHINFER_INLINE void store_global_release(half* addr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      st_global_release(data[i], (int4*)(addr + i * 8));
    }
  }
  FLASHINFER_INLINE void store_global_volatile(half* addr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      st_global_volatile(data[i], (int4*)(addr + i * 8));
    }
  }
  FLASHINFER_INLINE void load_global_volatile(half* addr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ld_global_volatile((int4*)(addr + i * 8));
    }
  }
*/

  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::half *dst, const sycl::half *src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::int4 *)dst)[i] = ((sycl::int4 *)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <> struct vec_t<sycl::ext::oneapi::bfloat16, 1> {
  sycl::ext::oneapi::bfloat16 data;
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 &operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16 *)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16 &
  operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16 *)(&data))[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 *ptr() {
    return reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val);
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16 *ptr);
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16 *ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16 *dst,
                                       const sycl::ext::oneapi::bfloat16 *src);
};

FLASHINFER_INLINE void
vec_t<sycl::ext::oneapi::bfloat16, 1>::fill(sycl::ext::oneapi::bfloat16 val) {
  data = val;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::load(
    const sycl::ext::oneapi::bfloat16 *ptr) {
  data = *ptr;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::store(
    sycl::ext::oneapi::bfloat16 *ptr) const {
      *ptr = data;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 1>::memcpy(
    sycl::ext::oneapi::bfloat16 *dst, const sycl::ext::oneapi::bfloat16 *src) {
  *dst = *src;
}

// nv_bfloat16 x 2
template <> struct vec_t<sycl::ext::oneapi::bfloat16, 2> {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> data;

  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 &operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16 *)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16 &
  operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16 *)(&data))[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 *ptr() {
    return reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val);
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16 *ptr);
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16 *ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16 *dst,
                                       const sycl::ext::oneapi::bfloat16 *src);
};

FLASHINFER_INLINE void
vec_t<sycl::ext::oneapi::bfloat16, 2>::fill(sycl::ext::oneapi::bfloat16 val) {
  data = {val, val};
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::load(
    const sycl::ext::oneapi::bfloat16 *ptr) {
  data = *((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)ptr);
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::store(
    sycl::ext::oneapi::bfloat16 *ptr) const {
  *((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 2>::memcpy(
    sycl::ext::oneapi::bfloat16 *dst, const sycl::ext::oneapi::bfloat16 *src) {
  *((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)dst) =
      *((sycl::marray<sycl::ext::oneapi::bfloat16, 2> *)src);
}

// nv_bfloat16 x 4

template <> struct vec_t<sycl::ext::oneapi::bfloat16, 4> {
  sycl::uint2 data;

  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 &operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16 *)(&data))[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16 &
  operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16 *)(&data))[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 *ptr() {
    return reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val);
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16 *ptr);
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16 *ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16 *dst,
                                       const sycl::ext::oneapi::bfloat16 *src);
};

FLASHINFER_INLINE void
vec_t<sycl::ext::oneapi::bfloat16, 4>::fill(sycl::ext::oneapi::bfloat16 val) {
  for (int i = 0; i < 4; i++)
    data[i] = val;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::load(
    const sycl::ext::oneapi::bfloat16 *ptr) {
  data = *((sycl::uint2 *)ptr);
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::store(
    sycl::ext::oneapi::bfloat16 *ptr) const {
  *((sycl::uint2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<sycl::ext::oneapi::bfloat16, 4>::memcpy(
    sycl::ext::oneapi::bfloat16 *dst, const sycl::ext::oneapi::bfloat16 *src) {
  *((sycl::uint2 *)dst) = *((sycl::uint2 *)src);
}

// nv_bfloat16 x 8 or more

template <size_t vec_size> struct vec_t<sycl::ext::oneapi::bfloat16, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  sycl::int4 data[vec_size / 8];

  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 &operator[](size_t i) {
    return ((sycl::ext::oneapi::bfloat16 *)data)[i];
  }
  FLASHINFER_INLINE const sycl::ext::oneapi::bfloat16 &
  operator[](size_t i) const {
    return ((const sycl::ext::oneapi::bfloat16 *)data)[i];
  }
  FLASHINFER_INLINE sycl::ext::oneapi::bfloat16 *ptr() {
    return reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(&data);
  }
  FLASHINFER_INLINE void fill(sycl::ext::oneapi::bfloat16 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
       data[i] = val;
    }
  }
  FLASHINFER_INLINE void load(const sycl::ext::oneapi::bfloat16 *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((sycl::int4 *)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(sycl::ext::oneapi::bfloat16 *ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::int4 *)ptr)[i] = data[i];
    }
  }
/*
  FLASHINFER_INLINE void store_global_release(nv_bfloat16* addr) const {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      st_global_release(data[i], (int4*)(addr + i * 8));
    }
  }
  FLASHINFER_INLINE void load_global_acquire(nv_bfloat16* addr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ld_global_acquire((int4*)(addr + i * 8));
    }
  }
  FLASHINFER_INLINE void store_global_volatile(nv_bfloat16* addr) const {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      st_global_volatile(data[i], (int4*)(addr + i * 8));
    }
  }
  FLASHINFER_INLINE void load_global_volatile(nv_bfloat16* addr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ld_global_volatile((int4*)(addr + i * 8));
    }
  }
*/
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(sycl::ext::oneapi::bfloat16 *dst,
                                       const sycl::ext::oneapi::bfloat16 *src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((sycl::int4 *)dst)[i] = ((sycl::int4 *)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

FLASHINFER_INLINE void vec_t<float, 1>::load(const float* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<float, 1>::store(float* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<float, 1>::memcpy(float* dst, const float* src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  sycl::float2 data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 2>::fill(float val) {
  data = sycl::float2(val, val);
}

FLASHINFER_INLINE void vec_t<float, 2>::load(const float *ptr) {
  data = *((sycl::float2 *)ptr);
}

FLASHINFER_INLINE void vec_t<float, 2>::store(float *ptr) const {
  *((sycl::float2 *)ptr) = data;
}

FLASHINFER_INLINE void vec_t<float, 2>::memcpy(float* dst, const float* src) {
  *((sycl::float2 *)dst) = *((sycl::float2 *)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  sycl::float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = sycl::float4(val, val, val, val);
    }
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((sycl::float4 *)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((sycl::float4 *)ptr)[i] = data[i];
    }
  }
/*
  FLASHINFER_INLINE void store_global_release(float* addr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      st_global_release(*(int4*)(data + i), (int4*)(addr + i * 4));
    }
  }
  FLASHINFER_INLINE void load_global_acquire(float* addr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      *((int4*)(data + i)) = ld_global_acquire((int4*)(addr + i * 4));
    }
  }
  FLASHINFER_INLINE void store_global_volatile(float* addr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      st_global_volatile(*(int4*)(data + i), (int4*)(addr + i * 4));
    }
  }
  FLASHINFER_INLINE void load_global_volatile(float* addr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      *((int4*)(data + i)) = ld_global_volatile((int4*)(addr + i * 4));
    }
  }
*/
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((sycl::float4 *)dst)[i] = ((sycl::float4 *)src)[i];
    }
  }
};

template <typename T>
struct vec2_dtype {
  using type = T;
};

template <> struct vec2_dtype<sycl::half> {
  using type = sycl::half2;
};

template <> struct vec2_dtype<sycl::ext::oneapi::bfloat16> {
  using type = sycl::marray<sycl::ext::oneapi::bfloat16, 2>;
};

template <typename T>
using vec2_dtype_t = typename vec2_dtype<T>::type;

template <typename T, size_t VEC_SIZE>
FLASHINFER_INLINE vec2_dtype_t<T> get_vec2_element(vec_t<T, VEC_SIZE>& vec, int i) {
  static_assert(VEC_SIZE % 2 == 0, "VEC_SIZE must be a multiple of 2");
  return ((vec2_dtype_t<T>*)&(vec[0]))[i];
}

}  // namespace flashinfer

#endif  // VEC_DTYPES_CUH_
