/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

template <typename T> inline T floatToType2(float a);

// Converts input to half precision in round-to-nearest-even mode
// and populates both halves of half2 with converted value.
template <> inline sycl::half2 floatToType2(float a) {
  return sycl::float2{a, a}.convert<sycl::half, sycl::rounding_mode::rte>();
}

template <>
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2> floatToType2(float a) {
  return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(a, a);
}

template <typename T> inline T float2ToType2(sycl::float2 a) {
  return a;
}

// Converts both components of float2 to half precision in round-to-nearest mode
// and combines the results into one half2 number.
template <> inline sycl::half2 float2ToType2(sycl::float2 a) {
  return a.convert<sycl::half, sycl::rounding_mode::rte>();
}

template <>
inline sycl::marray<sycl::ext::oneapi::bfloat16, 2>
float2ToType2(sycl::float2 a) {
  return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(a[0], a[1]);
}

template <typename T> inline sycl::float2 type2ToFloat2(T a) {
  return a;
}

template <> inline sycl::float2 type2ToFloat2(sycl::half2 a) {
  return a.convert<float, sycl::rounding_mode::automatic>();
}

template <>
inline sycl::float2
type2ToFloat2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> a) {
  return sycl::float2(a[0], a[1]);
}

// Convert between a vector type and a scalar type
template <typename T>
struct TypeConverter {
  using Type = sycl::half2;
}; // keep for generality

template <>
struct TypeConverter<sycl::ext::oneapi::bfloat16> {
  using Type = sycl::marray<sycl::ext::oneapi::bfloat16, 2>;
};

template <> struct TypeConverter<sycl::half> { using Type = sycl::half2; };

// general add
template<typename T>
inline T add(T a, T b) {
  return a + b;
}

template<typename T>
inline T add(T a, T b, T c) {
  return a + b + c;
}

// general sub
template<typename T>
inline T sub(T a, T b) {
  return a - b;
}

// general fma
template<typename T>
inline T fma(T a, T b, T c, T d) {
    return a * b * c + d;
}

// reduction 
template<typename T>
inline T warpReduceSum(T val, const sycl::nd_item<1> &item)
{
  auto sg = item.get_sub_group();
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = add(val, sycl::permute_group_by_xor(sg, val, mask));
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
inline T blockReduceSum(T val, const sycl::nd_item<1> &item, T *shared)
{
  int lid = item.get_local_id(0); 
  int lane = lid & 0x1f;
  int wid = lid >> 5;

  val = warpReduceSum<T>(val, item);

  if (lane == 0)
      shared[wid] = val;

  item.barrier(sycl::access::fence_space::local_space);

  val = (lid < (item.get_local_range(0) / 32)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val, item);

  return val;
}

template<typename T>
void addBiasResidualPostLayerNormV2(
          T* out,
    const T* __restrict input,
    const T* __restrict bias,
    const T* __restrict gamma,
    const T* __restrict beta,
    const float layernorm_eps,
    const int n,
    const sycl::nd_item<1> &item,
    float *shared,
    float &s_mean,
    float &s_variance)
{
  using T2             = typename TypeConverter<T>::Type;
  const int        ite = 4;

  const int lid = item.get_local_id(0);
  const int dim = item.get_local_range(0);
  const int bid = item.get_group(0);

  float            mean     = 0.0f;
  float            variance = 0.0f;
  T2               local_out_half2[ite];

  T2*       out_ptr   = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* bias_ptr  = (const T2*)bias;
  const T2* gamma_ptr = (const T2*)gamma;
  const T2* beta_ptr  = (const T2*)beta;

  T2 sum = floatToType2<T2>(0.0f);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * dim + lid;
    int id             = bid * n / 2 + col_id;
    local_out_half2[i] = add(out_ptr[id], input_ptr[id], ldg(&bias_ptr[col_id]));
    sum                = add(sum, local_out_half2[i]);
  }

  mean = blockReduceSum<float>((float)(sum[0] + sum[1]), item, shared);
  if (lid == 0) {
    s_mean = mean / n;
  }
  item.barrier(sycl::access::fence_space::local_space);

  float var      = 0.0f;
  T2    s_mean_2 = floatToType2<T2>(s_mean);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_half2[i] = sub(local_out_half2[i], s_mean_2);
    float v1 = (float)local_out_half2[i][0];
    float v2 = (float)local_out_half2[i][1];
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var, item, shared);
  if (lid == 0) {
    s_variance = sycl::rsqrt(variance / n + layernorm_eps);
  }
  item.barrier(sycl::access::fence_space::local_space);

  T2 s_var_2 = floatToType2<T2>(s_variance);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * dim + lid;
    int id      = bid * n / 2 + col_id;
    out_ptr[id] = fma(local_out_half2[i], s_var_2,
                      ldg(&gamma_ptr[col_id]), ldg(&beta_ptr[col_id]));
  }
}

template<typename T, int N>
void addBiasResidualPostLayerNorm(
          T* out,
    const T* __restrict input,
    const T* __restrict bias,
    const T* __restrict gamma,
    const T* __restrict beta,
    const float layernorm_eps,
    const int n,
    const sycl::nd_item<1> &item, float *shared, float &s_mean, float &s_variance)
{
  float            mean     = 0.0f;
  float            variance = 0.0f;
  float            local_out_cache[N];

  const int lid = item.get_local_id(0);
  const int dim = item.get_local_range(0);
  const int bid = item.get_group(0);

#pragma unroll N
  for (int idx = lid, i = 0; idx < n && i < N; ++i) {
    float local_out = (float)(add(out[bid * n + idx],
                      input[bid * n + idx], ldg(&bias[idx])));
    mean += local_out;
    // save local_out to local_out_cache to save some recompute
    local_out_cache[i] = local_out;
    idx += dim;
  }

  mean = blockReduceSum<float>(mean, item, shared);
  if (lid == 0) {
    s_mean = mean / n;
  }
  item.barrier(sycl::access::fence_space::local_space);

#pragma unroll N
  for (int idx = lid, i = 0; idx < n && i < N; ++i) {
    float local_out = local_out_cache[i];
    variance += (local_out - s_mean) * (local_out - s_mean);
    idx += dim;
  }
  variance = blockReduceSum<float>(variance, item, shared);
  if (lid == 0) {
    s_variance = variance / n + layernorm_eps;
  }
  item.barrier(sycl::access::fence_space::local_space);

#pragma unroll N
  for (int idx = lid, i = 0; idx < n && i < N; ++i) {
    float local_out = local_out_cache[i];
    out[bid * n + idx] =
        (T)(((local_out - s_mean) * sycl::rsqrt(s_variance)) *
                (float)(ldg(&gamma[idx])) + (float)(ldg(&beta[idx])));
    idx += dim;
  }
}

template<typename T>
void generalAddBiasResidualPostLayerNorm(
          T* out,
    const T* __restrict input,
    const T* __restrict bias,
    const T* __restrict gamma,
    const T* __restrict beta,
    const float layernorm_eps,
    const int n,
    const sycl::nd_item<1> &item, float *shared, float &s_mean,
    float &s_variance)
{
  using T2 = typename TypeConverter<T>::Type;

  float            mean     = 0.0f;
  float            variance = 0.0f;

  T2*       out_ptr   = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* bias_ptr  = (const T2*)bias;
  const T2* gamma_ptr = (const T2*)gamma;
  const T2* beta_ptr  = (const T2*)beta;

  const int lid = item.get_local_id(0);
  const int dim = item.get_local_range(0);
  const int bid = item.get_group(0);

  float local_out = 0.0f;
  for (int idx = lid; idx < n / 2; idx += dim) {
    int id = bid * n / 2 + idx;
    T2 tmp = add(add(out_ptr[id], input_ptr[id]), ldg(&bias_ptr[idx]));
    sycl::float2 local_out_fp2 = type2ToFloat2(tmp);
    local_out += local_out_fp2[0];
    local_out += local_out_fp2[1];
    // save tmp to out_ptr to save some recomputation
    out_ptr[id] = tmp;
  }

  mean = blockReduceSum<float>(local_out, item, shared);
  if (lid == 0) {
    s_mean = mean / n;
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int idx = lid; idx < n / 2; idx += dim) {
    int id = bid * n / 2 + idx;
    sycl::float2 local_out_fp2 = type2ToFloat2(out_ptr[id]);
    variance += (local_out_fp2[0] - s_mean) * (local_out_fp2[0] - s_mean);
    variance += (local_out_fp2[1] - s_mean) * (local_out_fp2[1] - s_mean);
  }

  variance = blockReduceSum<float>(variance, item, shared);
  if (lid == 0) {
    s_variance = sycl::rsqrt(variance / n + layernorm_eps);
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int idx = lid; idx < n / 2; idx += dim) {
    int id = bid * n / 2 + idx;
    sycl::float2 local_out_fp2 = type2ToFloat2(out_ptr[id]);
    sycl::float2 gamma_val = type2ToFloat2(ldg(&gamma_ptr[idx]));
    sycl::float2 beta_val = type2ToFloat2(ldg(&beta_ptr[idx]));
    local_out_fp2[0] =
        (local_out_fp2[0] - s_mean) * s_variance * gamma_val[0] + beta_val[0];
    local_out_fp2[1] =
        (local_out_fp2[1] - s_mean) * s_variance * gamma_val[1] + beta_val[1];
    out_ptr[id] = float2ToType2<T2>(local_out_fp2);
  }
}
