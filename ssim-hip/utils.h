//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#include <hip/hip_runtime.h>

#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

#ifdef HIP_CHECK
#undef HIP_CHECK
#endif 

#define HIP_CHECK(call)                                                                       \
  do {                                                                                         \
    hipError_t error = call;                                                                  \
    if (error != hipSuccess) {                                                                \
      const char* msg = hipGetErrorString(error);                                             \
      fprintf(stderr, "HIP error (%s: line %d): %s\n", __FILE__, __LINE__, msg);              \
      throw std::runtime_error(std::string("HIP error: " #call " failed with error ") + msg); \
    }                                                                                          \
  } while (0)


#ifdef __HIPCC__
#define HIP_UTIL_BOTH_INLINE __forceinline__ __device__ __host__
#else
#define HIP_UTIL_BOTH_INLINE 
#endif

namespace util {

  template <typename T>
    HIP_UTIL_BOTH_INLINE T div_round_up(T val, T divisor) {
      return (val + divisor - 1) / divisor;
    }

  template <typename T>
    HIP_UTIL_BOTH_INLINE T next_multiple(T val, T divisor) {
      return div_round_up(val, divisor) * divisor;
    }

  constexpr uint32_t n_threads_trilinear = 8;

#ifdef __HIPCC__

  //. ======================================================================== //
  // trilinear version 
  //. ======================================================================== //

  template<typename T>
    constexpr uint32_t
    n_blocks_trilinear(T n_elements)
    {
      return ((uint32_t)n_elements + n_threads_trilinear - 1) / n_threads_trilinear;
    }

  template<typename K, typename T, typename... Types>
    inline void
    trilinear_kernel(K kernel, uint32_t shmem_size, hipStream_t stream, T width, T height, T depth, Types... args)
    {
      if (width <= 0 || height <= 0 || depth <= 0) {
        return;
      }
      dim3 block_size(n_threads_trilinear, n_threads_trilinear, n_threads_trilinear);
      dim3 grid_size(n_blocks_trilinear(width), n_blocks_trilinear(height), n_blocks_trilinear(depth));
      kernel<<<grid_size, block_size, shmem_size, stream>>>((uint32_t)width, (uint32_t)height, (uint32_t)depth, args...);
    }

  template<typename K, typename... Types>
    inline void
    trilinear_kernel(K kernel, uint32_t shmem_size, hipStream_t stream, int3 dims, Types... args)
    {
      if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        return;
      }
      dim3 block_size(n_threads_trilinear, n_threads_trilinear, n_threads_trilinear);
      dim3 grid_size(n_blocks_trilinear(dims.x), n_blocks_trilinear(dims.y), n_blocks_trilinear(dims.z));
      kernel<<<grid_size, block_size, shmem_size, stream>>>(dims, args...);
    }

#endif
}
