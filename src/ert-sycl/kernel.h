#include <inttypes.h>
#include <type_traits>
#include <typeinfo>
#include "rep.h"

int gpu_blocks;
int gpu_threads;


#define KERNEL1(a, b, c) ((a) = (b) + (c))
#define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))
#define KERNEL2HALF(a, b, c) ((a) = ((b) + (c)))
#define KERNEL4HALF(a, b, c) ((a) = (a) * (b) + (c))

// forward declaration
template <typename T>
class blockStride;

// If data type is "sycl::half2"
template <typename T, typename std::enable_if<std::is_same<T, sycl::half2>::value, int>::type = 0>
void initialize(uint64_t nsize, T *__restrict A, float value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = sycl::half2(value);
  }
}

// If data type is float or double
template <typename T, typename std::enable_if<!std::is_same<T, sycl::half2>::value, int>::type = 0>
void initialize(uint64_t nsize, T *__restrict A, float value)
{
  uint64_t i;
  for (i = 0; i < nsize; ++i) {
    A[i] = (T)value;
  }
}

// If data type is "sycl::half2"
template <typename T, typename std::enable_if<std::is_same<T, sycl::half2>::value, int>::type = 0>
void block_stride(uint32_t ntrials, uint32_t nsize, T *__restrict A, sycl::nd_item<1> &item)
{
  uint32_t blockIdx_x   = item.get_group(0);
  uint32_t gridDim_x    = item.get_group_range(0);
  uint32_t blockDim_x   = item.get_local_range(0);
  uint32_t threadIdx_x  = item.get_local_id(0);
  uint32_t total_thr    = gridDim_x * blockDim_x;
  uint32_t elem_per_thr = (nsize + (total_thr - 1)) / total_thr;
  uint32_t blockOffset  = blockIdx_x * blockDim_x;

  uint32_t start_idx  = blockOffset + threadIdx_x;
  uint32_t end_idx    = start_idx + elem_per_thr * total_thr;
  uint32_t stride_idx = total_thr;

  if (start_idx > nsize) {
    start_idx = nsize;
  }

  if (end_idx > nsize) {
    end_idx = nsize;
  }

  // A needs to be initilized to -1 coming in
  // And with alpha=2 and beta=1, A=-1 is preserved upon return
  T alpha      = sycl::half2(2.0f);
  T const_beta = sycl::half2(1.0f);

  uint32_t i, j;
  for (j = 0; j < ntrials; ++j) {
    for (i = start_idx; i < end_idx; i += stride_idx) {
      T beta = const_beta;
      /* add 2+4+8+16+32+64+128+256+512+1024 flops */
      KERNEL2HALF(beta, A[i], alpha);
      KERNEL4HALF(beta, A[i], alpha);
      REP2(KERNEL4HALF(beta, A[i], alpha));
      REP4(KERNEL4HALF(beta, A[i], alpha));
      REP8(KERNEL4HALF(beta, A[i], alpha));
      REP16(KERNEL4HALF(beta, A[i], alpha));
      REP32(KERNEL4HALF(beta, A[i], alpha));
      REP64(KERNEL4HALF(beta, A[i], alpha));
      REP128(KERNEL4HALF(beta, A[i], alpha));
      REP256(KERNEL4HALF(beta, A[i], alpha));
      A[i] = -beta;

    }
  }
}

// If data type is float or double
template <typename T, typename std::enable_if<!std::is_same<T, sycl::half2>::value, int>::type = 0>
void block_stride(uint32_t ntrials, uint32_t nsize, T *__restrict A, sycl::nd_item<1> &item)
{
  uint32_t blockIdx_x   = item.get_group(0);
  uint32_t gridDim_x    = item.get_group_range(0);
  uint32_t blockDim_x   = item.get_local_range(0);
  uint32_t threadIdx_x  = item.get_local_id(0);
  uint32_t total_thr    = gridDim_x * blockDim_x;
  uint32_t elem_per_thr = (nsize + (total_thr - 1)) / total_thr;

  uint32_t start_idx  = blockIdx_x * blockDim_x + threadIdx_x;
  uint32_t end_idx    = start_idx + elem_per_thr * total_thr;
  uint32_t stride_idx = total_thr;

  if (start_idx > nsize) {
    start_idx = nsize;
  }

  if (end_idx > nsize) {
    end_idx = nsize;
  }

  // A needs to be initilized to -1 coming in
  // And with alpha=2 and beta=1, A=-1 is preserved upon return
  T alpha, const_beta;
  alpha      = 2.0;
  const_beta = 1.0;

  uint32_t i, j;
  for (j = 0; j < ntrials; ++j) {
    for (i = start_idx; i < end_idx; i += stride_idx) {
      T beta = const_beta;
      /* add 1+2+4+8+16+32+64+128+256+512+1024 flops */
      KERNEL1(beta, A[i], alpha);
      KERNEL2(beta, A[i], alpha);
      REP2(KERNEL2(beta, A[i], alpha));
      REP4(KERNEL2(beta, A[i], alpha));
      REP8(KERNEL2(beta, A[i], alpha));
      REP16(KERNEL2(beta, A[i], alpha));
      REP32(KERNEL2(beta, A[i], alpha));
      REP64(KERNEL2(beta, A[i], alpha));
      REP128(KERNEL2(beta, A[i], alpha));
      REP256(KERNEL2(beta, A[i], alpha));
      REP512(KERNEL2(beta, A[i], alpha));
      A[i] = -beta;
    }
  }
}

template <typename T>
void gpuKernel(sycl::queue &q, uint32_t nsize, uint32_t ntrials, T *A,
               int *bytes_per_elem, int *mem_accesses_per_elem)
{
  *bytes_per_elem        = sizeof(T);
  *mem_accesses_per_elem = 2;

  sycl::range<1> gws (gpu_blocks * gpu_threads);
  sycl::range<1> lws (gpu_threads);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class blockStride<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      block_stride<T>(ntrials, nsize, A, item);
    });
  });
}
