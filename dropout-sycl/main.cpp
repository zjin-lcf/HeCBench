#include <cstdio>
#include <chrono>
#include <utility> // std::pair
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include "common.h"

// philox generates 128 bits of randomness at a time. 
// Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
const int UNROLL = 4;

template <typename scalar_t,
          typename accscalar_t,
          typename IndexType>
void
fused_dropout_kernel(
  const scalar_t *__restrict a,
        scalar_t *__restrict b,
         uint8_t *__restrict c,
  IndexType totalElements,
  accscalar_t p,
  std::pair<uint64_t, uint64_t> seeds,
  sycl::nd_item<1> &item) 
{
  int blockDim = item.get_local_range(0);
  int gridDim = item.get_group_range(0);

  accscalar_t pinv = accscalar_t(1)/p;
  IndexType idx = item.get_global_id(0);

  oneapi::mkl::rng::device::philox4x32x10<4> engine (
      seeds.first, {seeds.second, idx * 4});

  IndexType rounded_size = ((totalElements - 1) / (blockDim * gridDim * UNROLL) + 1) *
                           blockDim * gridDim * UNROLL;

  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim * blockDim * UNROLL) {

    oneapi::mkl::rng::device::uniform<float> distr;
    sycl::float4 rand = oneapi::mkl::rng::device::generate(distr, engine);

    scalar_t src[UNROLL];
    rand.x() = rand.x() < p;
    rand.y() = rand.y() < p;
    rand.z() = rand.z() < p;
    rand.w() = rand.w() < p;
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim * gridDim * ii;
      if (li < totalElements) {
        const IndexType aOffset = li;
        src[ii] = a[aOffset];
      }
    }

    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim * gridDim * ii;
      if (li < totalElements) {
        const IndexType bOffset = li;
        b[bOffset] = src[ii] * (&rand.x())[ii] * pinv;
        c[bOffset] = (uint8_t)(&rand.x())[ii];
      }
    }
    item.barrier(access::fence_space::local_space);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nelem = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const int64_t block_size = 256;

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  rng_engine_inputs.first =  12345678;
  rng_engine_inputs.second = 87654321;

  int64_t self_size = nelem * sizeof(float);
  int64_t ret_size = self_size;
  int64_t mask_size = nelem * sizeof(uint8_t);

  float *self_info = (float*) malloc (self_size); 
  float *ret_info = (float*) malloc (ret_size); 
  uint8_t *mask_info = (uint8_t*) malloc (mask_size);

  for (int64_t i = 0; i < nelem; i++) {
    self_info[i] = 0.1f;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  float *d_self_info = (float *)sycl::malloc_device(self_size, q);
  q.memcpy(d_self_info, self_info, self_size);

  float *d_ret_info = (float *)sycl::malloc_device(ret_size, q);

  uint8_t *d_mask_info = (uint8_t *)sycl::malloc_device(mask_size, q);

  sycl::range<1> lws (block_size);
  sycl::range<1> gws (256 * block_size);

  double total_time = 0.0;

  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        fused_dropout_kernel<float, float, unsigned int>(
          d_self_info, d_ret_info, d_mask_info, nelem, pa, rng_engine_inputs, item);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

#ifdef DEBUG
    q.memcpy(ret_info, d_ret_info, ret_size);
    q.memcpy(mask_info, d_mask_info, mask_size);
    q.wait();

    double ret_sum = 0.0;
    int64_t mask_sum = 0;
    for (int64_t i = 0; i < nelem; i++) {
      ret_sum += ret_info[i];
      mask_sum += mask_info[i];
    }
    printf("p=%2d ret_sum=%lf mask_sum=%ld\n", p, ret_sum, mask_sum);
#endif
  }

  printf("Total kernel execution time %lf (s)\n", total_time * 1e-9f);

  sycl::free(d_self_info, q);
  sycl::free(d_ret_info, q);
  sycl::free(d_mask_info, q);
  free(self_info); 
  free(ret_info); 
  free(mask_info); 

  return 0;
}
