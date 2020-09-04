#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "rand_helpers.h"
#include "constants.h"
#include "wallace_kernel.cpp"

int main() 
{
  // host buffers
  float *hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
  {
  	float x = RandN();
  	hostPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
  	rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

  // device buffers
  float *device_randomNumbers;
  float *devicerngChi2Corrections;
  float *devPool;
 dpct::dpct_malloc((void **)&devPool, 4 * WALLACE_TOTAL_POOL_SIZE);
 dpct::dpct_malloc((void **)&devicerngChi2Corrections, 4 * WALLACE_CHI2_COUNT);
 dpct::dpct_malloc((void **)&device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE);

 dpct::dpct_memcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE,
                   dpct::host_to_device);
 dpct::dpct_memcpy(devicerngChi2Corrections, rngChi2Corrections,
                   4 * WALLACE_CHI2_COUNT, dpct::host_to_device);

 sycl::range<3> rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
 sycl::range<3> rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);
  const unsigned seed = 1;
  
  for (int i = 0; i < 100; i++) {
  {
   dpct::buffer_t devPool_buf_ct1 = dpct::get_buffer(devPool);
   dpct::buffer_t device_randomNumbers_buf_ct2 =
       dpct::get_buffer(device_randomNumbers);
   dpct::buffer_t devicerngChi2Corrections_buf_ct3 =
       dpct::get_buffer(devicerngChi2Corrections);
   dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        pool_acc_ct1(sycl::range<1>(
                         2049 /*WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE*/),
                     cgh);
    auto devPool_acc_ct1 =
        devPool_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
    auto device_randomNumbers_acc_ct2 =
        device_randomNumbers_buf_ct2.get_access<sycl::access::mode::read_write>(
            cgh);
    auto devicerngChi2Corrections_acc_ct3 =
        devicerngChi2Corrections_buf_ct3
            .get_access<sycl::access::mode::read_write>(cgh);

    auto dpct_global_range = rng_wallace_grid * rng_wallace_threads;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                         dpct_global_range.get(1),
                                         dpct_global_range.get(0)),
                          sycl::range<3>(rng_wallace_threads.get(2),
                                         rng_wallace_threads.get(1),
                                         rng_wallace_threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
         rng_wallace(seed, (float *)(&devPool_acc_ct1[0]),
                     (float *)(&device_randomNumbers_acc_ct2[0]),
                     (float *)(&devicerngChi2Corrections_acc_ct3[0]), item_ct1,
                     pool_acc_ct1.get_pointer());
        });
   });
  }
  dpct::dpct_memcpy(randomNumbers, device_randomNumbers,
                    4 * WALLACE_OUTPUT_SIZE, dpct::device_to_host);
#ifdef DEBUG
    // random numbers are different for each i iteration 
    for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) 
    	printf("%.3f\n", randomNumbers[n]);
#endif
  }
  
  free(rngChi2Corrections);
  free(randomNumbers);
  free(hostPool);
 dpct::dpct_free(devicerngChi2Corrections);
 dpct::dpct_free(device_randomNumbers);
 dpct::dpct_free(devPool);
  return 0;
}
