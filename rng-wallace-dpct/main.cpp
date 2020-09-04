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
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
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
 devPool = (float *)sycl::malloc_device(4 * WALLACE_TOTAL_POOL_SIZE, q_ct1);
 devicerngChi2Corrections =
     (float *)sycl::malloc_device(4 * WALLACE_CHI2_COUNT, q_ct1);
 device_randomNumbers =
     (float *)sycl::malloc_device(4 * WALLACE_OUTPUT_SIZE, q_ct1);

 q_ct1.memcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE).wait();
 q_ct1
     .memcpy(devicerngChi2Corrections, rngChi2Corrections,
             4 * WALLACE_CHI2_COUNT)
     .wait();

 sycl::range<3> rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
 sycl::range<3> rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);
  const unsigned seed = 1;
  
  for (int i = 0; i < 1000; i++) {
  q_ct1.submit([&](sycl::handler &cgh) {
   sycl::accessor<float, 1, sycl::access::mode::read_write,
                  sycl::access::target::local>
       pool_acc_ct1(sycl::range<1>(
                        2049 /*WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE*/),
                    cgh);

   auto dpct_global_range = rng_wallace_grid * rng_wallace_threads;

   cgh.parallel_for(
       sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                        dpct_global_range.get(1),
                                        dpct_global_range.get(0)),
                         sycl::range<3>(rng_wallace_threads.get(2),
                                        rng_wallace_threads.get(1),
                                        rng_wallace_threads.get(0))),
       [=](sycl::nd_item<3> item_ct1) {
        rng_wallace(seed, devPool, device_randomNumbers,
                    devicerngChi2Corrections, item_ct1,
                    pool_acc_ct1.get_pointer());
       });
  });
  q_ct1.memcpy(randomNumbers, device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE)
      .wait();
#ifdef DEBUG
    // random numbers are different for each i iteration 
    for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) 
    	printf("%.3f\n", randomNumbers[n]);
#endif
  }
  
  free(rngChi2Corrections);
  free(randomNumbers);
  free(hostPool);
 sycl::free(devicerngChi2Corrections, q_ct1);
 sycl::free(device_randomNumbers, q_ct1);
 sycl::free(devPool, q_ct1);
  return 0;
}
