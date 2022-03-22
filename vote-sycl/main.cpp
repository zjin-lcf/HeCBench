#include <assert.h>
#include <stdio.h>
#include <CL/sycl.hpp>
#include "reference.h"
#include "kernels.h"

// WARP_SIZE is defined in Makefile

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <warp_size> <repeat>\n", argv[0]);
    return 1;
  }

  const int warp_size = atoi(argv[1]);
  if (warp_size != WARP_SIZE) {
    printf("The warp size should be %d\n", WARP_SIZE);
    return 1;
  }
  const int repeat = atoi(argv[2]);

  bool *dinfo = NULL, *hinfo = NULL;
  int error_count[3] = {0, 0, 0};

  unsigned int *h_input, *h_result;
  unsigned int *d_input, *d_result;

  h_input = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                   sizeof(unsigned int));
  h_result = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                    sizeof(unsigned int));
  genVoteTestPattern(h_input, VOTE_DATA_GROUP * warp_size);

#ifdef USE_GPU
  sycl::gpu_selector dev_sel;
#else
  sycl::cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  d_input = sycl::malloc_device<unsigned int>(
      VOTE_DATA_GROUP * warp_size, q);

  d_result = sycl::malloc_device<unsigned int>(
      VOTE_DATA_GROUP * warp_size, q);

  q.memcpy(d_input, h_input,
           VOTE_DATA_GROUP * warp_size * sizeof(unsigned int)).wait();

  // Start of Vote Any Test Kernel #1
  printf("[VOTE Kernel Test 1/3]\n");
  printf("\tRunning <<Vote.Any>> kernel1 ...\n");
  sycl::range<1> gws (VOTE_DATA_GROUP * warp_size);
  sycl::range<1> lws (VOTE_DATA_GROUP * warp_size);

  // reqd_sub_group_size requires a constant
  q.parallel_for<class any_test>(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) 
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        VoteAnyKernel1(d_input, d_result, repeat, item);
  }).wait();

  q.memcpy(h_result, d_result,
           VOTE_DATA_GROUP * warp_size * sizeof(unsigned int)).wait();

  error_count[0] += checkResultsVoteAnyKernel1(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Start of Vote All Test Kernel #2
  printf("\tRunning <<Vote.All>> kernel2 ...\n");
  q.parallel_for<class all_test>(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        VoteAllKernel2(d_input, d_result, repeat, item);
  }).wait();

  q.memcpy(h_result, d_result,
           VOTE_DATA_GROUP * warp_size * sizeof(unsigned int)).wait();

  error_count[1] += checkResultsVoteAllKernel2(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Second Vote Kernel Test #3 (both Any/All)
  hinfo = reinterpret_cast<bool *>(calloc(warp_size * 3 * 3, sizeof(bool)));
  *(reinterpret_cast<void **>(&dinfo)) =
      (void *)sycl::malloc_device(warp_size * 3 * 3 * sizeof(bool), q);
  q.memcpy(dinfo, hinfo, warp_size * 3 * 3 * sizeof(bool)).wait();

  printf("\tRunning <<Vote.Any>> kernel3 ...\n");
  q.parallel_for<class vote_any> (sycl::nd_range<1>(sycl::range<1>(warp_size * 3),
                                   sycl::range<1>(warp_size * 3)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        VoteAnyKernel3(dinfo, warp_size, repeat, item);
  }).wait();

  q.memcpy(hinfo, dinfo, warp_size * 3 * 3 * sizeof(bool)).wait();

  error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

  sycl::free(d_input, q);
  sycl::free(d_result, q);
  sycl::free(dinfo, q);

  free(h_input);
  free(h_result);
  free(hinfo);

  return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
