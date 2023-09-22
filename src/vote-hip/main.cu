#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "reference.h"
#include "kernels.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <warp_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int warp_size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  bool *dinfo = NULL, *hinfo = NULL;
  unsigned int *h_input, *h_result;
  unsigned int *d_input, *d_result;

  int error_count[3] = {0, 0, 0};

  h_input = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                   sizeof(unsigned int));
  h_result = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                    sizeof(unsigned int));
  genVoteTestPattern(h_input, VOTE_DATA_GROUP * warp_size);

  hipMalloc((void**)&d_input,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  hipMalloc((void**)&d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  hipMemcpy(d_input, h_input,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyHostToDevice);

  dim3 gridBlock(1);
  dim3 threadBlock(VOTE_DATA_GROUP * warp_size);

  // Start of Vote Any Test Kernel #1
  printf("\tRunning <<Vote.Any>> kernel1 ...\n");

  // Warmup
  hipLaunchKernelGGL(VoteAnyKernel1, gridBlock, threadBlock, 0, 0, d_input, d_result, repeat);
  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  hipLaunchKernelGGL(VoteAnyKernel1, gridBlock, threadBlock, 0, 0, d_input, d_result, repeat);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\tkernel execution time: %f (s)\n", time * 1e-9f);

  hipMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyDeviceToHost);

  error_count[0] += checkResultsVoteAnyKernel1(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Start of Vote All Test Kernel #2
  printf("\tRunning <<Vote.All>> kernel2 ...\n");

  // Warmup
  hipLaunchKernelGGL(VoteAllKernel2, gridBlock, threadBlock, 0, 0, d_input, d_result, repeat);
  hipDeviceSynchronize();

  start = std::chrono::steady_clock::now();

  hipLaunchKernelGGL(VoteAllKernel2, gridBlock, threadBlock, 0, 0, d_input, d_result, repeat);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\tkernel execution time: %f (s)\n", time * 1e-9f);

  hipMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyDeviceToHost);

  error_count[1] += checkResultsVoteAllKernel2(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Second Vote Kernel Test #3 (both Any/All)
  hipMalloc((void**)&dinfo, warp_size * 3 * 3 * sizeof(bool));

  // Warmup
  hipLaunchKernelGGL(VoteAnyKernel3, 1, warp_size * 3, 0, 0, dinfo, warp_size, repeat);
  hipDeviceSynchronize();

  hipMemset(dinfo, 0, warp_size * 3 * 3 * sizeof(bool));

  printf("\tRunning <<Vote.Any>> kernel3 ...\n");

  start = std::chrono::steady_clock::now();

  hipLaunchKernelGGL(VoteAnyKernel3, 1, warp_size * 3, 0, 0, dinfo, warp_size, repeat);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\tkernel execution time: %f (s)\n", time * 1e-9f);

  hinfo = (bool*) malloc (warp_size * 3 * 3 * sizeof(bool));
  hipMemcpy(hinfo, dinfo, warp_size * 3 * 3 * sizeof(bool),
             hipMemcpyDeviceToHost);

  error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

  hipFree(d_input);
  hipFree(d_result);
  hipFree(dinfo);

  free(h_input);
  free(h_result);
  free(hinfo);

  return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
