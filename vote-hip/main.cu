#include <assert.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include "reference.h"
#include "kernels.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <warp_size> <repeat>\n", argv[0]);
    return 1;
  }

  bool *dinfo = NULL, *hinfo = NULL;
  int error_count[3] = {0, 0, 0};

  unsigned int *h_input, *h_result;
  unsigned int *d_input, *d_result;

  const int warp_size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  h_input = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                   sizeof(unsigned int));
  h_result = (unsigned int *)malloc(VOTE_DATA_GROUP * warp_size *
                                    sizeof(unsigned int));
  genVoteTestPattern(h_input, VOTE_DATA_GROUP * warp_size);

  hipMalloc(reinterpret_cast<void **>(&d_input),
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  hipMalloc(reinterpret_cast<void **>(&d_result),
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int));
  hipMemcpy(d_input, h_input,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyHostToDevice);

  // Start of Vote Any Test Kernel #1
  printf("[VOTE Kernel Test 1/3]\n");
  printf("\tRunning <<Vote.Any>> kernel1 ...\n");
  dim3 gridBlock(1);
  dim3 threadBlock(VOTE_DATA_GROUP * warp_size);
  hipLaunchKernelGGL(VoteAnyKernel1, gridBlock, threadBlock, 0, 0, d_input, d_result, repeat);
  hipMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyDeviceToHost);

  error_count[0] += checkResultsVoteAnyKernel1(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Start of Vote All Test Kernel #2
  printf("\tRunning <<Vote.All>> kernel2 ...\n");
  hipLaunchKernelGGL(VoteAllKernel2, gridBlock, threadBlock, 0, 0, d_input, d_result, repeat);
  hipMemcpy(h_result, d_result,
             VOTE_DATA_GROUP * warp_size * sizeof(unsigned int),
             hipMemcpyDeviceToHost);

  error_count[1] += checkResultsVoteAllKernel2(
      h_result, VOTE_DATA_GROUP * warp_size, warp_size);

  // Second Vote Kernel Test #3 (both Any/All)
  hinfo = reinterpret_cast<bool *>(calloc(warp_size * 3 * 3, sizeof(bool)));
  hipMalloc(reinterpret_cast<void **>(&dinfo),
             warp_size * 3 * 3 * sizeof(bool));
  hipMemcpy(dinfo, hinfo, warp_size * 3 * 3 * sizeof(bool),
             hipMemcpyHostToDevice);

  printf("\tRunning <<Vote.Any>> kernel3 ...\n");
  hipLaunchKernelGGL(VoteAnyKernel3, 1, warp_size * 3, 0, 0, dinfo, warp_size, repeat);

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
