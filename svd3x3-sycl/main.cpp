#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include "common.h"

#include "kernels.cpp"

void runDevice(float* input, float* output, int n)
{

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_answer (output, 21*n);
  buffer<float, 1> d_input (input, 9*n);

  range<1> lws (256);
  range<1> gws ((n+255)/256*256);

  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      auto output = d_answer.get_access<sycl_discard_write>(cgh);
      auto input = d_input.get_access<sycl_read>(cgh);
      cgh.parallel_for<class svd3x3_soa>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        svd3_SOA(input.get_pointer(), output.get_pointer(), n, item);
      });
    });
  }
}

void svd3x3_ref(float* input, float* output, int testsize)
{
  for (int tid = 0; tid < testsize; tid++) 
    svd(
      input[tid + 0 * testsize], input[tid + 1 * testsize], input[tid + 2 * testsize],
      input[tid + 3 * testsize], input[tid + 4 * testsize], input[tid + 5 * testsize],
      input[tid + 6 * testsize], input[tid + 7 * testsize], input[tid + 8 * testsize],

      output[tid + 0 * testsize], output[tid + 1 * testsize], output[tid + 2 * testsize],
      output[tid + 3 * testsize], output[tid + 4 * testsize], output[tid + 5 * testsize],
      output[tid + 6 * testsize], output[tid + 7 * testsize], output[tid + 8 * testsize],
      output[tid + 9 * testsize], output[tid + 10 * testsize], output[tid + 11 * testsize],
      output[tid + 12 * testsize], output[tid + 13 * testsize], output[tid + 14 * testsize],
      output[tid + 15 * testsize], output[tid + 16 * testsize], output[tid + 17 * testsize],
      output[tid + 18 * testsize], output[tid + 19 * testsize], output[tid + 20 * testsize]
    );
}

int main(int argc, char* argv[])
{
  // Load data
  const char* filename = argv[1];
  std::ifstream myfile;
  myfile.open(filename);
  int testsSize;
  myfile >> testsSize;
  std::cout << "dataset size: " << testsSize << std::endl;
  if (testsSize <= 0) {
    std::cout << "ERROR: invalid dataset size\n";
    return -1;
  }

  float* input = (float*)malloc(sizeof(float) * 9 * testsSize);

  // host and device results
  float* result = (float*)malloc(sizeof(float) * 21 * testsSize);
  float* result_h = (float*)malloc(sizeof(float) * 21 * testsSize);

  int count = 0;
  for (int i = 0; i < testsSize; i++)
    for (int j = 0; j < 9; j++) myfile >> input[count++];
  myfile.close();

  // SVD 3x3 on a GPU 
  runDevice(input, result, testsSize);

  bool ok = true;
  svd3x3_ref(input, result_h, testsSize);

  for (int i = 0; i < testsSize; i++)
  {
    if (fabsf(result[i] - result_h[i]) > 1e-3) {
      std::cout << result[i] << " " << result_h[i] << std::endl;
      ok = false;
      break;
    }
  }

  if (ok)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";

  free(input);
  free(result);
  free(result_h);
  return 0;
}
