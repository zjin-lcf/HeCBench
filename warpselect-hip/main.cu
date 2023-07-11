#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "Tensor.cuh"
#include "WarpSelectKernel.cuh"

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <k> <repeat>\n", argv[0]);
    return 1;
  }
  const int rows = atoi(argv[1]);
  const int cols = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  
  size_t numInputElem = (size_t)rows * cols;
  size_t numOutputElem = (size_t)rows * k;
  size_t numInputElemBytes = sizeof(float) * numInputElem;
  size_t numOutputElemBytes = sizeof(float) * numOutputElem;
  
  std::default_random_engine g (19937);
  std::uniform_real_distribution<float> uniform_distr (0.f, 1.f);

  std::vector<float> h_in(numInputElem);
  for (size_t i = 0; i < numInputElem; i++) h_in[i] = uniform_distr(g);

  std::vector<float> h_k(numOutputElem);
  std::vector<int> h_v(numOutputElem);
  
  float *d_in, *d_k;
  int *d_v;
  hipMalloc((void**)&d_in, numInputElemBytes); 
  hipMalloc((void**)&d_k, numOutputElemBytes); 
  hipMalloc((void**)&d_v, numOutputElemBytes); 

  faiss::gpu::Tensor<float, 2, true> input (d_in, {rows, cols});  // input value
  faiss::gpu::Tensor<float, 2, true> output (d_k, {rows, k}); // output value
  faiss::gpu::Tensor<int, 2, true> index (d_v, {rows, k}); // output index
  input.copyFrom(h_in, 0);
  output.copyFrom(h_k, 0);
  index.copyFrom(h_v, 0);
  
  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    faiss::gpu::runWarpSelect(input, output, index, true, k, 0);
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of runWarpSelect: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    faiss::gpu::runWarpSelect(input, output, index, false, k, 0);
  }
  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of runWarpSelect: %f (us)\n", (time * 1e-3f) / repeat);

  h_k = output.copyToVector(0);
  h_v = index.copyToVector(0);

  double s1  = 0.0, s2 = 0.0;
  for (int i = 0; i < rows; i++) {
    s1 += h_k[i * k];
    s2 += h_in[h_v[i * k]];
  }
  printf("checksum: %lf %lf\n", s1, s2);
  
  hipFree(d_in);
  hipFree(d_k);
  hipFree(d_v);
  return 0;
}
