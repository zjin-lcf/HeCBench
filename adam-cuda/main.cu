#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <cuda.h>
#include "reference.h"

template <typename T, typename G>
__global__
void adam (
        T* __restrict__ p,
        T* __restrict__ m,
        T* __restrict__ v,
  const G* __restrict__ g,
  const float b1,
  const float b2,
  const float eps,
  const float grad_scale,
  const float step_size,
  const size_t tsize,
  adamMode_t mode,
  const float decay)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int totThreads = gridDim.x*blockDim.x;

  for (size_t j = i; j < tsize; j += totThreads) {
    T scaled_grad = g[j]/grad_scale;
    m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
    v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
    float denom;
    if (mode == ADAM_MODE_0)
      denom = sqrtf(v[j] + eps);
    else // Mode 1
      denom = sqrtf(v[j]) + eps;
    float update = (m[j]/denom) + (decay*p[j]);
    p[j] -= (step_size*update);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <size> <repeat>\n", argv[0]);
    return 1;
  }

  const int tsize = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t size_bytes = tsize * sizeof(float);

  float *m = (float*) malloc (size_bytes);
  float *v = (float*) malloc (size_bytes);
  float *g = (float*) malloc (size_bytes);
  float *p = (float*) malloc (size_bytes);
  float *r = (float*) malloc (size_bytes);

  srand(123);
  for (int i = 0; i < tsize; i++) {
    m[i] = rand() / (float)RAND_MAX;
    v[i] = rand() / (float)RAND_MAX;
    g[i] = rand() / (float)RAND_MAX;
    r[i] = p[i] = rand() / (float)RAND_MAX;
  }

  float *d_m, *d_v, *d_g, *d_p;

  cudaMalloc((void**)&d_m, size_bytes);
  cudaMemcpy(d_m, m, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_v, size_bytes);
  cudaMemcpy(d_v, v, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_g, size_bytes);
  cudaMemcpy(d_g, g, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_p, size_bytes);
  cudaMemcpy(d_p, p, size_bytes, cudaMemcpyHostToDevice);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-10f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  const dim3 grids ((tsize+threadsPerBlock-1) / threadsPerBlock);
  const dim3 blocks (threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    adam<float, float><<<grids, blocks>>> (
      d_p, d_m, d_v, d_g,
      beta1, beta2,
      eps,
      grad_scale,
      step_size,
      tsize,
      mode,
      decay);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  cudaMemcpy(p, d_p, size_bytes, cudaMemcpyDeviceToHost); 

  cudaFree(d_p);
  cudaFree(d_m);
  cudaFree(d_v);
  cudaFree(d_g);

  // verify
  reference<float, float>(
    repeat,
    r, m, v, g,
    beta1, beta2,
    eps,
    grad_scale,
    step_size,
    tsize,
    mode,
    decay);

  bool ok = true; 
  for (int i = 0; i < tsize; i++) {
    if (r[i] - p[i] > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}
