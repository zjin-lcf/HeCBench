#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>

__device__ __forceinline__
float michalewicz(const float *xValues, const int dim) {
  float result = 0;
  for (int i = 0; i < dim; ++i) {
      float a = sinf(xValues[i]);
      float b = sinf(((i + 1) * xValues[i] * xValues[i]) / (float)M_PI);
      float c = powf(b, 20); // m = 10
      result += a * c;
  }
  return -1.0f * result;
}

__global__ void eval (const float *values, float *minima,
                      const size_t nVectors, const int dim)
{
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < nVectors) {
    atomicMin(minima, michalewicz(values + n * dim, dim));
  }
}

// https://www.sfu.ca/~ssurjano/michal.html
void Error(float value, int dim) {
  printf("Global minima = %f\n", value);
  float trueMin = 0.0;
  if (dim == 2)
    trueMin = -1.8013;
  else if (dim == 5)
    trueMin = -4.687658;
  else if (dim == 10)
    trueMin = -9.66015;
  printf("Error = %f\n", fabsf(trueMin - value));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  // generate random numbers
  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0, 4.0);
  
  // dimensions
  const int dims[] = {2, 5, 10}; 

  for (int d = 0; d < 3; d++) {

    const int dim = dims[d];

    const size_t size = n * dim;

    const size_t size_bytes = size * sizeof(float);
    
    float *values = (float*) malloc (size_bytes);
    
    for (size_t i = 0; i < size; i++) {
      values[i] = dis(gen);
    }
    
    float *d_values;
    hipMalloc((void**)&d_values, size_bytes);
    hipMemcpy(d_values, values, size_bytes, hipMemcpyHostToDevice);

    float *d_minValue;
    float minValue;
    hipMalloc((void**)&d_minValue, sizeof(float));

    dim3 grids ((n + 255) / 256);
    dim3 blocks (256);

    hipMemset(d_minValue, 0, sizeof(float));
    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      eval<<<grids, blocks>>>(d_values, d_minValue, n, dim);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    hipMemcpy(&minValue, d_minValue, sizeof(float), hipMemcpyDeviceToHost);
    Error(minValue, dim);

    hipFree(d_values);
    hipFree(d_minValue);
    free(values);
  }

  return 0;
}
