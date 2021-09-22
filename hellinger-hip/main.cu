#include <iostream>
#include <limits>
#include <cmath>
#include <hip/hip_runtime.h>

using namespace std;

#define BLOCK_SIZE 16

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 768 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#include "verify.cpp"

__global__ 
void hellinger(
  const float *__restrict a, 
  const float *__restrict b, 
        float *__restrict c, 
  const int m, const int n, const int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if( col < k && row < m)
    {
        float sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += sqrtf(a[row * n + i] * b[i * k + col]);
        }
        const float value = 1.f - sum;
        const float gate = (!signbit(value));
        c[row * k + col] = sqrtf(gate * value);
    }
}

int main() {
  int i, j;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[P] = new float[N][P];
  // host-side cpu result
  float(*c_host)[P] = new float[M][P];
  // host-side gpu result
  float(*c_back)[P] = new float[M][P];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    float sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  float *a_device, *b_device, *c_device;

  hipMalloc((void **) &a_device, sizeof(float)*M*N);
  hipMalloc((void **) &b_device, sizeof(float)*N*P);
  hipMalloc((void **) &c_device, sizeof(float)*M*P);

  hipMemcpy(a_device, a_host, sizeof(float)*M*N, hipMemcpyHostToDevice);
  hipMemcpy(b_device, b_host, sizeof(float)*N*P, hipMemcpyHostToDevice);

  unsigned int grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  for (int i = 0; i < 100; i++)
    hipLaunchKernelGGL(hellinger, dimGrid, dimBlock, 0, 0, a_device, b_device, c_device, M, N, P);

  hipMemcpy(c_back, c_device, sizeof(int)*M*P, hipMemcpyDeviceToHost);

  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
       << ") * b(" << N << "," << P << ")\n";

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  hipFree(a_device);
  hipFree(b_device);
  hipFree(c_device);
  return 0;
}

