
#include <iostream>
#include <limits>
#include <cmath>
#include <cuda.h>

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

#ifdef VERIFY
/**
 * Perform matrix multiplication on host to verify results from device.
 */
bool ValueSame(float a, float b) {
  return fabsf(a - b) < numeric_limits<float>::epsilon();
}

void VerifyResult(float (*a_host)[N], float (*b_host)[P], 
                  float (*c_host)[P], float (*c_back)[P]) {
  // Check that the results are correct by comparing with host computing.
  int i, j, k;

  // c_host is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = 0.0f;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      for (j = 0; j < P; j++) {
        c_host[i][j] += sqrtf(a_host[i][k] * b_host[k][j]);
      }
    }
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      float value = (1 - c_host[i][j]);
      float gate = (!signbit(value));
      c_host[i][j] = sqrtf(gate * value);
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  if (!mismatch_found) {
    cout << "PASS\n";
  } else {
    cout << "FAIL\n";
  }
}
#endif

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

  cudaMalloc((void **) &a_device, sizeof(float)*M*N);
  cudaMalloc((void **) &b_device, sizeof(float)*N*P);
  cudaMalloc((void **) &c_device, sizeof(float)*M*P);

  cudaMemcpy(a_device, a_host, sizeof(float)*M*N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, b_host, sizeof(float)*N*P, cudaMemcpyHostToDevice);

  unsigned int grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  for (int i = 0; i < 100; i++)
    hellinger<<<dimGrid, dimBlock>>>(a_device, b_device, c_device, M, N, P);

  cudaMemcpy(c_back, c_device, sizeof(int)*M*P, cudaMemcpyDeviceToHost);

  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
       << ") * b(" << N << "," << P << ")\n";

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);
  return 0;
}

