#include <iostream>
#include <limits>
#include <cmath>
#include "common.h"

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

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
  {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    sycl::queue q(dev_sel);

    // Create buffers for matrices, buffer c is bound with host memory c_back
    buffer<float, 1> a(reinterpret_cast<float*>(a_host), range<1>(M*N));
    buffer<float, 1> b(reinterpret_cast<float*>(b_host), range<1>(N*P));
    buffer<float, 1> c(reinterpret_cast<float*>(c_back), range<1>(M*P));

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    auto grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    auto grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    range<2> gws (grid_rows, grid_cols);
    range<2> lws (BLOCK_SIZE, BLOCK_SIZE);

    for (int i = 0; i < 100; i++) {
      q.submit([&](handler &h) {
        auto A = a.get_access<sycl_read>(h);
        auto B = b.get_access<sycl_read>(h);
        auto C = c.get_access<sycl_discard_write>(h);
        h.parallel_for<class hellinger>(nd_range<2>(gws, lws), [=](nd_item<2> index) {
          int row = index.get_global_id(0);
          int col = index.get_global_id(1);
          if( col < P && row < M) {
            float sum = 0;
            for (int i = 0; i < N; i++) {
              sum += sycl::sqrt(A[row * N + i] * B[i * P + col]);
            }
            const float value = 1.f - sum;
            const float gate = (!sycl::signbit(value));
            C[row * P + col] = sycl::sqrt(gate * value);
          }
        });
      });
    }
  }

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return 0;
}

