#include <iostream>
#include <new>
#include <cmath>
#include <chrono>
#include "common.h"

#define BLOCK_SIZE 16

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FABS fabs
  #define FP double
#else
  #define SQRT sqrtf
  #define FABS fabsf
  #define FP float
#endif

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
#include "verify.h"
#endif

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j;

  // 2D arrays on host side.
  FP(*a_host)[N] = new FP[M][N];
  FP(*b_host)[P] = new FP[N][P];
  // host-side cpu result
  FP(*c_host)[P] = new FP[M][P];
  // host-side gpu result
  FP(*c_back)[P] = new FP[M][P];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = (FP)1.0 / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    FP sum = 0;
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
    buffer<FP, 1> a(reinterpret_cast<FP*>(a_host), range<1>(M*N));
    buffer<FP, 1> b(reinterpret_cast<FP*>(b_host), range<1>(N*P));
    buffer<FP, 1> c(reinterpret_cast<FP*>(c_back), range<1>(M*P));

    std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
              << ") * b(" << N << "," << P << ")\n";

    auto grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    auto grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    range<2> gws (grid_rows, grid_cols);
    range<2> lws (BLOCK_SIZE, BLOCK_SIZE);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&](handler &h) {
        auto A = a.get_access<sycl_read>(h);
        auto B = b.get_access<sycl_read>(h);
        auto C = c.get_access<sycl_discard_write>(h);
        h.parallel_for<class hellinger>(nd_range<2>(gws, lws), [=](nd_item<2> index) {
          int row = index.get_global_id(0);
          int col = index.get_global_id(1);
          if( col < P && row < M) {
            FP sum = (FP)0.0;
            for (int i = 0; i < N; i++) {
              sum += sycl::sqrt(A[row * N + i] * B[i * P + col]);
            }
            const FP value = (FP)1.0 - sum;
            const FP gate = (!sycl::signbit(value));
            C[row * P + col] = sycl::sqrt(gate * value);
          }
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat << " (s)\n";
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

