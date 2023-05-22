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

  std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
            << ") * b(" << N << "," << P << ")\n";

  // 2D arrays on host side.
  FP(*h_a)[N] = new FP[M][N];
  FP(*h_b)[P] = new FP[N][P];
  // host-side cpu result
  FP(*h_c)[P] = new FP[M][P];
  // host-side gpu result
  FP(*c_back)[P] = new FP[M][P];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      h_a[i][j] = (FP)1.0 / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      h_b[i][j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    FP sum = 0;
    for (i = 0; i < N; i++)
      sum += h_b[i][j];
    for (i = 0; i < N; i++)
      h_b[i][j] /= sum;
  }

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Create buffers for matrices, buffer c is bound with host memory c_back
  FP *d_a = sycl::malloc_device<FP>(M*N, q);
  q.memcpy(d_a, h_a, sizeof(FP)*M*N);

  FP *d_b = sycl::malloc_device<FP>(N*P, q);
  q.memcpy(d_b, h_b, sizeof(FP)*N*P);

  FP *d_c = sycl::malloc_device<FP>(M*P, q);

  auto grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  auto grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  sycl::range<2> gws (grid_rows, grid_cols);
  sycl::range<2> lws (BLOCK_SIZE, BLOCK_SIZE);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class hellinger>(
        sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> index) {
        int row = index.get_global_id(0);
        int col = index.get_global_id(1);
        if( col < P && row < M) {
          FP sum = (FP)0.0;
          for (int i = 0; i < N; i++) {
            sum += sycl::sqrt(d_a[row * N + i] * d_b[i * P + col]);
          }
          const FP value = (FP)1.0 - sum;
          const FP gate = (!sycl::signbit(value));
          d_c[row * P + col] = sycl::sqrt(gate * value);
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat << " (s)\n";
  
  q.memcpy(c_back, d_c, sizeof(int)*M*P).wait();

#ifdef VERIFY
  VerifyResult(h_a, h_b, h_c, c_back);
#endif

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  delete[] c_back;
  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);
  return 0;
}
