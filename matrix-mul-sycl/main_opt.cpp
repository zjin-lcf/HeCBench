#include <CL/sycl.hpp>
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
constexpr int m_size = 2048 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#ifdef VERIFY
/**
 * Perform matrix multiplication on host to verify results from device.
 */
bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}

int VerifyResult(float (*a_host)[N], float (*b_host)[P], float (*c_host)[P], float (*c_back)[P]) {
  // Check that the results are correct by comparing with host computing.
  int i, j, k;

  // c_host is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = 0.0f;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      // Each element of the product is just the sum 1+2+...+n
      for (j = 0; j < P; j++) {
        c_host[i][j] += a_host[i][k] * b_host[k][j];
      }
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
    cout << "Success - The results are correct!\n";
    return 0;
  } else {
    cout << "Fail - The results mismatch!\n";
    return -1;
  }
}
#endif

int main() {
  int i, j;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[P] = new float[N][P];
  // host-side cpu result
  float(*c_host)[P] = new float[M][P];
  // host-side gpu result
  float(*c_back)[P] = new float[M][P];

  // Each element of matrix a is 1.
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a_host[i][j] = 1.0f;

  // Each column of b_host is the sequence 1,2,...,N
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b_host[i][j] = i + 1.0f;


  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
  {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    // Create 2D buffers for matrices, buffer c is bound with host memory c_back
    buffer<float, 1> a(reinterpret_cast<float*>(a_host), range(M*N));
    buffer<float, 1> b(reinterpret_cast<float*>(b_host), range(N*P));
    buffer<float, 1> c(reinterpret_cast<float*>(c_back), range(M*P));

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    auto grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    auto grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    auto local_ndrange = range<2>(BLOCK_SIZE, BLOCK_SIZE);
    auto global_ndrange = range<2>(grid_rows, grid_cols);

    // Submit command group to queue to multiply matrices: c = a * b
    q.submit([&](handler &h) {
      // Read from a and b, write to c
      auto A = a.get_access<sycl_read>(h);
      auto B = b.get_access<sycl_read>(h);
      auto C = c.get_access<sycl_discard_write>(h);

      // Execute kernel.
      h.parallel_for(nd_range<2>(global_ndrange, local_ndrange), [=](nd_item<2> index) {
        int row = index.get_global_id(0);
        int col = index.get_global_id(1);
        if( col < P && row < M) {
          float sum = 0;
          // Compute the result of one element of c
          for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * P + col];
          }
          C[row * P + col] = sum;
	}
      });
    });
  }

  int result = 0;
#ifdef VERIFY
  cout << "Result of matrix multiplication using DPC++: ";
  result = VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return result;
}

