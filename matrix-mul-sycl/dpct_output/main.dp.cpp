
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <limits>
#include <cmath>

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

void matrix_mul(const float *a, const float *b, float *c, 
                           const int m, const int n, const int k,
                           sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
              item_ct1.get_local_id(2);
    int col = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
              item_ct1.get_local_id(1);
    if( col < k && row < m)
    {
        float sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

int main() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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

  float *a_device, *b_device, *c_device;

    a_device = (float *)sycl::malloc_device(sizeof(float) * M * N, q_ct1);
    b_device = (float *)sycl::malloc_device(sizeof(float) * N * P, q_ct1);
    c_device = (float *)sycl::malloc_device(sizeof(float) * M * P, q_ct1);

    q_ct1.memcpy(a_device, a_host, sizeof(float) * M * N).wait();
    q_ct1.memcpy(b_device, b_host, sizeof(float) * N * P).wait();

  unsigned int grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //dim3 dimGrid(grid_cols, grid_rows);
    sycl::range<3> dimGrid(grid_rows, grid_cols, 1);
    sycl::range<3> dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = dimGrid * dimBlock;

        auto M_ct3 = M;
        auto N_ct4 = N;
        auto P_ct5 = P;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(dimBlock.get(2), dimBlock.get(1),
                                             dimBlock.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                matrix_mul(a_device, b_device, c_device, M_ct3, N_ct4, P_ct5,
                           item_ct1);
            });
    });
    q_ct1.memcpy(c_back, c_device, sizeof(int) * M * P).wait();

  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
       << ") * b(" << N << "," << P << ")\n";

  int result = 0;
#ifdef VERIFY
  cout << "Result of matrix multiplication using DPC++: ";
  result = VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
    sycl::free(a_device, q_ct1);
    sycl::free(b_device, q_ct1);
    sycl::free(c_device, q_ct1);
  return result;
}

