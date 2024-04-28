/*******************************************************************************
 * Copyright 2022 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted  materials, and
 * your use of  them is  governed by the  express license  under which  they
 *were provided to you (License).  Unless the License provides otherwise, you
 *may not use, modify, copy, publish, distribute,  disclose or transmit this
 *software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents  are provided as  is,  with no
 *express or implied  warranties,  other  than those  that are  expressly stated
 *in the License.
 *******************************************************************************/

/*
 *  Content:
 *       This example demonstrates use of oneapi::mkl::lapack::gels_batch
 *       to perform batched calculation of least squares.
 *
 *       The supported floating point data types for matrix data are:
 *           float
 *           double
 *           std::complex<float>
 *           std::complex<double>
 *******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <complex>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

template <typename data_t, typename real_t = decltype(std::real((data_t)0)),
          bool is_real = std::is_same_v<data_t, real_t>>
int run_gels_batch_example(sycl::queue &q, const int repeat) {
  oneapi::mkl::transpose nontrans = oneapi::mkl::transpose::nontrans;
  const int64_t m = 5, n = 5, nrhs = 1, lda = m, stride_a = n * lda, ldb = m,
                stride_b = nrhs * ldb, batch_size = 2;

  auto v = [](real_t arg) {
    if constexpr (is_real)
      return arg;
    else
      return data_t{0, arg};
  };

  data_t A[] = {
      v( 1.0), v( 0.0), v( 0.0), v( 0.0), v( 0.0),
      v( 1.0), v( 0.2), v(-0.4), v(-0.4), v(-0.8),
      v( 1.0), v( 0.6), v(-0.2), v( 0.4), v(-1.2),
      v( 1.0), v( 1.0), v(-1.0), v( 0.6), v(-0.8),
      v( 1.0), v( 1.8), v(-0.6), v( 0.2), v(-0.6)
                                                 ,
      v( 0.2), v(-0.4), v(-0.4), v(-0.8), v( 0.0),
      v( 0.4), v( 0.2), v( 0.8), v(-0.4), v( 0.0),
      v( 0.4), v(-0.8), v( 0.2), v( 0.4), v( 0.0),
      v( 0.8), v( 0.4), v(-0.4), v( 0.2), v( 0.0),
      v( 0.0), v( 0.0), v( 0.0), v( 0.0), v( 1.0)
  };

  data_t B[] = {
      v(5.0), v(3.6),  v(-2.2), v(0.8),  v(-3.4),
      v(1.8), v(-0.6), v(0.2),  v(-0.6), v(1.0),
  };

  data_t X[] = {
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  };


  data_t *A_dev = sycl::aligned_alloc_device<data_t>(64, stride_a * batch_size, q);
  data_t *B_dev = sycl::aligned_alloc_device<data_t>(64, stride_b * batch_size, q);

  int64_t scratchpad_size =
      oneapi::mkl::lapack::gels_batch_scratchpad_size<data_t>(
          q, nontrans, m, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
  data_t *scratchpad =
      sycl::aligned_alloc_device<data_t>(64, scratchpad_size, q);

  long time = 0;
  for (int i = 0; i <= repeat; i++) {
    q.copy(A, A_dev, stride_a * batch_size);
    q.copy(B, B_dev, stride_b * batch_size);

    q.wait();
    auto start = std::chrono::steady_clock::now();
    
    oneapi::mkl::lapack::gels_batch(q, nontrans, m, n, nrhs, A_dev, lda,
                                    stride_a, B_dev, ldb, stride_b, batch_size,
                                    scratchpad, scratchpad_size).wait();

    auto end = std::chrono::steady_clock::now();
    if (i != 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time : %f (us)\n", (time * 1e-3f) / repeat);

  q.copy(B_dev, B, stride_b * batch_size).wait();

  const real_t bound = std::is_same_v<real_t, float> ? 1e-6 : 1e-8;
  bool passed = true;

  printf("Results:\n");
  auto print = [](data_t &v) {
    if constexpr (is_real)
      printf("%6.2f", v);
    else
      printf("<%6.2f,%6.2f> ", v.real(), v.imag());
  };
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < n; j++) {
      data_t result = B[i * stride_b + j];
      data_t residual = result - X[j + i * m];
      passed = passed and (result == result) and
               (std::sqrt(std::abs(std::real(residual * residual))) < bound);
      print(result);
    }
    printf("\n");
  }

  sycl::free(scratchpad, q);
  sycl::free(A_dev, q);
  sycl::free(B_dev, q);

  if (passed) {
    printf("Calculations successfully finished\n");
  } else {
    printf("ERROR: results mismatch!\n");
    printf("Expected:\n");
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < m; j++) {
        print(X[j + i * m]);
      }
      printf("\n");
    }
    return 1;
  }

  return 0;
}

//
// Description of example setup, APIs used and supported floating point type
// precisions
//
void print_info() {
  std::cout << "" << std::endl;
  std::cout << "########################################################################" << std::endl;
  std::cout << "# Batched strided GELS example:" << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "# Computes least squares of a batch of matrices and right hand sides." << std::endl;
  std::cout << "# Supported floating point type precisions:" << std::endl;
  std::cout << "#   float" << std::endl;
  std::cout << "#   double" << std::endl;
  std::cout << "#   std::complex<float>" << std::endl;
  std::cout << "#   std::complex<double>" << std::endl;
  std::cout << "# " << std::endl;
  std::cout << "########################################################################" << std::endl;
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  print_info();

  bool failed = false;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::cout << "Running with single precision real data type:" << std::endl;
  failed |= run_gels_batch_example<float>(q, repeat);

  std::cout << "Running with single precision complex data type:" << std::endl;
  failed |= run_gels_batch_example<std::complex<float>>(q, repeat);

  std::cout << "Running with double precision real data type:" << std::endl;
  failed |= run_gels_batch_example<double>(q, repeat);

  std::cout << "Running with double precision complex data type:" << std::endl;
  failed |= run_gels_batch_example<std::complex<double>>(q, repeat);

  return failed;
}
