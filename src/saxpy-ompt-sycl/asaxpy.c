/**
 * @file asaxpy.c
 * @brief Function definition for performing the \c saxpy operation on accelerator.
 *
 * This source file contains function definition for the \c saxpy operation,
 * which is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are single-precision vectors each with n elements.
 *
 * @author Xin Wu (PC²)
 * @date 05.04.2020
 * @copyright CC BY-SA 2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "wtcalc.h"
#include "asaxpy.h"

void asaxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ial)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  struct timespec rt[2];
  int m = (n >> 4);

  switch (ial) {
    case 0:
/*
 * - <<<2^7 , 2^7 >>>, auto   scheduling
 */
#pragma omp target data  device(0) map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
  #pragma omp target teams distribute parallel for device(0) \
  num_teams(128) num_threads(128) dist_schedule(static, 128) shared(a, n, x, y)
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
    break;
    case 1:
/*
 * - <<<2^16, 2^10>>>, manual scheduling
 */
#pragma omp target data  device(0) \
  map(to:a, n, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
  #pragma omp target teams distribute parallel for device(0) \
  num_teams(65536) num_threads(1024) dist_schedule(static, 1024) shared(a, n, x, y)
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
    break;
    case 2:
/*
 * - <<<2^15, 2^7 >>>, manual scheduling, 16x loop unrolling (2^15*2^7*16==2^26)
 */
#pragma omp target data  device(0) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
  #pragma omp target teams distribute parallel for device(0) \
  num_teams(65536/2) num_threads(128) dist_schedule(static, 128) shared(a, m, x, y)
  for (int i = 0; i < m; ++i) {
    y[i          ] = a * x[i          ] + y[i          ];
    y[i +       m] = a * x[i +       m] + y[i +       m];
    y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
    y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
    y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
    y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
    y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
    y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
    y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
    y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
    y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
    y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
    y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
    y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
    y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
    y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
    break;
    case 3:
/*
 * - <<<2^12, 2^7 >>>, auto   scheduling, 16x loop unrolling
 */
#pragma omp target data  device(0) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
  #pragma omp target teams distribute parallel for device(0) \
  num_teams(4096) num_threads(128) dist_schedule(static, 128) shared(a, m, x, y)
  for (int i = 0; i < m; ++i) {
    y[i          ] = a * x[i          ] + y[i          ];
    y[i +       m] = a * x[i +       m] + y[i +       m];
    y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
    y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
    y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
    y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
    y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
    y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
    y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
    y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
    y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
    y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
    y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
    y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
    y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
    y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
    break;
    case 4:
/*
 * - <<<2^16, 2^9>>>:
 *     * de-linearize the vector (convert the vector to matrix)
 *     * collapse the ji-loop
 *     * 2x i-loop unrolling
 */
#pragma omp target data  device(0) \
  map(to:a, x[0:n]) map(tofrom:y[0:n])
{
  clock_gettime(CLOCK_REALTIME, rt + 0);
  #pragma omp target teams distribute parallel for device(0) \
  num_teams(65536) thread_limit(512) dist_schedule(static, 512) \
  collapse(2) shared(a, x, y)
  for (int j = 0; j < 65536; ++j) {
    for (int i = 0; i < 512; ++i) { /* 2x i-loop unrolling */
      y[j * 1024 + i      ] += a * x[j * 1024 + i      ];
      y[j * 1024 + i + 512] += a * x[j * 1024 + i + 512];
    }
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
}
    break;

    default:

/*
 * axpy in MKL
 */
  float *x_dev = sycl::malloc_device<float>(n, q);
  q.memcpy(x_dev, x, sizeof(float) * n);
  float *y_dev = sycl::malloc_device<float>(n, q);
  q.memcpy(y_dev, y, sizeof(float) * n);
  q.wait();

  clock_gettime(CLOCK_REALTIME, rt + 0);
  try {
    oneapi::mkl::blas::row_major::axpy(q, n, a, x_dev, 1, y_dev, 1).wait();
  }
  catch(sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during AXPY:\n"
                  << e.what() << std::endl;
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  q.memcpy(y, y_dev, sizeof(float) * n).wait();
  sycl::free(x_dev, q);
  sycl::free(y_dev, q);
    break;
  } /* end switch (ial) */

  if (wtcalc >= 0.0) {
    wtcalc += (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  }
}
