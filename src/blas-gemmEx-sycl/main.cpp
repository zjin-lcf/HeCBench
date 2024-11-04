#include <sys/time.h>
#include <stdio.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "utils.h"

using sycl::ext::oneapi::bfloat16;

template <typename T, typename S>
void allocate_memory(sycl::queue &q, int m, int n, int k, T **A, T **B, S **C) {
  *A = sycl::malloc_shared<T> (m * k, q);
  *B = sycl::malloc_shared<T> (k * n, q);
  *C = sycl::malloc_shared<S> (m * n, q);
}

template <typename T, typename S>
void free_memory(sycl::queue &q, T *A, T *B, S *C) {
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
}

template <typename Ta, typename Tc, typename Ts>
bool mkl_gemm_ex(
    sycl::queue &q,
    oneapi::mkl::transpose transA,
    oneapi::mkl::transpose transB,
    int m, int n, int k,
    Ta *A, Ta *B, Tc *C,
    int lda, int ldb, int ldc,
    Ts alpha, Ts beta)
{
  sycl::event status;
  try {
    status = oneapi::mkl::blas::column_major::gemm(
      q,
      transA, transB,
      m, n, k,
      alpha, A, lda,
      B, ldb, beta,
      C, ldc);
  } catch(sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
              << e.what() << std::endl;
    return false;
  }
  status.wait();
  return true;
}

template <typename Ta, typename Tc, typename Ts>
void test_gemm(sycl::queue &q,
               const int m, const int n, const int k,
               Ta *A, Ta *B, Tc *C,
               const Ts alpha, const Ts beta, int iteration)
{
  double total_time = 0;
  struct timeval start, end;

  for (int i = 0; i < iteration; ++i) {
    gettimeofday(&start, NULL);
    bool success = mkl_gemm_ex(q,
                               oneapi::mkl::transpose::nontrans,
                               oneapi::mkl::transpose::nontrans,
                               n, // number of rows of matrix A and C
                               m, // number of columns of matrix B and C
                               k, // number of columns of A and rows of B
                               B,
                               A,
                               C,
                               n, // lda
                               k, // ldb
                               n, // ldc
                               alpha,
                               beta);
    gettimeofday(&end, NULL);
    if (!success) break;
    else if (i > 0) {
      total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
  }
  if (total_time > 0.0) {
    double avg_time = total_time / (iteration - 1);
    printf("%.3f ms\n", avg_time);
    performance(m, n, k, std::is_same<Ta, int8_t>::value, avg_time * 1e-3);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
    printf("C = A X B (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int iteration = atoi(argv[4]);

  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  sycl::half h_alpha = sycl::vec<float, 1>{1.f}
                           .convert<sycl::half, sycl::rounding_mode::rte>()[0],
             h_beta = sycl::vec<float, 1>{0.f}
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const float bf_alpha = 1.f, bf_beta = 0.f;
  const int32_t i_alpha = 1, i_beta = 0;

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  sycl::half *hA, *hB, *hC;
  bfloat16 *bfA, *bfB, *bfC;
  int8_t *iA, *iB; int32_t *iC;

  allocate_memory(q, m, n, k, &dA, &dB, &dC);
  allocate_memory(q, m, n, k, &fA, &fB, &fC);
  allocate_memory(q, m, n, k, &hA, &hB, &hC);
  allocate_memory(q, m, n, k, &bfA, &bfB, &bfC);
  allocate_memory(q, m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = sycl::vec<float, 1>{fA[i]}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    bfA[i] = bfloat16(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = sycl::vec<float, 1>{fB[i]}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    bfB[i] = bfloat16(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, dA, dB, dC, d_alpha, d_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, fA, fB, fC, f_alpha, f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, hA, hB, hC, h_alpha, h_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test bfloat16 >>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, bfA, bfB, bfC, bf_alpha, bf_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, iA, iB, iC, i_alpha, i_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("fp64: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5lf%c", dC[i], " \n"[i==9]);

  printf("fp32: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i==9]);

  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i==9]);

  printf("bf16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(bfC[i]), " \n"[i==9]);

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

  free_memory(q, dA, dB, dC);
  free_memory(q, fA, fB, fC);
  free_memory(q, hA, hB, hC);
  free_memory(q, bfA, bfB, bfC);
  free_memory(q, iA, iB, iC);
  return 0;
}
