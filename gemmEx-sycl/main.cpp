#include <sys/time.h>
#include <stdio.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

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

template <typename T, typename S>
int mkl_gemm_ex(
    sycl::queue &q, 
    oneapi::mkl::transpose transA,
    oneapi::mkl::transpose transB,
    int m, int n, int k,
    T *A, T *B, S *C,
    int lda, int ldb, int ldc,
    S alpha, S beta)
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
    return 0;
  }
  status.wait();
  return 1;
}

template <typename T, typename S>
void test_gemm(sycl::queue &q,
               const int m, const int n, const int k,
               T *A, T *B, S *C,
               const S alpha, const S beta, int iteration)
{
  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int success = mkl_gemm_ex(q,
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
        
    q.wait();
    gettimeofday(&end, NULL);
    if (success > 0 && i > 0)
      total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
  }
  if (total_time > 0)
    printf("%.3f ms\n", total_time / (iteration - 1));
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  const int iteration = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int m = 4096, n = 8192, k = 1024;
  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  sycl::half h_alpha = sycl::vec<float, 1>{1.f}
                           .convert<sycl::half, sycl::rounding_mode::rte>()[0],
             h_beta = sycl::vec<float, 1>{0.f}
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0];

  const int32_t i_alpha = 1, i_beta = 0;

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  sycl::half *hA, *hB, *hC;
  int8_t *iA, *iB; int32_t *iC;

  allocate_memory(q, m, n, k, &dA, &dB, &dC);
  allocate_memory(q, m, n, k, &fA, &fB, &fC);
  allocate_memory(q, m, n, k, &hA, &hB, &hC);
  allocate_memory(q, m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = sycl::vec<float, 1>{fA[i]}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    iA[i] = float2int8(fA[i], 127);
  } 
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = sycl::vec<float, 1>{fB[i]}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    iB[i] = float2int8(fB[i], 127);
  }

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, dA, dB, dC, d_alpha, d_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, fA, fB, fC, f_alpha, f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, hA, hB, hC, h_alpha, h_beta, iteration);

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

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

  free_memory(q, dA, dB, dC);
  free_memory(q, fA, fB, fC);
  free_memory(q, hA, hB, hC);
  free_memory(q, iA, iB, iC);
  return 0;
}
