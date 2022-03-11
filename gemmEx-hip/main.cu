#include <sys/time.h>
#include <stdio.h>
#include <type_traits> // is_same
#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
  hipMallocManaged(A, m * k * sizeof(T));
  hipMallocManaged(B, k * n * sizeof(T));
  hipMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
  hipFree(A);
  hipFree(B);
  hipFree(C);
}

template <typename T, typename S>
int cublas_gemm_ex(
    hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB,
    const int m, const int n, const int k,
    T *A, T *B, S *C,
    int lda, int ldb, int ldc,
    const S *alpha, const S *beta, int algo)
{
  hipblasDatatype_t AType, BType, CType, ComputeType;
  if (std::is_same<T, double>::value) {
    AType = BType = CType = ComputeType = HIPBLAS_R_64F;
  } else if (std::is_same<T, float>::value) {
    AType = BType = CType = ComputeType = HIPBLAS_R_32F;
  } else if (std::is_same<T, __half>::value) {
    AType = BType = CType = ComputeType = HIPBLAS_R_16F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = BType = HIPBLAS_R_8I;
    CType = ComputeType = HIPBLAS_R_32I;
  } else {
    printf("Not supported data type.");
    return -1;
  }

  hipblasStatus_t status;
  status = hipblasGemmEx(
      handle,
      transA,
      transB,
      m,
      n,
      k,
      alpha,
      A,
      AType,
      lda,
      B,
      BType,
      ldb,
      beta,
      C,
      CType,
      ldc,
      ComputeType,
      static_cast<hipblasGemmAlgo_t>(algo));

  if (status == HIPBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void test_gemm(hipblasHandle_t handle,
  const int m,  const int n,  const int k,
  T *A, T *B, S *C,
  const S *alpha, const S *beta, int algo, const int iteration)
{
  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    hipDeviceSynchronize();
    gettimeofday(&start, NULL);
    int success = cublas_gemm_ex(handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
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
        beta,
        static_cast<hipblasGemmAlgo_t>(algo));
    hipDeviceSynchronize();
    gettimeofday(&end, NULL);
    if (success > 0 && i > 0)
      total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
  }
  if (total_time > 0)
    printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  const int iteration = atoi(argv[1]);

  const int m = 4096, n = 8192, k = 1024;
  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
  int start_algo = HIPBLAS_GEMM_DEFAULT;
  int end_algo = HIPBLAS_GEMM_DEFAULT;

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  const __half h_alpha = __float2half_rn(1.f), h_beta = __float2half_rn(0.f);
  const int32_t i_alpha = 1, i_beta = 0;

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  __half *hA, *hB, *hC;
  int8_t *iA, *iB; int32_t *iC;

  allocate_memory(m, n, k, &dA, &dB, &dC);
  allocate_memory(m, n, k, &fA, &fB, &fC);
  allocate_memory(m, n, k, &hA, &hB, &hC);
  allocate_memory(m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = __float2half_rn(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  } 
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = __float2half_rn(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }
  hipblasHandle_t handle;
  hipblasCreate(&handle);

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo <= end_algo; ++algo)
    test_gemm(handle, m, n, k, dA, dB, dC, &d_alpha, &d_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo <= end_algo; ++algo)
    test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo <= end_algo; ++algo)
    test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  for (int algo = start_algo; algo <= end_algo; ++algo)
    test_gemm(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("fp64: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5lf%c", fC[i], " \n"[i==9]);

  printf("fp32: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i==9]);

  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i==9]);

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

  free_memory(dA, dB, dC);
  free_memory(fA, fB, fC);
  free_memory(hA, hB, hC);
  free_memory(iA, iB, iC);
  return 0;
}

