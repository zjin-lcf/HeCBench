#include <stdio.h>
#include <chrono>
#include <type_traits> // is_same
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocblas/rocblas.h>
#include "utils.h"

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

// T: aType, bType; S: cType; CT: compute type
template <typename T, typename S, typename CT>
bool gemm_ex(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    const int m, const int n, const int k,
    T *A, T *B, S *C,
    int lda, int ldb, int ldc,
    const CT *alpha, const CT *beta,
    rocblas_gemm_algo algo)
{
  int solution_idx = 0;

  rocblas_datatype AType, BType, CType, ComputeType;
  if (std::is_same<T, double>::value) {
    AType = BType = CType = ComputeType = rocblas_datatype_f64_r;

  } else if (std::is_same<T, float>::value) {
    AType = BType = CType = ComputeType = rocblas_datatype_f32_r;

  } else if (std::is_same<T, __half>::value) {
    AType = BType = CType = rocblas_datatype_f16_r;
    if (std::is_same<CT, __half>::value)
      ComputeType = rocblas_datatype_f16_r;
    else
      ComputeType = rocblas_datatype_f32_r;
    // TODO rocblas_datatype_f16_r = a_type = b_type; rocblas_datatype_f32_r = c_type = d_type = compute_type
  } else if (std::is_same<T, hip_bfloat16>::value) {
    AType = BType = CType = rocblas_datatype_bf16_r;
    ComputeType = rocblas_datatype_f32_r;
    // TODO rocblas_datatype_bf16_r = a_type = b_type; rocblas_datatype_f32_r = c_type = d_type = compute_type
  } else if (std::is_same<T, int8_t>::value) {
    AType = BType = rocblas_datatype_i8_r;
    CType = ComputeType = rocblas_datatype_i32_r;
  } else {
    printf("Not supported data type.");
    return -1;
  }

  rocblas_status status = rocblas_gemm_ex(handle,
                                          transA, transB,
                                          m, n, k,
                                          alpha, A, AType, lda,
                                          B, BType, ldb, beta,
                                          nullptr, CType, ldc,
                                          C, CType, ldc,  // output
                                          ComputeType,
                                          algo, solution_idx, rocblas_gemm_flags_none);

  return (status == rocblas_status_success);
}

template <typename T, typename S, typename CT>
void test_gemm(rocblas_handle handle,
               const int m,  const int n,  const int k,
               T *A, T *B, S *C,
               const CT *alpha, const CT *beta,
               rocblas_gemm_algo algo, const int iteration)
{
  double total_time = 0;

  for (int i = 0; i < iteration; ++i) {
    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    bool success = gemm_ex(handle,
                           rocblas_operation_none,
                           rocblas_operation_none,
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
                           algo);
    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    if (!success) break;
    else if (i > 0) {
      total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
  }
  if (total_time > 0.0) {
    double avg_time = total_time / (iteration - 1);
    printf("algo %d: %.3f ms\n", algo, avg_time);
    performance(m, n, k, std::is_same<T, int8_t>::value, avg_time * 1e-3);
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
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  const __half h_alpha = __float2half_rn(1.f),
               h_beta = __float2half_rn(0.f);
  const float h_alpha2 = 1.f, h_beta2 = 0.f;
  const float bf_alpha(1.f), bf_beta(0.f);
  const int32_t i_alpha = 1, i_beta = 0;

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  __half *hA, *hB, *hC;
  hip_bfloat16 *bfA, *bfB, *bfC;
  int8_t *iA, *iB; int32_t *iC;

  allocate_memory(m, n, k, &dA, &dB, &dC);
  allocate_memory(m, n, k, &fA, &fB, &fC);
  allocate_memory(m, n, k, &hA, &hB, &hC);
  allocate_memory(m, n, k, &bfA, &bfB, &bfC);
  allocate_memory(m, n, k, &iA, &iB, &iC);

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = __float2half_rn(fA[i]);
    bfA[i] = hip_bfloat16(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = __float2half_rn(fB[i]);
    bfB[i] = hip_bfloat16(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(handle, m, n, k, dA, dB, dC, &d_alpha, &d_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 (compute type fp16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha2, &h_beta2, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test bfloat16 (compute type fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(handle, m, n, k, bfA, bfB, bfC, &bf_alpha, &bf_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration);

  printf(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>\n");
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

  rocblas_destroy_handle(handle);
  free_memory(dA, dB, dC);
  free_memory(fA, fB, fC);
  free_memory(hA, hB, hC);
  free_memory(bfA, bfB, bfC);
  free_memory(iA, iB, iC);
  return 0;
}
