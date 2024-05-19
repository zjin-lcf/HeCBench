/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include "utils.h"

#define CHECK_HIP(func)                                                      \
{                                                                            \
    hipError_t status = (func);                                              \
    if (status != hipSuccess) {                                              \
        printf("HIP API failed at line %d with error: %s (%d)\n",            \
               __LINE__, hipGetErrorString(status), status);                 \
        return EXIT_FAILURE;                                                 \
    }                                                                        \
}

#define CHECK_HIPSPARSE(func)                                                \
{                                                                            \
    hipsparseStatus_t status = (func);                                       \
    if (status != HIPSPARSE_STATUS_SUCCESS) {                                \
        printf("HIPSPARSE API failed at line %d with error: %s (%d)\n",      \
               __LINE__, hipsparseGetErrorString(status), status);           \
        return EXIT_FAILURE;                                                 \
    }                                                                        \
}

int COO(int m, int k, int n, int a_nnz, int repeat, int verify) {
  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = n;
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int ldc        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;
  const int C_size     = ldc * A_num_rows;

  float *hA = (float*) malloc (A_size * sizeof(float));
  float *hB = (float*) malloc (B_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = A_nnz * sizeof(int);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_rows = (int*) malloc (A_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_coo(hA_rows, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  init_matrix(hB, B_num_rows, B_num_cols, B_size);

  float               alpha       = 1.0f;
  float               beta        = 0.0f;
  hipsparseOperation_t opA        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  hipsparseOperation_t opB        = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_rows, *dA_columns;
  float *dA_values, *dB, *dC;
  // allocate A
  CHECK_HIP( hipMalloc((void**) &dA_rows, A_nnz * sizeof(int)) )
  CHECK_HIP( hipMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
  CHECK_HIP( hipMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
  // allocate B
  CHECK_HIP( hipMalloc((void**) &dB, B_size * sizeof(float)) )
  // allocate C
  CHECK_HIP( hipMalloc((void**) &dC, C_size * sizeof(float)) )

  // copy A
  CHECK_HIP( hipMemcpy(dA_rows, hA_rows, A_nnz * sizeof(int),
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dA_values, hA_values,
                       A_nnz * sizeof(float), hipMemcpyHostToDevice) )
  // copy B
  CHECK_HIP( hipMemcpy(dB, hB, B_size * sizeof(float), hipMemcpyHostToDevice) )

  //--------------------------------------------------------------------------
  // HIPSPARSE APIs
  hipsparseHandle_t     handle = NULL;
  hipsparseSpMatDescr_t matA;
  hipsparseDnMatDescr_t matB, matC;
  void*  dBuffer    = NULL;
  size_t bufferSize = 0;
  CHECK_HIPSPARSE( hipsparseCreate(&handle) )
  // Create sparse matrix A in COO format
  CHECK_HIPSPARSE( hipsparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_rows, dA_columns, dA_values,
                                      HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F) )
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb,
                                        dB, HIP_R_32F, HIPSPARSE_ORDER_ROW) )
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc,
                                        dC, HIP_R_32F, HIPSPARSE_ORDER_ROW) )

  // allocate an external buffer if needed
  CHECK_HIPSPARSE( hipsparseSpMM_bufferSize(
                               handle,
                               opA,
                               opB,
                               &alpha, matA, matB, &beta, matC, HIP_R_32F,
                               HIPSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  CHECK_HIP( hipMalloc(&dBuffer, bufferSize) )

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    // compute the product of A * B
    CHECK_HIPSPARSE( hipsparseSpMM(handle,
                                   opA,
                                   opB,
                                   &alpha, matA, matB, &beta, matC, HIP_R_32F,
                                   HIPSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPGEMM (COO) compute: %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroySpMat(matA) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matB) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matC) )
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference SPGEMM results..\n");
    float *hC = (float*) malloc (C_size * sizeof(float));
    gemm (hA, hB, hC, A_num_cols, A_num_rows, B_num_cols);

    float *hC_tmp = (float*) malloc (C_size * sizeof(float));
    CHECK_HIP( hipMemcpy(hC_tmp, dC, C_size * sizeof(float),
                           hipMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < C_size; i++) {
      if (fabsf(hC_tmp[i] - hC[i]) > 1e-2f) {
        printf("@%d %f != %f\n", i, hC_tmp[i], hC[i]);
        correct = 0;                       
        break;
      }
    }

    free(hC_tmp);
    free(hC);

    if (correct)
      printf("spgemm_example test PASSED\n");
    else
      printf("spgemm_example test FAILED: wrong result\n");
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_HIP( hipFree(dBuffer) )
  CHECK_HIP( hipFree(dA_rows) )
  CHECK_HIP( hipFree(dA_columns) )
  CHECK_HIP( hipFree(dA_values) )
  CHECK_HIP( hipFree(dB) )
  CHECK_HIP( hipFree(dC) )
  free(hA);
  free(hB);
  free(hA_values);
  free(hA_columns);
  free(hA_rows);
  return EXIT_SUCCESS;
}

int CSR(int m, int k, int n, int a_nnz, int repeat, int verify) {
  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = n;
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int ldc        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;
  const int C_size     = ldc * A_num_rows;

  float *hA = (float*) malloc (A_size * sizeof(float));
  float *hB = (float*) malloc (B_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(int);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_offsets = (int*) malloc (A_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  init_matrix(hB, B_num_rows, B_num_cols, B_size);

  float               alpha       = 1.0f;
  float               beta        = 0.0f;
  hipsparseOperation_t opA        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  hipsparseOperation_t opB        = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_offsets, *dA_columns;
  float *dA_values, *dB, *dC;
  // allocate A
  CHECK_HIP( hipMalloc((void**) &dA_offsets,
                         (A_num_rows + 1) * sizeof(int)) )
  CHECK_HIP( hipMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
  CHECK_HIP( hipMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )
  // allocate B
  CHECK_HIP( hipMalloc((void**) &dB, B_size * sizeof(float)) )
  // allocate C
  CHECK_HIP( hipMalloc((void**) &dC, C_size * sizeof(float)) )

  // copy A
  CHECK_HIP( hipMemcpy(dA_offsets, hA_offsets,
                       (A_num_rows + 1) * sizeof(int),
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                       hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dA_values, hA_values,
                       A_nnz * sizeof(float), hipMemcpyHostToDevice) )
  // copy B
  CHECK_HIP( hipMemcpy(dB, hB, B_size * sizeof(float), hipMemcpyHostToDevice) )

  //--------------------------------------------------------------------------
  // HIPSPARSE APIs
  hipsparseHandle_t     handle = NULL;
  hipsparseSpMatDescr_t matA;
  hipsparseDnMatDescr_t matB, matC;
  void*  dBuffer    = NULL;
  size_t bufferSize = 0;
  CHECK_HIPSPARSE( hipsparseCreate(&handle) )
  // Create sparse matrix A in CSR format
  CHECK_HIPSPARSE( hipsparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_offsets, dA_columns, dA_values,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F) )
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb,
                                        dB, HIP_R_32F, HIPSPARSE_ORDER_ROW) )
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc,
                                        dC, HIP_R_32F, HIPSPARSE_ORDER_ROW) )

  // allocate an external buffer if needed
  CHECK_HIPSPARSE( hipsparseSpMM_bufferSize(
                               handle,
                               opA,
                               opB,
                               &alpha, matA, matB, &beta, matC, HIP_R_32F,
                               HIPSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  CHECK_HIP( hipMalloc(&dBuffer, bufferSize) )

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    // compute the product of A * B
    CHECK_HIPSPARSE( hipsparseSpMM(handle,
                                   opA,
                                   opB,
                                   &alpha, matA, matB, &beta, matC, HIP_R_32F,
                                   HIPSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPGEMM (CSR) compute: %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroySpMat(matA) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matB) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matC) )
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference SPGEMM results..\n");
    float *hC = (float*) malloc (C_size * sizeof(float));
    gemm (hA, hB, hC, A_num_cols, A_num_rows, B_num_cols);

    float *hC_tmp = (float*) malloc (C_size * sizeof(float));
    CHECK_HIP( hipMemcpy(hC_tmp, dC, C_size * sizeof(float),
                           hipMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < C_size; i++) {
      if (fabsf(hC_tmp[i] - hC[i]) > 1e-2f) {
        printf("@%d %f != %f\n", i, hC_tmp[i], hC[i]);
        correct = 0;                       
        break;
      }
    }

    free(hC_tmp);
    free(hC);

    if (correct)
      printf("spgemm_example test PASSED\n");
    else
      printf("spgemm_example test FAILED: wrong result\n");
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_HIP( hipFree(dBuffer) )
  CHECK_HIP( hipFree(dA_offsets) )
  CHECK_HIP( hipFree(dA_columns) )
  CHECK_HIP( hipFree(dA_values) )
  CHECK_HIP( hipFree(dB) )
  CHECK_HIP( hipFree(dC) )
  free(hA);
  free(hB);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 7) {
    printf("Single-precision sparse matrix-dense matrix multiplication into dense matrix,\n");
    printf("where the sparse matrix is represented in COO and CSR storage format\n");
    printf("Usage %s <M> <K> <N> <A_nnz> <repeat> <verify>\n", argv[0]);
    printf("SPMM (A, B, C) where (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }

  int m, k, n, a_nnz, verify;

  m = atoi(argv[1]);
  k = atoi(argv[2]);
  n = atoi(argv[3]);
  a_nnz = atoi(argv[4]);
  repeat = atoi(argv[5]);
  verify = atoi(argv[6]);

  COO(m, k, n, a_nnz, repeat, verify);
  CSR(m, k, n, a_nnz, repeat, verify);

  return EXIT_SUCCESS;
}
