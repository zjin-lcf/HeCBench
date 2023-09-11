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
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
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

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, "HIPSPARSE API failed at line %d with error: %s\n",         \
               __LINE__, #token_); \
        break

#define CHECK_HIPSPARSE(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);    \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            return local_error;                                                           \
        }                                                                                 \
    }                                                                                     

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 5) {
    printf("The function converts a dense MxN matrix into a sparse matrix\n");
    printf("The sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    printf("nnz is the number of non-zero elements\n");
    return 1;
  }

  int64_t m, n, h_nnz;

  m = atol(argv[1]);
  n = atol(argv[2]);
  h_nnz = atol(argv[3]);
  repeat = atoi(argv[4]);

  // Host problem definition
  const int64_t num_rows = m;
  const int64_t num_cols = n;
  const int64_t dense_size = m * n;

  const int64_t dense_size_bytes  = dense_size * sizeof(float);
  const int64_t value_size_bytes  = h_nnz * sizeof(float);
  const int64_t colidx_size_bytes = h_nnz * sizeof(int64_t);
  const int64_t rowidx_size_bytes = (num_rows + 1) * sizeof(int64_t);

  float *h_dense = (float*) malloc (dense_size_bytes);

  // device results
  float *h_csr_values = (float*) malloc (value_size_bytes);
  int64_t *h_csr_columns = (int64_t*) malloc (colidx_size_bytes);
  int64_t *h_csr_offsets = (int64_t*) malloc (rowidx_size_bytes);

  // reference results
  float *h_csr_values_result = (float*) malloc (value_size_bytes);
  int64_t *h_csr_columns_result = (int64_t*) malloc (colidx_size_bytes);
  int64_t *h_csr_offsets_result = (int64_t*) malloc (rowidx_size_bytes);

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, num_rows, num_cols, h_nnz);
  init_csr(h_csr_offsets_result, h_csr_values_result, h_csr_columns_result,
           h_dense, num_rows, num_cols, h_nnz);
  //--------------------------------------------------------------------------
  // Device memory management
  int64_t *d_csr_offsets, *d_csr_columns;
  float *d_csr_values,  *d_dense;
  CHECK_HIP( hipMalloc((void**) &d_dense, dense_size_bytes))
  CHECK_HIP( hipMalloc((void**) &d_csr_offsets,
                       (num_rows + 1) * sizeof(int64_t)) )
  CHECK_HIP( hipMemcpy(d_dense, h_dense, dense_size_bytes,
                       hipMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // HIPSPARSE APIs
  hipsparseHandle_t     handle = NULL;
  hipsparseSpMatDescr_t matB;
  hipsparseDnMatDescr_t matA;
  void*                 dBuffer    = NULL;
  size_t                bufferSize = 0;
  CHECK_HIPSPARSE( hipsparseCreate(&handle) )
  // Create dense matrix A
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matA, num_rows, num_cols, num_cols, d_dense,
                                        HIP_R_32F, HIPSPARSE_ORDER_ROW) )

  // Create sparse matrix B in CSR format (nnz is not known yet)
  CHECK_HIPSPARSE( hipsparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      HIPSPARSE_INDEX_64I, HIPSPARSE_INDEX_64I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F) )

  // allocate an external buffer if needed
  CHECK_HIPSPARSE( hipsparseDenseToSparse_bufferSize(
                                      handle, matA, matB,
                                      HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                      &bufferSize) )
  CHECK_HIP( hipMalloc(&dBuffer, bufferSize) )

  // updates the number of non-zero elements in the sparse matrix descriptor
  CHECK_HIPSPARSE( hipsparseDenseToSparse_analysis(handle, matA, matB,
                                      HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                      dBuffer) )
  // get number of non-zero elements
  int64_t num_rows_tmp, num_cols_tmp, nnz;
  CHECK_HIPSPARSE( hipsparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )
  // allocate CSR column indices and values
  CHECK_HIP( hipMalloc((void**) &d_csr_columns, nnz * sizeof(int64_t)) )
  CHECK_HIP( hipMalloc((void**) &d_csr_values,  nnz * sizeof(float)) )
  // reset offsets, column indices, and values pointers
  CHECK_HIPSPARSE( hipsparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values) )
  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute Dense to Sparse conversion
    CHECK_HIPSPARSE( hipsparseDenseToSparse_convert(handle, matA, matB,
                                        HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of DenseToSparse_convert : %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matA) )
  CHECK_HIPSPARSE( hipsparseDestroySpMat(matB) )
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_HIP( hipMemcpy(h_csr_offsets, d_csr_offsets,
                       (num_rows + 1) * sizeof(int64_t),
                       hipMemcpyDeviceToHost) )
  CHECK_HIP( hipMemcpy(h_csr_columns, d_csr_columns, nnz * sizeof(int64_t),
                       hipMemcpyDeviceToHost) )
  CHECK_HIP( hipMemcpy(h_csr_values, d_csr_values, nnz * sizeof(float),
                       hipMemcpyDeviceToHost) )

  int correct = 1; 
  if (h_nnz != nnz) {
    printf("nnz: %ld != %ld\n", h_nnz, nnz);
    correct = 0;
    goto print_error;
  }

  for (int64_t i = 0; i < num_rows + 1; i++) {
    if (h_csr_offsets[i] != h_csr_offsets_result[i]) {
      printf("rowidx: @%ld %ld != %ld\n", i, h_csr_offsets[i], h_csr_offsets_result[i]);
      correct = 0;
      goto print_error;
    }
  }
  for (int64_t i = 0; i < nnz; i++) {
    if (h_csr_columns[i] != h_csr_columns_result[i]) {
      printf("colidx: @%ld %ld != %ld\n", i, h_csr_columns[i], h_csr_columns_result[i]);
      correct = 0;
      goto print_error;
    }
  }
  for (int64_t i = 0; i < nnz; i++) {
    if (h_csr_values[i] != h_csr_values_result[i]) {
      printf("value: @%ld %f != %f\n", i, h_csr_values[i], h_csr_values_result[i]);
      correct = 0;
      goto print_error;
    }
  }

  print_error:
  if (correct)
      printf("dense2sparse_csr_example test PASSED\n");
  else
      printf("dense2sparse_csr_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_HIP( hipFree(dBuffer) )
  CHECK_HIP( hipFree(d_csr_offsets) )
  CHECK_HIP( hipFree(d_csr_columns) )
  CHECK_HIP( hipFree(d_csr_values) )
  CHECK_HIP( hipFree(d_dense) )
  free(h_dense);
  free(h_csr_values);
  free(h_csr_columns);
  free(h_csr_offsets);
  free(h_csr_values_result);
  free(h_csr_columns_result);
  free(h_csr_offsets_result);
  return EXIT_SUCCESS;
}
