#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
#include "mv.h"

// sparse matrix vector multiply using the CSR format
__global__ void mv_csr(const int num_rows,
                       const size_t *row_indices,
                       const int *col_indices,
                       const REAL *values,
                       const REAL *x,
                             REAL *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    size_t row_start = row_indices[i];
    size_t row_end = row_indices[i+1];

    REAL temp = 0;
    for(size_t n = row_start; n < row_end; n++){
      temp += values[n] * x[col_indices[n]];
    }
    y[i] = temp;
  }
}

// dense matrix vector multiply
__global__ void mv_dense(const int num_rows, const REAL* matrix, const REAL* x, REAL* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    REAL temp = 0;
    for (int j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != (REAL)0) 
        temp += matrix[i * num_rows + j] * x[j];
    }
    y[i] = temp;
  }
}

long mv_dense_parallel(const int repeat,
                       const int bs,
                       const int num_rows,
                       const REAL* x,
                             REAL* matrix,
                             REAL* y)
{
  REAL *d_x, *d_matrix, *d_y;
  hipMalloc(&d_x, num_rows*sizeof(REAL));
  hipMalloc(&d_matrix, num_rows * num_rows * sizeof(REAL));
  hipMalloc(&d_y, num_rows*sizeof(REAL));

  hipMemcpy(d_x, x, num_rows*sizeof(REAL), hipMemcpyHostToDevice);
  hipMemcpy(d_matrix, matrix, num_rows*num_rows*sizeof(REAL), hipMemcpyHostToDevice);

  dim3 grids ((num_rows + bs - 1) / bs);
  dim3 blocks (bs);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mv_dense<<<grids, blocks>>>(num_rows, d_matrix, d_x, d_y);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  hipMemcpy(y, d_y, num_rows*sizeof(REAL), hipMemcpyDeviceToHost);

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_matrix);

  return time;
}

long mv_csr_parallel(const int repeat,
                     const int bs,
                     const int num_rows,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  size_t *row_indices = (size_t *) malloc((num_rows+1) * sizeof(size_t));
  int *col_indices = (int *) malloc(nnz * sizeof(int));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize csr structure
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);

  size_t *d_row_indices;
  int *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  hipMalloc(&d_row_indices, (num_rows+1)*sizeof(size_t));
  hipMalloc(&d_col_indices, nnz*sizeof(int));
  hipMalloc(&d_values, nnz*sizeof(REAL));
  hipMalloc(&d_x, num_rows*sizeof(REAL));
  hipMalloc(&d_y, num_rows*sizeof(REAL));

  hipMemcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(d_col_indices, col_indices, nnz*sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_values, values, nnz*sizeof(REAL), hipMemcpyHostToDevice);
  hipMemcpy(d_x, x, num_rows*sizeof(REAL), hipMemcpyHostToDevice);

  dim3 grids ((num_rows + bs - 1) / bs);
  dim3 blocks (bs);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mv_csr<<<grids, blocks>>>(num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  hipMemcpy(y, d_y, num_rows*sizeof(REAL), hipMemcpyDeviceToHost);

  free(values);
  free(row_indices);
  free(col_indices);

  hipFree(d_row_indices);
  hipFree(d_col_indices);
  hipFree(d_values);
  hipFree(d_x);
  hipFree(d_y);

  return time;
}

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, "HIPSPARSE API failed at line %d with error: %s\n",         \
               __LINE__, #token_); \
        break

#define CHECK_HIPSPARSE(func)                                                \
{                                                                            \
    hipsparseStatus_t status = (func);                                       \
    if (status != HIPSPARSE_STATUS_SUCCESS) {                                \
        printf("HIPSPARSE API failed at line %d with error: %s (%d)\n",      \
               __LINE__, hipsparseGetErrorString(status), status);           \
        return EXIT_FAILURE;                                                 \
    }                                                                        \
}

long spmv_csr(const int repeat,
              const int num_rows,
              const REAL* x,
              const size_t nnz,
              REAL* matrix,
              REAL* y)
{
  size_t *row_indices = (size_t *) malloc((num_rows+1) * sizeof(size_t));
  int *col_indices = (int *) malloc(nnz * sizeof(int));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize csr structure
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);

  size_t *d_row_indices;
  int *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  hipMalloc(&d_row_indices, (num_rows+1)*sizeof(size_t));
  hipMalloc(&d_col_indices, nnz*sizeof(int));

  hipMalloc(&d_values, nnz*sizeof(REAL));
  hipMalloc(&d_x, num_rows*sizeof(REAL));
  hipMalloc(&d_y, num_rows*sizeof(REAL));

  hipMemcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(d_col_indices, col_indices, nnz*sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_values, values, nnz*sizeof(REAL), hipMemcpyHostToDevice);
  hipMemcpy(d_x, x, num_rows*sizeof(REAL), hipMemcpyHostToDevice);

  hipsparseHandle_t     handle = NULL;
  hipsparseSpMatDescr_t matA;
  hipsparseDnVecDescr_t vecX, vecY;
  const hipDataType DT = (sizeof(REAL) == 4) ? HIP_R_32F : HIP_R_64F;
 
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  REAL                 alpha      = 1.f;
  REAL                 beta       = 0.f;
  CHECK_HIPSPARSE( hipsparseCreate(&handle) )

  // Create sparse matrix A in CSR format
  CHECK_HIPSPARSE( hipsparseCreateCsr(&matA, num_rows, num_rows, nnz,
                                    d_row_indices, d_col_indices, d_values,
                                    HIPSPARSE_INDEX_64I, HIPSPARSE_INDEX_32I,
                                    HIPSPARSE_INDEX_BASE_ZERO, DT) )
  // Create dense vector X
  CHECK_HIPSPARSE( hipsparseCreateDnVec(&vecX, num_rows, d_x, DT) )
  // Create dense vector y
  CHECK_HIPSPARSE( hipsparseCreateDnVec(&vecY, num_rows, d_y, DT) )
  // allocate an external buffer if needed
  CHECK_HIPSPARSE( hipsparseSpMV_bufferSize(
                               handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY, DT,
                               HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
  hipMalloc(&dBuffer, bufferSize);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpMV
    CHECK_HIPSPARSE( hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, DT,
                                 HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer) )
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroySpMat(matA) )
  CHECK_HIPSPARSE( hipsparseDestroyDnVec(vecX) )
  CHECK_HIPSPARSE( hipsparseDestroyDnVec(vecY) )
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )

  hipMemcpy(y, d_y, num_rows*sizeof(REAL), hipMemcpyDeviceToHost);

  free(values);
  free(row_indices);
  free(col_indices);

  hipFree(d_row_indices);
  hipFree(d_col_indices);
  hipFree(d_values);
  hipFree(d_x);
  hipFree(d_y);

  return time;
}
