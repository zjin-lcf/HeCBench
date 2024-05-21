#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "mv.h"

// sparse matrix vector multiply using the CSR format
__global__ void mv_csr(const size_t num_rows,
                       const size_t *row_indices,
                       const size_t *col_indices,
                       const REAL *values,
                       const REAL *x,
                             REAL *y)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void mv_dense(const size_t num_rows, const REAL* matrix, const REAL* x, REAL* y)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_rows) {
    REAL temp = 0;
    for (size_t j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != (REAL)0) 
        temp += matrix[i * num_rows + j] * x[j];
    }
    y[i] = temp;
  }
}

long mv_dense_parallel(const int repeat,
                       const int bs,
                       const size_t num_rows,
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
                     const size_t num_rows,
                     const size_t *row_indices,
                     const size_t *col_indices,
                     const REAL* values,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  size_t *d_row_indices;
  size_t *d_col_indices;
  REAL *d_values, *d_x, *d_y;

  hipMalloc(&d_row_indices, (num_rows+1)*sizeof(size_t));
  hipMalloc(&d_col_indices, nnz*sizeof(size_t));
  hipMalloc(&d_values, nnz*sizeof(REAL));
  hipMalloc(&d_x, num_rows*sizeof(REAL));
  hipMalloc(&d_y, num_rows*sizeof(REAL));

  hipMemcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t), hipMemcpyHostToDevice);
  hipMemcpy(d_col_indices, col_indices, nnz*sizeof(size_t), hipMemcpyHostToDevice);
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

  hipFree(d_row_indices);
  hipFree(d_col_indices);
  hipFree(d_values);
  hipFree(d_x);
  hipFree(d_y);

  return time;
}
