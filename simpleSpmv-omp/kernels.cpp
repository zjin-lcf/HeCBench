#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "mv.h"

// dense matrix vector multiply
long mv_dense_parallel(const int repeat,
                       const int bs,
                       const int num_rows,
                       const REAL* x,
                             REAL* matrix,
                             REAL* y)
{
  long time;

  #pragma omp target data map(to: matrix[0:num_rows*num_rows], \
                                  x[0:num_rows]) \
                          map(from: y[0:num_rows])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for num_threads(bs)
      for (int i = 0; i < num_rows; i++) {
        REAL temp = 0;
        for (int j = 0; j < num_rows; j++) {
          if (matrix[i * num_rows + j] != (REAL)0) 
            temp += matrix[i * num_rows + j] * x[j];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}

// sparse matrix vector multiply using the CSR format
long mv_csr_parallel(const int repeat,
                     const int bs,
                     const int num_rows,
                     const REAL* x,
                     const int nnz,
                     REAL* matrix,
                     REAL* y)
{
  int *row_indices = (int *) malloc((num_rows+1) * sizeof(int));
  int *col_indices = (int *) malloc(nnz * sizeof(int));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize csr structure
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);

  long time;

  #pragma omp target data map(to: row_indices[0:num_rows+1], \
                                  col_indices[0:nnz], \
                                  values[0:nnz], \
                                  x[0:num_rows]) \
                          map(from: y[0:num_rows])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for num_threads(bs)
      for (int i = 0; i < num_rows; i++) {
        int row_start = row_indices[i];
        int row_end = row_indices[i+1];

        REAL temp = 0;
        for(int j = row_start; j < row_end; j++){
          temp += values[j] * x[col_indices[j]];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  free(values);
  free(row_indices);
  free(col_indices);

  return time;
}
