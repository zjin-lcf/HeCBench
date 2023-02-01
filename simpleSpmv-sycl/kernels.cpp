#include <stdlib.h>
#include <chrono>
#include "common.h"
#include "mv.h"

// sparse matrix vector multiply using the CSR format
void mv_csr(nd_item<1> &item,
            const int num_rows,
            const size_t *row_indices,
            const int *col_indices,
            const REAL *values,
            const REAL *x,
                  REAL *y)
{
  int i = item.get_global_id(0);
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
void mv_dense(nd_item<1> &item,
              const int num_rows,
              const REAL* matrix,
              const REAL* x, REAL* y)
{
  int i = item.get_global_id(0);
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
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  size_t num_elems = (size_t)num_rows * num_rows; 

  REAL *d_x, *d_matrix, *d_y;
  d_x = malloc_device<REAL>(num_rows, q);
  d_matrix = malloc_device<REAL>(num_elems, q);
  d_y = malloc_device<REAL>(num_rows, q);

  q.memcpy(d_x, x, num_rows*sizeof(REAL));
  q.memcpy(d_matrix, matrix, num_elems*sizeof(REAL));

  range<1> gws ((num_rows + bs - 1) / bs * bs);
  range<1> lws (bs);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class dmvm>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        mv_dense(item, num_rows, d_matrix, d_x, d_y);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  free(d_x, q);
  free(d_y, q);
  free(d_matrix, q);

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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  size_t *d_row_indices = malloc_device<size_t>(num_rows+1, q);
  int *d_col_indices = malloc_device<int>(nnz, q);
  REAL *d_values = malloc_device<REAL>(nnz, q);
  REAL *d_x = malloc_device<REAL>(num_rows, q);
  REAL *d_y = malloc_device<REAL>(num_rows, q);

  q.memcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t));
  q.memcpy(d_col_indices, col_indices, nnz*sizeof(int));
  q.memcpy(d_values, values, nnz*sizeof(REAL));
  q.memcpy(d_x, x, num_rows*sizeof(REAL));

  range<1> gws ((num_rows + bs - 1) / bs * bs);
  range<1> lws (bs);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class spmv>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        mv_csr(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  free(values);
  free(row_indices);
  free(col_indices);

  free(d_row_indices, q);
  free(d_col_indices, q);
  free(d_values, q);
  free(d_x, q);
  free(d_y, q);

  return time;
}
