#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "mv.h"

// dense matrix vector multiply
void mv_dense(sycl::nd_item<1> &item,
              const size_t num_rows,
              const REAL* matrix,
              const REAL* x, REAL* y)
{
  size_t i = item.get_global_id(0);
  if (i < num_rows) {
    REAL temp = 0;
    for (size_t j = 0; j < num_rows; j++) {
      if (matrix[i * num_rows + j] != (REAL)0)
        temp += matrix[i * num_rows + j] * x[j];
    }
    y[i] = temp;
  }
}

long reference(const int repeat,
               const size_t num_rows,
               const REAL* x,
               REAL* matrix,
               REAL* y)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t num_elems = num_rows * num_rows;

  REAL *d_x, *d_matrix, *d_y;
  d_x = sycl::malloc_device<REAL>(num_rows, q);
  d_matrix = sycl::malloc_device<REAL>(num_elems, q);
  d_y = sycl::malloc_device<REAL>(num_rows, q);

  q.memcpy(d_x, x, num_rows*sizeof(REAL));
  q.memcpy(d_matrix, matrix, num_elems*sizeof(REAL));

  sycl::range<1> gws ((num_rows + 256 - 1) / 256 * 256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class dmvm>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        mv_dense(item, num_rows, d_matrix, d_x, d_y);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  sycl::free(d_x, q);
  sycl::free(d_y, q);
  sycl::free(d_matrix, q);

  return time;
}

long spmv_csr(const int repeat,
              const size_t num_rows,
              const REAL* x,
              const size_t nnz,
              REAL* matrix,
              REAL* y)
{
  size_t *row_indices = (size_t *) malloc((num_rows+1) * sizeof(size_t));
  size_t *col_indices = (size_t *) malloc(nnz * sizeof(size_t));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize csr structure
  init_csr((size_t*)row_indices, values, (size_t*)col_indices, matrix, num_rows, nnz);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t *d_row_indices = sycl::malloc_device<size_t>(num_rows+1, q);
  size_t *d_col_indices = sycl::malloc_device<size_t>(nnz, q);
  REAL *d_values = sycl::malloc_device<REAL>(nnz, q);
  REAL *d_x = sycl::malloc_device<REAL>(num_rows, q);
  REAL *d_y = sycl::malloc_device<REAL>(num_rows, q);

  q.memcpy(d_row_indices, row_indices, (num_rows+1)*sizeof(size_t));
  q.memcpy(d_col_indices, col_indices, nnz*sizeof(size_t));
  q.memcpy(d_values, values, nnz*sizeof(REAL));
  q.memcpy(d_x, x, num_rows*sizeof(REAL));

  // create and initialize handle for a Sparse Matrix in CSR format
  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  REAL alpha = 1.f;
  REAL beta = 0.f;

  // fills the internal state of the matrix handle with the user provided arrays in CSR format.
  oneapi::mkl::sparse::set_csr_data(q, handle, num_rows, num_rows, 
                                    oneapi::mkl::index_base::zero,
                                    (int64_t*)d_row_indices, (int64_t*)d_col_indices, d_values);

  // analyzes matrix structure and performs optimizations.
  // optimized data is then stored in the matrix handle
  optimize_gemv(q, oneapi::mkl::transpose::nontrans, handle).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, alpha,
                              handle, d_x, beta, d_y);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &handle).wait();

  free(values);
  free(row_indices);
  free(col_indices);

  sycl::free(d_row_indices, q);
  sycl::free(d_col_indices, q);
  sycl::free(d_values, q);
  sycl::free(d_x, q);
  sycl::free(d_y, q);

  return time;
}

long spmv_coo(const int repeat,
              const size_t num_rows,
              const REAL* x,
              const size_t nnz,
              REAL* matrix,
              REAL* y)
{
  size_t *row_indices = (size_t *) malloc(nnz * sizeof(size_t));
  size_t *col_indices = (size_t *) malloc(nnz * sizeof(size_t));
  REAL *values = (REAL *) malloc(nnz * sizeof(REAL));

  // initialize coo structure
  init_coo(row_indices, values, col_indices, matrix, num_rows, nnz);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t *d_row_indices = sycl::malloc_device<size_t>(nnz, q);
  size_t *d_col_indices = sycl::malloc_device<size_t>(nnz, q);
  REAL *d_values = sycl::malloc_device<REAL>(nnz, q);
  REAL *d_x = sycl::malloc_device<REAL>(num_rows, q);
  REAL *d_y = sycl::malloc_device<REAL>(num_rows, q);

  q.memcpy(d_row_indices, row_indices, nnz*sizeof(size_t));
  q.memcpy(d_col_indices, col_indices, nnz*sizeof(size_t));
  q.memcpy(d_values, values, nnz*sizeof(REAL));
  q.memcpy(d_x, x, num_rows*sizeof(REAL));

  // create and initialize handle for a Sparse Matrix in CSR format
  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  REAL alpha = 1.f;
  REAL beta = 0.f;

  // fills the internal state of the matrix handle with the user provided arrays in COO format.
  oneapi::mkl::sparse::set_coo_data(q, handle, num_rows, num_rows, nnz,
                                    oneapi::mkl::index_base::zero,
                                    (int64_t*)d_row_indices, (int64_t*)d_col_indices, d_values);

  // analyzes matrix structure and performs optimizations.
  // optimized data is then stored in the matrix handle
  optimize_gemv(q, oneapi::mkl::transpose::nontrans, handle).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, alpha,
                              handle, d_x, beta, d_y);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &handle).wait();

  free(values);
  free(row_indices);
  free(col_indices);

  sycl::free(d_row_indices, q);
  sycl::free(d_col_indices, q);
  sycl::free(d_values, q);
  sycl::free(d_x, q);
  sycl::free(d_y, q);

  return time;
}
