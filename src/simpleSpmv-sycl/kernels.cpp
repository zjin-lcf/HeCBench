#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "mv.h"

// sparse matrix vector multiply using the CSR format
void mv_csr(sycl::nd_item<1> &item,
            const size_t num_rows,
            const size_t *row_indices,
            const size_t *col_indices,
            const REAL *values,
            const REAL *x,
                  REAL *y)
{
  size_t i = item.get_global_id(0);
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

// Reference (SYCLomatic) 
template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index = (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = sycl::shift_group_left(g, x, delta);
  if ((id + delta) >= end_index) result = x;
  return result;
}

template <int BS>
void vector_mv_csr(sycl::nd_item<2> &item,
                   const size_t num_rows,
                   const size_t *row_indices,
                   const size_t *col_indices,
                   const REAL *values,
                   const REAL *x,
                   REAL *y)
{
  size_t m = item.get_group(1) * item.get_local_range(0) + item.get_local_id(0);
  if (m < num_rows) {
    size_t row_start = row_indices[m];
    size_t row_end = row_indices[m+1];

    REAL temp = 0;
    for (size_t n = row_start + item.get_local_id(1); n < row_end;
         n += BS) {
      temp += values[n] * x[col_indices[n]];
    }
    #pragma unroll
    for (int i = BS >> 1; i > 0; i >>= 1)
      temp += shift_sub_group_left(item.get_sub_group(), temp, i, BS);

    y[m] = temp;
  }
}


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

long mv_dense_parallel(const int repeat,
                       const int bs,
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

  sycl::range<1> gws ((num_rows + bs - 1) / bs * bs);
  sycl::range<1> lws (bs);

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

  sycl::range<1> gws ((num_rows + bs - 1) / bs * bs);
  sycl::range<1> lws (bs);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class spmv>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        mv_csr(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  sycl::free(d_row_indices, q);
  sycl::free(d_col_indices, q);
  sycl::free(d_values, q);
  sycl::free(d_x, q);
  sycl::free(d_y, q);

  return time;
}

// Reference
// https://github.com/ROCm/rocm-blogs/blob/release/blogs/high-performance-computing/spmv/part-1/examples/vector_csr.cpp
size_t prevPowerOf2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}

long vector_mv_csr_parallel(const int repeat,
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

  int nnz_per_row = nnz / num_rows;
  int threads_per_row = prevPowerOf2(nnz_per_row);
  const int warpSize = 32;
  threads_per_row = threads_per_row > warpSize ? warpSize : threads_per_row;
  int rows_per_block = bs / threads_per_row;
  if (rows_per_block == 0) rows_per_block = 1;
  int num_blocks = (num_rows + rows_per_block - 1) / rows_per_block;

  sycl::range<2> gws (rows_per_block, num_blocks * threads_per_row);
  sycl::range<2> lws (rows_per_block, threads_per_row);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    if (threads_per_row <= 2)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
           [[sycl::reqd_sub_group_size(warpSize)]] {
           vector_mv_csr<2>(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
        });
      });
    else if (threads_per_row <= 4)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
           [[sycl::reqd_sub_group_size(warpSize)]] {
           vector_mv_csr<4>(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
        });
      });
    else if (threads_per_row <= 8)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
           [[sycl::reqd_sub_group_size(warpSize)]] {
           vector_mv_csr<8>(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
        });
      });
    else if (threads_per_row <= 16)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
           [[sycl::reqd_sub_group_size(warpSize)]] {
           vector_mv_csr<16>(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
        });
      });
    else if (threads_per_row <= 32)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
           [[sycl::reqd_sub_group_size(warpSize)]] {
           vector_mv_csr<32>(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
        });
      });
    else
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
           [[sycl::reqd_sub_group_size(warpSize)]] {
           vector_mv_csr<64>(item, num_rows, d_row_indices, d_col_indices, d_values, d_x, d_y);
        });
      });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(y, d_y, num_rows*sizeof(REAL)).wait();

  sycl::free(d_row_indices, q);
  sycl::free(d_col_indices, q);
  sycl::free(d_values, q);
  sycl::free(d_x, q);
  sycl::free(d_y, q);

  return time;
}
