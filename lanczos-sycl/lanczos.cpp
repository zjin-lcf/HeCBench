#include <cassert>
#include <iostream>
#include <unistd.h>
#include <cmath>
#include <utility>

#include "eigen.h"
#include "matrix.h"
#include "lanczos.h"
#include "cycle_timer.h"

#define THREADS_PER_BLOCK 256

// Forward declarations
template <typename T>
class k_multiply_inplace;

template <typename T>
class k_saxpy;

template <typename T>
class k_multiply;

template <typename T>
class k_dot_product;

/**
 * @brief   Cuda kernel function for vector multiply in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to multiply.
 */
template <typename T>
void multiply_inplace_kernel(sycl::queue &q, const int n, T *x, const T k) {

  const int threads = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
  sycl::range<1> gws (threads);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class k_multiply_inplace<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int index = item.get_global_id(0);
      if (index < n) x[index] *= k;
    });
  });
}


/**
 * @brief   Cuda kernel function for vector saxpy in place(y += a * x).
 *
 * @param   N   The vector size.
 * @param   y   The output vector.
 * @param   x   The input vector.
 * @param   a   The value to multiply.
 */
template <typename T>
void saxpy_inplace_kernel(sycl::queue &q, const int n, T *y, T *x, const T a) {

  const int threads = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
  sycl::range<1> gws (threads);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class k_saxpy<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int index = item.get_global_id(0);
      if (index < n) y[index] += a * x[index];
    });
  });
}

/**
 * @brief   Cuda kernel function for warp sparse matrix multiplication.
 *
 * @param   group_size  The number of threads used to calculate one row.
 * @param   rows        The row number of the matrix.
 * @param   begin_row   The row to begin from in this kernel launch.
 * @param   row_ptr     Row pointers in the CSR matrix.
 * @param   col_ind     Column indexes in the CSR matrix.
 * @param   values      Data values in the CSR matrix.
 * @param   x           The input vector x to multiply.
 * @param   y           The output vector y.
 */
template <typename T>
void warp_multiply_kernel(sycl::queue &q, const int group_size, const int rows,
                          int *row_ptr, int *col_ind,
                          T *values, T *x, T *y) {

  const int groups_per_block = THREADS_PER_BLOCK / group_size;
  const int multiply_blocks = (rows + groups_per_block - 1) / groups_per_block;
  sycl::range<1> gws (multiply_blocks*THREADS_PER_BLOCK);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<T, 1> result(sycl::range<1>(THREADS_PER_BLOCK), cgh);
    cgh.parallel_for<class k_multiply<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int index = item.get_global_id(0);
      int lid = item.get_local_id(0);
      int r = index / group_size;
      int lane = index % group_size;

      result[lid] = 0;
      if (r < rows) {
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int i = start + lane; i < end; i+= group_size)
          result[lid] += values[i] * x[col_ind[i]];

        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
        while (half > 0) {
          if (lane < half) result[lid] += result[lid + half];
          half /= 2;
        }
        if (lane == 0) y[r] = result[lid];
      }
    });
  });
}

template <typename T>
T device_dot_product(sycl::queue &q, const int n, T *x, T *y, T *z) {

  const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const int threads = blocks * THREADS_PER_BLOCK;
  sycl::range<1> gws (threads);
  sycl::range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<T, 1> result (sycl::range<1>(THREADS_PER_BLOCK), cgh);
    cgh.parallel_for<class k_dot_product<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int index = item.get_global_id(0);
      int lid = item.get_local_id(0);
      int bid = item.get_group(0);

      if (index < n) 
        result[lid] = x[index] * y[index];
      else 
        result[lid] = 0;

      item.barrier(sycl::access::fence_space::local_space);

      int half = THREADS_PER_BLOCK / 2;
      while (half > 0) {
        if (lid < half) 
          result[lid] += result[lid + half];
        item.barrier(sycl::access::fence_space::local_space);
        half /= 2;
      }
      if (lid == 0) z[bid] = result[0];
    });
  });

  // Transfer result back from device to host
  T host_scratch[blocks];

  q.memcpy(host_scratch, z, sizeof(T) * blocks);

  T result(0);

  q.wait();

  for (int i = 0; i < blocks; i++) {
    result += host_scratch[i];
  }
  return result;
}


/**
 * @brief   Caller function for naive Lanczos algorithm in CUDA.
 *
 * @param   m       The matrix to do operations on.
 * @param   v       The initial vector with norm 1.
 * @param   steps   The iteration times for lanczos algorithm.
 *
 * @return  The tridiagonal matrix result of lanczos algorithm.
 */
template <typename T>
symm_tridiag_matrix<T> gpu_lanczos(sycl::queue &q, const csr_matrix<T> &m,
    const vector<T> &v, const int steps) {
  symm_tridiag_matrix<T> result(steps + 1);

  int rows = m.row_size();
  int cols = m.col_size();
  int nonzeros = m.nonzeros();
  const int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  assert(rows == cols);
  assert(cols == v.size());

  // Malloc device space

  T *x_prev = sycl::malloc_device<T>(cols, q);

  T *y = sycl::malloc_device<T>(cols, q);

  T *scratch = sycl::malloc_device<T>(blocks, q);

  int *row_ptr = sycl::malloc_device<int>(rows+1, q); 
  q.memcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows+1));

  int *col_ind = sycl::malloc_device<int>(nonzeros, q);
  q.memcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros);

  T *values = sycl::malloc_device<T>(nonzeros, q);
  q.memcpy(values, m.values_data(), sizeof(T) * nonzeros);

  T *x = sycl::malloc_device<T>(cols, q);
  q.memcpy(x, v.data(), sizeof(T) * cols);

  const int row_nonzeros = nonzeros / rows;
  int group_size = row_nonzeros > 16 ? 32 : 16;
  group_size = row_nonzeros > 8 ? group_size : 8;
  group_size = row_nonzeros > 4 ? group_size : 4;
  group_size = row_nonzeros > 2 ? group_size : 2;

  // Run kernel and the values of alpha and beta are saved in the 'result' array
  double start_time = cycle_timer::current_seconds();
  for (int i = 0; i < steps; i++) {
    // y_i = M*x_i
    warp_multiply_kernel<T>(q, group_size,
        rows, row_ptr, col_ind, values, x, y);

    // alpha_i <- y_i*x_i
    T product = device_dot_product(q, rows, x, y, scratch);

    result.alpha(i) = product;

    // y_i <- y_i - alpha_i*x_i - beta_i*x_(i-1)
    saxpy_inplace_kernel<T>(q, rows, y, x, -product);

    if (i > 0) {
      saxpy_inplace_kernel<T>(q, rows, y, x_prev, -result.beta(i - 1));
    }

    std::swap(x, x_prev);

    // beta_(i+1) <- ||y_i||
    result.beta(i) = T(std::sqrt(device_dot_product(q, rows, y, y, scratch)));

    // x_(i+1) <- y_i / beta_(i+1)
    multiply_inplace_kernel<T>(q, rows, y, 1 / result.beta(i));

    std::swap(x, y);
  }
  q.wait();
  double end_time = cycle_timer::current_seconds();
  std::cout << "GPU Lanczos iterations: " << steps << std::endl;
  std::cout << "GPU Lanczos time: " << end_time - start_time << " sec" << std::endl;

  result.resize(steps);
  sycl::free(x_prev, q);
  sycl::free(y, q);
  sycl::free(scratch, q);
  sycl::free(row_ptr, q);
  sycl::free(col_ind, q);

  return result;
}

/**
 * @brief   Lanczos algorithm for eigendecomposition in CUDA.
 * 
 * @param   matrix  CSR matrix to decompose
 * @param   k       number of largest eigenvalues to compute
 * @param   steps   maximum steps for the iteration
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> gpu_lanczos_eigen(sycl::queue &q, const csr_matrix<T> &matrix, int k, int steps) {
  int cols = matrix.col_size();
  assert(cols > 0);
  vector<T> v(cols, 0);
  v[0] = 1;
  symm_tridiag_matrix<T> tridiag = gpu_lanczos(q, matrix, v, steps);
  return lanczos_no_spurious(tridiag, k);
}


template vector<float> gpu_lanczos_eigen(sycl::queue &q, const csr_matrix<float> &matrix, int k, int steps);
template vector<double> gpu_lanczos_eigen(sycl::queue &q, const csr_matrix<double> &matrix, int k, int steps);
