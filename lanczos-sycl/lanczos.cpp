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


/**
 * @brief   Cuda kernel function for vector multiply in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to multiply.
 */
template <typename T>
void multiply_inplace_kernel(queue &q, const int n, buffer<T, 1> &device_x, const T k) {

  const int threads = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
  range<1> gws (threads);
  range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (handler &cgh) {
      auto x = device_x.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class multiply_inplace>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
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
void saxpy_inplace_kernel(queue &q, const int n, buffer<T,1> &device_y, buffer<T, 1> &device_x,
    const T a) {

  const int threads = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
  range<1> gws (threads);
  range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (handler &cgh) {
      auto x = device_x.template get_access<sycl_read>(cgh);
      auto y = device_y.template get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class saxpy>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
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
void warp_multiply_kernel(queue &q, const int group_size, const int rows,
    buffer<int, 1> &row_ptr, buffer<int, 1> &col_ind,
    buffer<T, 1> &values, buffer<T, 1> &x, buffer<T, 1> &y) {

  const int groups_per_block = THREADS_PER_BLOCK / group_size;
  const int multiply_blocks = (rows + groups_per_block - 1) / groups_per_block;
  range<1> gws (multiply_blocks*THREADS_PER_BLOCK);
  range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (handler &cgh) {
    auto row_ptr_acc = row_ptr.get_access<sycl_read>(cgh);
    auto col_ind_acc = col_ind.get_access<sycl_read>(cgh);
    auto values_acc = values.template get_access<sycl_read>(cgh);
    auto x_acc = x.template get_access<sycl_read>(cgh);
    auto y_acc = y.template get_access<sycl_write>(cgh);

    accessor<T, 1, sycl_read_write, access::target::local> result(THREADS_PER_BLOCK, cgh);
    cgh.parallel_for<class multiply>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int index = item.get_global_id(0);
      int lid = item.get_local_id(0);
      int r = index / group_size;
      int lane = index % group_size;

      result[lid] = 0;
      if (r < rows) {
        int start = row_ptr_acc[r];
        int end = row_ptr_acc[r + 1];
        for (int i = start + lane; i < end; i+= group_size)
          result[lid] += values_acc[i] * x_acc[col_ind_acc[i]];

        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
        while (half > 0) {
          if (lane < half) result[lid] += result[lid + half];
          half /= 2;
        }
        if (lane == 0) y_acc[r] = result[lid];
      }
    });
  });
}

template <typename T>
T device_dot_product(queue &q, const int n, buffer<T, 1> &device_x, buffer<T, 1> &device_y, buffer<T, 1> &device_scratch) {

  const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const int threads = blocks * THREADS_PER_BLOCK;
  range<1> gws (threads);
  range<1> lws (THREADS_PER_BLOCK);

  q.submit([&] (handler &cgh) {
    auto x = device_x.template get_access<sycl_read>(cgh);
    auto y = device_y.template get_access<sycl_read>(cgh);
    auto z = device_scratch.template get_access<sycl_discard_write>(cgh);
    accessor<T, 1, sycl_read_write, access::target::local> result(THREADS_PER_BLOCK, cgh);
    cgh.parallel_for<class dot_product>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int index = item.get_global_id(0);
      int lid = item.get_local_id(0);
      int bid = item.get_group(0);

      if (index < n) 
        result[lid] = x[index] * y[index];
      else 
        result[lid] = 0;

      item.barrier(access::fence_space::local_space);

      int half = THREADS_PER_BLOCK / 2;
      while (half > 0) {
        if (lid < half) 
          result[lid] += result[lid + half];
        item.barrier(access::fence_space::local_space);
        half /= 2;
      }
      if (lid == 0) z[bid] = result[0];
    });
  });

  // Transfer result back from device to host
  T host_scratch[blocks];
  T result(0);

  q.submit([&] (handler &cgh) {
    auto z = device_scratch.template get_access<sycl_read>(cgh);
    cgh.copy(z, host_scratch);
  });
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
symm_tridiag_matrix<T> gpu_lanczos(queue &q, const csr_matrix<T> &m,
    const vector<T> &v, const int steps) {
  symm_tridiag_matrix<T> result(steps + 1);

  int rows = m.row_size();
  int cols = m.col_size();
  int nonzeros = m.nonzeros();
  const int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  assert(rows == cols);
  assert(cols == v.size());

  // Malloc device space

  buffer<T, 1> x_prev (cols);
  buffer<T, 1> y (cols);
  buffer<T, 1> scratch (blocks);

  buffer<int, 1> row_ptr (m.row_ptr_data(), rows+1);
  buffer<int, 1> col_ind (m.col_ind_data(), nonzeros);
  buffer<T, 1> values (m.values_data(), nonzeros);
  buffer<T, 1> x (v.data(), cols);

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
vector<T> gpu_lanczos_eigen(queue &q, const csr_matrix<T> &matrix, int k, int steps) {
  int cols = matrix.col_size();
  assert(cols > 0);
  vector<T> v(cols, 0);
  v[0] = 1;
  symm_tridiag_matrix<T> tridiag = gpu_lanczos(q, matrix, v, steps);
  return lanczos_no_spurious(tridiag, k);
}


template vector<float> gpu_lanczos_eigen(queue &q, const csr_matrix<float> &matrix, int k, int steps);
template vector<double> gpu_lanczos_eigen(queue &q, const csr_matrix<double> &matrix, int k, int steps);
