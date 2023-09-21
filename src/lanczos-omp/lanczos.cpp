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
#pragma omp declare target
template <typename T>
void multiply_inplace_kernel(const int n, T* x, const T k) {
  #pragma omp target teams distribute parallel for thread_limit(THREADS_PER_BLOCK)
  for (int i = 0; i < n; i++) x[i] *= k;
}
#pragma omp end declare target


/**
 * @brief   Cuda kernel function for vector saxpy in place(y += a * x).
 *
 * @param   N   The vector size.
 * @param   y   The output vector.
 * @param   x   The input vector.
 * @param   a   The value to multiply.
 */
#pragma omp declare target
template <typename T>
void saxpy_inplace_kernel(const int n, T* y, const T *x, const T a) {
  #pragma omp target teams distribute parallel for thread_limit(THREADS_PER_BLOCK)
  for (int i = 0; i < n; i++) y[i] += a * x[i];
}
#pragma omp end declare target


#pragma omp declare target
template <typename T>
T device_dot_product(const int n, const T *x, const T *y) {
  T result = 0;
  const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma omp target teams distribute parallel for reduction(+:result) \
num_teams(blocks) thread_limit(THREADS_PER_BLOCK)
  for (int index = 0; index < n; index++) 
    result += x[index] * y[index];

  return result;
}
#pragma omp end declare target


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
symm_tridiag_matrix<T> gpu_lanczos(const csr_matrix<T> &m,
    const vector<T> &v, const int steps) {
  symm_tridiag_matrix<T> result(steps + 1);

  int rows = m.row_size();
  int cols = m.col_size();
  int nonzeros = m.nonzeros();
  assert(rows == cols);
  assert(cols == v.size());

  // compiler errors when const is not added
  const int *row_ptr = m.row_ptr_data();
  const int *col_ind = m.col_ind_data();
  const T *values = m.values_data();
  // remove const due to swap()
  T *x = const_cast<T*>(v.data());

  T* y = (T*) malloc (sizeof(T)*cols);
  T* x_prev = (T*) malloc (sizeof(T)*cols);

  double start_time, end_time;
  const int row_nonzeros = nonzeros / rows;
  int group_size = row_nonzeros > 16 ? 32 : 16;
  group_size = row_nonzeros > 8 ? group_size : 8;
  group_size = row_nonzeros > 4 ? group_size : 4;
  group_size = row_nonzeros > 2 ? group_size : 2;

  
#pragma omp target data map (to: row_ptr[0:rows+1], \
                                 col_ind[0:nonzeros], \
                                 values[0:nonzeros], \
                                 x[0:cols]) \
                        map (alloc: y[0:cols], x_prev[0:cols])
{
  start_time = cycle_timer::current_seconds();
  for (int i = 0; i < steps; i++) {
    // y_i = M*x_i
    const int groups_per_block = THREADS_PER_BLOCK / group_size;
    const int multiply_blocks = (rows + groups_per_block - 1) / groups_per_block;
    #pragma omp target teams num_teams(multiply_blocks) thread_limit(THREADS_PER_BLOCK)
    {
      T result[THREADS_PER_BLOCK];
      #pragma omp parallel 
      {
        int lid = omp_get_thread_num();
        int index = omp_get_team_num()*omp_get_num_threads() + lid;
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
      }
    }

    // alpha_i <- y_i*x_i
    T product = device_dot_product(rows, x, y);

    // the values of alpha and beta are saved in the 'result' array
    result.alpha(i) = product;

    // y_i <- y_i - alpha_i*x_i - beta_i*x_(i-1)
    saxpy_inplace_kernel<T>(rows, y, x, -product);

    if (i > 0) {
      saxpy_inplace_kernel<T>(rows, y, x_prev, -result.beta(i - 1));
    }

    std::swap(x, x_prev);

    // beta_(i+1) <- ||y_i||
    result.beta(i) = T(std::sqrt(device_dot_product(rows, y, y)));

    // x_(i+1) <- y_i / beta_(i+1)
    multiply_inplace_kernel<T>(rows, y, 1 / result.beta(i));

    std::swap(x, y);
  }
  end_time = cycle_timer::current_seconds();
}
  std::cout << "GPU Lanczos iterations: " << steps << std::endl;
  std::cout << "GPU Lanczos time: " << end_time - start_time << " sec" << std::endl;

  // TODO memory leak or double free
  // free(y);
  // free(x_prev);

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
vector<T> gpu_lanczos_eigen(const csr_matrix<T> &matrix, int k, int steps) {
  int cols = matrix.col_size();
  assert(cols > 0);
  vector<T> v(cols, 0);
  v[0] = 1;
  symm_tridiag_matrix<T> tridiag = gpu_lanczos(matrix, v, steps);
  return lanczos_no_spurious(tridiag, k);
}


template vector<float> gpu_lanczos_eigen(const csr_matrix<float> &matrix, int k, int steps);
template vector<double> gpu_lanczos_eigen(const csr_matrix<double> &matrix, int k, int steps);
