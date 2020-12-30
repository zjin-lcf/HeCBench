#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

using std::cout;
using std::endl;
using std::vector;

/**
 * @brief   Cuda kernel function for vector dot product.
 *
 * @param   N   The vector size.
 * @param   x   The first input vector.
 * @param   y   The second input vector.
 * @param   z   The temp sum per block.
 */
template <typename T>
void dot_product_kernel(const int N, const T *x, const T *y,
    T *z, sycl::nd_item<3> item_ct1, T *result) {

    int index = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);

    if (index < N) {
        result[item_ct1.get_local_id(2)] = x[index] * y[index];
    } else {
        result[item_ct1.get_local_id(2)] = 0;
    }
    item_ct1.barrier();

    int half = THREADS_PER_BLOCK / 2;
    while (half > 0) {
        if (item_ct1.get_local_id(2) < half) {
            result[item_ct1.get_local_id(2)] +=
                result[item_ct1.get_local_id(2) + half];
        }
        item_ct1.barrier();
        half /= 2;
    }

    if (item_ct1.get_local_id(2) == 0) {
        z[item_ct1.get_group(2)] = result[0];
    }
}

/**
 * @brief   Cuda kernel function for vector multiply in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to multiply.
 */
template <typename T>
void multiply_inplace_kernel(const int N, T *x, const T k,
                             sycl::nd_item<3> item_ct1) {

    int index = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);

    if (index < N) {
        x[index] = x[index] * k;
    }
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
void saxpy_inplace_kernel(const int N, T *y, const T *x,
    const T a, sycl::nd_item<3> item_ct1) {

    int index = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);

    if (index < N) {
        y[index] += a * x[index];
    }
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
void warp_multiply_kernel(const int group_size, const int rows,
    const int begin_row, const int *row_ptr, const int *col_ind,
    const T *values, const T *x, T *y, sycl::nd_item<3> item_ct1, T *result) {

    int index = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);
    int r = index / group_size + begin_row;
    int lane = index % group_size;

    result[item_ct1.get_local_id(2)] = 0;
    if (r < rows) {
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int i = start + lane; i < end; i+= group_size) {
            result[item_ct1.get_local_id(2)] += values[i] * x[col_ind[i]];
        }
        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
        while (half > 0) {
            if (lane < half) {
                result[item_ct1.get_local_id(2)] +=
                    result[item_ct1.get_local_id(2) + half];
            }
            half /= 2;
        }
        if (lane == 0) {
            y[r] = result[item_ct1.get_local_id(2)];
        }
    }
}

template <typename T>
T device_dot_product(int n, const T *device_x, const T *device_y, T *device_scratch) {
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Run kernel
    {
        std::pair<dpct::buffer_t, size_t> device_x_buf_ct1 =
            dpct::get_buffer_and_offset(device_x);
        size_t device_x_offset_ct1 = device_x_buf_ct1.second;
        std::pair<dpct::buffer_t, size_t> device_y_buf_ct2 =
            dpct::get_buffer_and_offset(device_y);
        size_t device_y_offset_ct2 = device_y_buf_ct2.second;
        std::pair<dpct::buffer_t, size_t> device_scratch_buf_ct3 =
            dpct::get_buffer_and_offset(device_scratch);
        size_t device_scratch_offset_ct3 = device_scratch_buf_ct3.second;
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<T, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                result_acc_ct1(sycl::range<1>(256 /*THREADS_PER_BLOCK*/), cgh);
            auto device_x_acc_ct1 =
                device_x_buf_ct1.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto device_y_acc_ct2 =
                device_y_buf_ct2.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto device_scratch_acc_ct3 =
                device_scratch_buf_ct3.first
                    .get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                      sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                                  sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                [=](sycl::nd_item<3> item_ct1) {
                    const T *device_x_ct1 =
                        (const T *)(&device_x_acc_ct1[0] + device_x_offset_ct1);
                    const T *device_y_ct2 =
                        (const T *)(&device_y_acc_ct2[0] + device_y_offset_ct2);
                    T *device_scratch_ct3 = (T *)(&device_scratch_acc_ct3[0] +
                                                  device_scratch_offset_ct3);
                    dot_product_kernel<T>(n, device_x_ct1, device_y_ct2,
                                          device_scratch_ct3, item_ct1,
                                          result_acc_ct1.get_pointer());
                });
        });
    }

    // Transfer result back from device to host
    T host_scratch[blocks];
    T result(0);
    dpct::dpct_memcpy(host_scratch, device_scratch, sizeof(T) * blocks,
                      dpct::device_to_host);
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
symm_tridiag_matrix<T> gpu_lanczos(const csr_matrix<T> &m, const vector<T> &v,
                                   const int steps) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    symm_tridiag_matrix<T> result(steps + 1);

    int rows = m.row_size();
    int cols = m.col_size();
    int nonzeros = m.nonzeros();
    const int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    assert(rows == cols);
    assert(cols == v.size());

    // Malloc device space
    int *row_ptr, *col_ind;
    T *values, *x, *x_prev, *y, *scratch;
    dpct::dpct_malloc(&row_ptr, sizeof(int) * (rows + 1));
    dpct::dpct_malloc(&col_ind, sizeof(int) * nonzeros);
    dpct::dpct_malloc(&values, sizeof(T) * nonzeros);
    dpct::dpct_malloc(&x, sizeof(T) * cols);
    dpct::dpct_malloc(&x_prev, sizeof(T) * cols);
    dpct::dpct_malloc(&y, sizeof(T) * cols);
    dpct::dpct_malloc(&scratch, sizeof(T) * blocks);

    // Transfer data from host to device
    dpct::dpct_memcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
                      dpct::host_to_device);
    dpct::dpct_memcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
                      dpct::host_to_device);
    dpct::dpct_memcpy(values, m.values_data(), sizeof(T) * nonzeros,
                      dpct::host_to_device);
    dpct::dpct_memcpy(x, v.data(), sizeof(T) * cols, dpct::host_to_device);

    const int row_nonzeros = nonzeros / rows;
    int group_size = row_nonzeros > 16 ? 32 : 16;
    group_size = row_nonzeros > 8 ? group_size : 8;
    group_size = row_nonzeros > 4 ? group_size : 4;
    group_size = row_nonzeros > 2 ? group_size : 2;
    const int groups_per_block = THREADS_PER_BLOCK / group_size;
    const int multiply_blocks = (rows + groups_per_block - 1) / groups_per_block;
    // Run kernel and the values of alpha and beta are saved in the 'result' array
    double start_time = cycle_timer::current_seconds();
    for (int i = 0; i < steps; i++) {
        // y_i = M*x_i
        {
            std::pair<dpct::buffer_t, size_t> row_ptr_buf_ct3 =
                dpct::get_buffer_and_offset(row_ptr);
            size_t row_ptr_offset_ct3 = row_ptr_buf_ct3.second;
            std::pair<dpct::buffer_t, size_t> col_ind_buf_ct4 =
                dpct::get_buffer_and_offset(col_ind);
            size_t col_ind_offset_ct4 = col_ind_buf_ct4.second;
            std::pair<dpct::buffer_t, size_t> values_buf_ct5 =
                dpct::get_buffer_and_offset(values);
            size_t values_offset_ct5 = values_buf_ct5.second;
            std::pair<dpct::buffer_t, size_t> x_buf_ct6 =
                dpct::get_buffer_and_offset(x);
            size_t x_offset_ct6 = x_buf_ct6.second;
            std::pair<dpct::buffer_t, size_t> y_buf_ct7 =
                dpct::get_buffer_and_offset(y);
            size_t y_offset_ct7 = y_buf_ct7.second;
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::accessor<T, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    result_acc_ct1(sycl::range<1>(256 /*THREADS_PER_BLOCK*/),
                                   cgh);
                auto row_ptr_acc_ct3 =
                    row_ptr_buf_ct3.first
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto col_ind_acc_ct4 =
                    col_ind_buf_ct4.first
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto values_acc_ct5 =
                    values_buf_ct5.first
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto x_acc_ct6 =
                    x_buf_ct6.first.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto y_acc_ct7 =
                    y_buf_ct7.first.get_access<sycl::access::mode::read_write>(
                        cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, multiply_blocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        int *row_ptr_ct3 =
                            (int *)(&row_ptr_acc_ct3[0] + row_ptr_offset_ct3);
                        int *col_ind_ct4 =
                            (int *)(&col_ind_acc_ct4[0] + col_ind_offset_ct4);
                        T *values_ct5 =
                            (T *)(&values_acc_ct5[0] + values_offset_ct5);
                        T *x_ct6 = (T *)(&x_acc_ct6[0] + x_offset_ct6);
                        T *y_ct7 = (T *)(&y_acc_ct7[0] + y_offset_ct7);
                        warp_multiply_kernel<T>(
                            group_size, rows, 0, row_ptr_ct3, col_ind_ct4,
                            values_ct5, x_ct6, y_ct7, item_ct1,
                            result_acc_ct1.get_pointer());
                    });
            });
        }

        // alpha_i <- y_i*x_i
        T product = device_dot_product(rows, x, y, scratch);

        result.alpha(i) = product;

        // y_i <- y_i - alpha_i*x_i - beta_i*x_(i-1)
        {
            std::pair<dpct::buffer_t, size_t> y_buf_ct1 =
                dpct::get_buffer_and_offset(y);
            size_t y_offset_ct1 = y_buf_ct1.second;
            std::pair<dpct::buffer_t, size_t> x_buf_ct2 =
                dpct::get_buffer_and_offset(x);
            size_t x_offset_ct2 = x_buf_ct2.second;
            q_ct1.submit([&](sycl::handler &cgh) {
                auto y_acc_ct1 =
                    y_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto x_acc_ct2 =
                    x_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                        cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, blocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        T *y_ct1 = (T *)(&y_acc_ct1[0] + y_offset_ct1);
                        T *x_ct2 = (T *)(&x_acc_ct2[0] + x_offset_ct2);
                        saxpy_inplace_kernel<T>(rows, y_ct1, x_ct2, -product,
                                                item_ct1);
                    });
            });
        }

        if (i > 0) {
            std::pair<dpct::buffer_t, size_t> y_buf_ct1 =
                dpct::get_buffer_and_offset(y);
            size_t y_offset_ct1 = y_buf_ct1.second;
            std::pair<dpct::buffer_t, size_t> x_prev_buf_ct2 =
                dpct::get_buffer_and_offset(x_prev);
            size_t x_prev_offset_ct2 = x_prev_buf_ct2.second;
            q_ct1.submit([&](sycl::handler &cgh) {
                auto y_acc_ct1 =
                    y_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto x_prev_acc_ct2 =
                    x_prev_buf_ct2.first
                        .get_access<sycl::access::mode::read_write>(cgh);

                auto result_beta_i_ct3 = -result.beta(i - 1);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, blocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        T *y_ct1 = (T *)(&y_acc_ct1[0] + y_offset_ct1);
                        T *x_prev_ct2 =
                            (T *)(&x_prev_acc_ct2[0] + x_prev_offset_ct2);
                        saxpy_inplace_kernel<T>(rows, y_ct1, x_prev_ct2,
                                                result_beta_i_ct3, item_ct1);
                    });
            });
        }
        std::swap(x, x_prev);

        // beta_(i+1) <- ||y_i||
        result.beta(i) = T(sqrt(device_dot_product(rows, y, y, scratch)));

        // x_(i+1) <- y_i / beta_(i+1)
        {
            std::pair<dpct::buffer_t, size_t> y_buf_ct1 =
                dpct::get_buffer_and_offset(y);
            size_t y_offset_ct1 = y_buf_ct1.second;
            q_ct1.submit([&](sycl::handler &cgh) {
                auto y_acc_ct1 =
                    y_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto result_beta_i_ct2 = 1 / result.beta(i);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, blocks) *
                            sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                        T *y_ct1 = (T *)(&y_acc_ct1[0] + y_offset_ct1);
                        multiply_inplace_kernel<T>(rows, y_ct1,
                                                   result_beta_i_ct2, item_ct1);
                    });
            });
        }
        std::swap(x, y);
    }
    dev_ct1.queues_wait_and_throw();
    double end_time = cycle_timer::current_seconds();
    cout << "GPU Lanczos iterations: " << steps << endl;
    cout << "GPU Lanczos time: " << end_time - start_time << " sec" << endl;

    // Release device space
    dpct::dpct_free(row_ptr);
    dpct::dpct_free(col_ind);
    dpct::dpct_free(values);
    dpct::dpct_free(x);
    dpct::dpct_free(x_prev);
    dpct::dpct_free(y);
    dpct::dpct_free(scratch);

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
