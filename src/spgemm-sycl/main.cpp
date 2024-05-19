#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <oneapi/mkl/spblas.hpp>
#include "utils.h"

int COO(int m, int k, int n, int a_nnz, int repeat, int verify) {
  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = n;
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int ldc        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;
  const int C_num_rows = A_num_rows;
  const int C_num_cols = B_num_cols;
  const int C_size     = C_num_rows * C_num_cols;

  float *hA = (float*) malloc (A_size * sizeof(float));
  float *hB = (float*) malloc (B_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = A_nnz * sizeof(int);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_rows = (int*) malloc (A_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_coo(hA_rows, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  init_matrix(hB, B_num_rows, B_num_cols, B_size);

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_rows, *dA_columns;
  float *dA_values, *dB, *dC;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // allocate A
  dA_rows = sycl::malloc_device<int>(A_num_rows + 1, q);
  dA_columns = sycl::malloc_device<int>(A_nnz, q);
  dA_values = sycl::malloc_device<float>(A_nnz, q);

  // allocate B
  dB = sycl::malloc_device<float>(B_size, q);

  // allocate C
  dC = sycl::malloc_device<float>(C_size, q);

  // copy A
  q.memcpy(dA_rows, hA_rows, (A_num_rows + 1) * sizeof(int));
  q.memcpy(dA_columns, hA_columns, A_nnz * sizeof(int));
  q.memcpy(dA_values, hA_values, A_nnz * sizeof(float));
  // copy B
  q.memcpy(dB, hB, B_size * sizeof(float));
  //--------------------------------------------------------------------------

  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  float alpha = 1.f;
  float beta  = 0.f;
  oneapi::mkl::transpose opA = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose opB = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::layout layout = oneapi::mkl::layout::row_major;
  oneapi::mkl::index_base A_index = oneapi::mkl::index_base::zero;

  oneapi::mkl::sparse::set_coo_data(q, handle, A_num_rows, A_num_cols, A_nnz, 
                                    A_index, dA_rows, dA_columns, dA_values, {}).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::sparse::gemm(q, layout, opA, opB, alpha, handle,
                              dB, C_num_cols, ldb, beta, dC, ldc, {});
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPGEMM (COO) compute: %f (us)\n", (time * 1e-3f) / repeat);

  oneapi::mkl::sparse::release_matrix_handle(q, &handle, {}).wait();

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference SPGEMM results..\n");
    float *hC = (float*) malloc (C_size * sizeof(float));
    gemm (hA, hB, hC, A_num_cols, A_num_rows, B_num_cols);

    float *hC_tmp = (float*) malloc (C_size * sizeof(float));
    q.memcpy(hC_tmp, dC, C_size * sizeof(float)).wait();

    int correct = 1;
    for (int i = 0; i < C_size; i++) {
      if (fabsf(hC_tmp[i] - hC[i]) > 1e-2f) {
        printf("@%d %f != %f\n", i, hC_tmp[i], hC[i]);
        correct = 0;                       
        break;
      }
    }

    free(hC_tmp);
    free(hC);

    if (correct)
      printf("spgemm_example test PASSED\n");
    else
      printf("spgemm_example test FAILED: wrong result\n");
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  sycl::free(dA_rows, q);
  sycl::free(dA_columns, q);
  sycl::free(dA_values, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
  free(hA);
  free(hB);
  free(hA_values);
  free(hA_columns);
  free(hA_rows);
  return EXIT_SUCCESS;
}

int CSR(int m, int k, int n, int a_nnz, int repeat, int verify) {
  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = n;
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int ldc        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;
  const int C_num_rows = A_num_rows;
  const int C_num_cols = B_num_cols;
  const int C_size     = C_num_rows * C_num_cols;

  float *hA = (float*) malloc (A_size * sizeof(float));
  float *hB = (float*) malloc (B_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(size_t);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_offsets = (int*) malloc (A_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  init_matrix(hB, B_num_rows, B_num_cols, B_size);

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_offsets, *dA_columns;
  float *dA_values, *dB, *dC;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // allocate A
  dA_offsets = sycl::malloc_device<int>(A_num_rows + 1, q);
  dA_columns = sycl::malloc_device<int>(A_nnz, q);
  dA_values = sycl::malloc_device<float>(A_nnz, q);

  // allocate B
  dB = sycl::malloc_device<float>(B_size, q);

  // allocate C
  dC = sycl::malloc_device<float>(C_size, q);

  // copy A
  q.memcpy(dA_offsets, hA_offsets, (A_num_rows + 1) * sizeof(int));
  q.memcpy(dA_columns, hA_columns, A_nnz * sizeof(int));
  q.memcpy(dA_values, hA_values, A_nnz * sizeof(float));
  // copy B
  q.memcpy(dB, hB, B_size * sizeof(float));
  //--------------------------------------------------------------------------

  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  float alpha = 1.f;
  float beta  = 0.f;
  oneapi::mkl::transpose opA = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose opB = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::layout layout = oneapi::mkl::layout::row_major;
  oneapi::mkl::index_base A_index = oneapi::mkl::index_base::zero;

  oneapi::mkl::sparse::set_csr_data(q, handle, A_num_rows, A_num_cols, 
                                    A_index, dA_offsets, dA_columns, dA_values, {}).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::sparse::gemm(q, layout, opA, opB, alpha, handle,
                              dB, C_num_cols, ldb, beta, dC, ldc, {});
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SPGEMM (CSR) compute: %f (us)\n", (time * 1e-3f) / repeat);

  oneapi::mkl::sparse::release_matrix_handle(q, &handle, {}).wait();

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference SPGEMM results..\n");
    float *hC = (float*) malloc (C_size * sizeof(float));
    gemm (hA, hB, hC, A_num_cols, A_num_rows, B_num_cols);

    float *hC_tmp = (float*) malloc (C_size * sizeof(float));
    q.memcpy(hC_tmp, dC, C_size * sizeof(float)).wait();

    int correct = 1;
    for (int i = 0; i < C_size; i++) {
      if (fabsf(hC_tmp[i] - hC[i]) > 1e-2f) {
        printf("@%d %f != %f\n", i, hC_tmp[i], hC[i]);
        correct = 0;                       
        break;
      }
    }

    free(hC_tmp);
    free(hC);

    if (correct)
      printf("spgemm_example test PASSED\n");
    else
      printf("spgemm_example test FAILED: wrong result\n");
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  sycl::free(dA_offsets, q);
  sycl::free(dA_columns, q);
  sycl::free(dA_values, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
  free(hA);
  free(hB);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 7) {
    printf("Single-precision sparse matrix-dense matrix multiplication into dense matrix,\n");
    printf("where the sparse matrix is represented in COO and CSR storage format\n");
    printf("Usage %s <M> <K> <N> <A_nnz> <repeat> <verify>\n", argv[0]);
    printf("SPMM (A, B, C) where (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }

  int m, k, n, a_nnz, verify;

  m = atoi(argv[1]);
  k = atoi(argv[2]);
  n = atoi(argv[3]);
  a_nnz = atoi(argv[4]);
  repeat = atoi(argv[5]);
  verify = atoi(argv[6]);

  // The supported cases for <sparse_matrix_type> are csr on CPU and GPU devices, and coo only on CPU device.
  COO(m, k, n, a_nnz, repeat, verify);
  CSR(m, k, n, a_nnz, repeat, verify);

  return EXIT_SUCCESS;
}
