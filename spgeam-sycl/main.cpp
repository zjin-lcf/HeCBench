#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <oneapi/mkl.hpp>
#include "utils.h"

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 6) {
    printf("The function performs an out-of-place transpose of a sparse matrix into a new one,");
    printf("where the input sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");
    printf("Usage %s <M> <K> <nnz> <repeat> <verify>\n", argv[0]);
    printf("Input matrix A: M * K\n");
    printf("Output matrix B: M * K or K * M\n");
    return 1;
  }

  int m, k, a_nnz, verify;

  m = atoi(argv[1]);
  k = atoi(argv[2]);
  a_nnz = atoi(argv[3]);
  repeat = atoi(argv[4]);
  verify = atoi(argv[5]);

  const int transpose = 1;

  oneapi::mkl::transpose opA = transpose ? oneapi::mkl::transpose::trans :
                                           oneapi::mkl::transpose::nontrans;
 
  // Host problem definition
  const int A_num_rows = m;
  const int A_num_cols = k;
  const int A_nnz      = a_nnz;
  const int B_nnz      = a_nnz;
  const int B_num_rows = (opA == oneapi::mkl::transpose::nontrans ? A_num_rows : A_num_cols);
  const int B_num_cols = (opA == oneapi::mkl::transpose::nontrans ? A_num_cols : A_num_rows);
  const int lda        = A_num_cols;
  const int ldb        = B_num_cols;
  const int A_size     = lda * A_num_rows;
  const int B_size     = ldb * B_num_rows;

  float *hA = (float*) malloc (A_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(int);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_offsets = (int*) malloc (A_rowidx_size_bytes);

  init_matrix(hA, A_num_rows, A_num_cols, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  //--------------------------------------------------------------------------
  // Device memory management: Allocate and copy A, B
  int   *dA_offsets, *dA_columns, *dB_offsets, *dB_columns;
  float *dA_values, *dB_values;

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
  dB_offsets = sycl::malloc_device<int>(B_num_rows + 1, q);
  dB_columns = sycl::malloc_device<int>(B_nnz, q);
  dB_values = sycl::malloc_device<float>(B_nnz, q);

  // copy A
  q.memcpy(dA_offsets, hA_offsets, A_rowidx_size_bytes);
  q.memcpy(dA_columns, hA_columns, A_colidx_size_bytes);
  q.memcpy(dA_values, hA_values, A_value_size_bytes);
  //--------------------------------------------------------------------------

  oneapi::mkl::sparse::matrix_handle_t A = nullptr;
  oneapi::mkl::sparse::matrix_handle_t B = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&A);
  oneapi::mkl::sparse::init_matrix_handle(&B);

  // float alpha = 1.f;
  // float beta  = 0.f;
  oneapi::mkl::index_base A_index = oneapi::mkl::index_base::zero;
  oneapi::mkl::index_base B_index = oneapi::mkl::index_base::zero;
  oneapi::mkl::sparse::set_csr_data(q, A, A_num_rows, A_num_cols, A_index,
                                    dA_offsets, dA_columns, dA_values);
  oneapi::mkl::sparse::set_csr_data(q, B, B_num_rows, B_num_cols, B_index,
                                    dB_offsets, dB_columns, dB_values);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::sparse::omatcopy(q, opA, A, B, {});
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Intel oneMKL sparse omatcopy : %f (us)\n", (time * 1e-3f) / repeat);

  oneapi::mkl::sparse::release_matrix_handle(q, &A, {});
  oneapi::mkl::sparse::release_matrix_handle(q, &B, {});
  q.wait();

  //--------------------------------------------------------------------------
  // device result check
  
  if (verify) {
    printf("Computing the reference results..\n");
    float *hB = (float*) malloc (B_size * sizeof(float));
    transpose_dense (hA, hB, A_num_cols, A_num_rows, transpose);

    const size_t B_value_size_bytes  = B_nnz * sizeof(float);
    const size_t B_colidx_size_bytes = B_nnz * sizeof(int);
    const size_t B_rowidx_size_bytes = (B_num_rows + 1) * sizeof(size_t);

    float *hB_values = (float*) malloc (B_value_size_bytes);
    int *hB_columns = (int*) malloc (B_colidx_size_bytes);
    int *hB_offsets = (int*) malloc (B_rowidx_size_bytes);
    float *hB_values_tmp = (float*) malloc (B_value_size_bytes);
    int *hB_columns_tmp = (int*) malloc (B_colidx_size_bytes);
    int *hB_offsets_tmp = (int*) malloc (B_rowidx_size_bytes);

    init_csr(hB_offsets, hB_values, hB_columns, hB,
             B_num_rows, B_num_cols, B_nnz);

    q.memcpy(hB_offsets_tmp, dB_offsets, B_rowidx_size_bytes);
    q.memcpy(hB_columns_tmp, dB_columns, B_colidx_size_bytes);
    q.memcpy(hB_values_tmp, dB_values, B_value_size_bytes);
    q.wait();

    int correct = 1;
    for (int i = 0; i < B_num_rows + 1; i++) {
      if (hB_offsets_tmp[i] != hB_offsets[i]) {
        correct = 0;
        break;
      }
    }
    for (int i = 0; i < B_nnz; i++) {
      if (hB_columns_tmp[i] != hB_columns[i] ||
          fabsf(hB_values_tmp[i] - hB_values[i]) > 1e-2f) {
        correct = 0;                       
        break;
      }
    }

    free(hB);
    free(hB_values);
    free(hB_columns);
    free(hB_offsets);
    free(hB_values_tmp);
    free(hB_columns_tmp);
    free(hB_offsets_tmp);

    if (correct)
        printf("spgeam_example test PASSED\n");
    else {
        printf("spgeam_example test FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  sycl::free(dA_offsets, q);
  sycl::free(dA_columns, q);
  sycl::free(dA_values, q);
  sycl::free(dB_offsets, q);
  sycl::free(dB_columns, q);
  sycl::free(dB_values, q);
  free(hA);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}
