/*******************************************************************************
* Copyright 2021-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include "utils.h"

//
// Main example for Sparse Matrix sorting
//
template <typename fp, typename intType>
int sort_sparse_matrix(sycl::queue &q, intType a_nrows, intType a_ncols,
                       intType a_nnz, int repeat)
{
  // Initialize data for Sparse Matrix
  oneapi::mkl::index_base a_index = oneapi::mkl::index_base::zero;

  fp *m = (fp*) malloc (sizeof(fp) * a_nrows * a_ncols);
  init_matrix(m, a_nrows, a_ncols, a_nnz);
  //
  // setup data in CSR format
  //
  intType *ia = (intType*) malloc ((a_nrows+1) * sizeof(intType));
  intType *ja = (intType*) malloc (a_nnz * sizeof(intType));
  fp *a = (fp*) malloc (a_nnz * sizeof(fp));

  // reference results
  init_csr(ia, a, ja, m, a_nrows, a_ncols, a_nnz);

  // device results
  intType *ia_result = (intType*) malloc ((a_nrows+1) * sizeof(intType));
  intType *ja_result = (intType*) malloc (a_nnz * sizeof(intType));
  fp *a_result = (fp*) malloc (a_nnz * sizeof(fp));

  // shuffle the inputs
  memcpy(ia_result, ia, (a_nrows+1) * sizeof(intType));
  memcpy(ja_result, ja, a_nnz * sizeof(intType));
  memcpy(a_result, a, a_nnz * sizeof(fp));
  shuffle_matrix_data<fp, intType>(ia_result, ja_result, a_result, a_nrows, a_nnz);

  intType *a_rowptr = (intType *)malloc_device((a_nrows + 1) * sizeof(intType), q);
  intType *a_colind = (intType *)malloc_device((a_nnz) * sizeof(intType), q);
  fp *a_val         = (fp *)malloc_device((a_nnz) * sizeof(fp), q);

  if (!a_rowptr || !a_colind || !a_val) {
    printf("Failed to allocate USM memory");
    if (!a_rowptr) sycl::free(a_rowptr, q);
    if (!a_colind) sycl::free(a_colind, q);
    if (!a_val) sycl::free(a_val, q);
    return 1;
  }

  // copy data to device arrays
  q.memcpy(a_rowptr, ia_result, (a_nrows+1) * sizeof(intType));
  q.memcpy(a_colind, ja_result, a_nnz * sizeof(intType));
  q.memcpy(a_val, a_result, a_nnz * sizeof(fp));

  std::cout << "\n\t\tBasic info of the sparse matrix:\n";

  const fp ave_nnz_per_row = static_cast<fp>(a_nnz)/a_nrows;
  std::cout << "\t\t\tA_nrows = A_ncols = " << a_nrows << std::endl;
  std::cout << "\t\t\tA_nnz   = " << a_nnz << std::endl;

  intType max_nnz_per_row = 0;
  intType min_nnz_per_row = a_nnz;
  for (intType row = 0; row < a_nrows; ++row) {
    const intType loc_nnz_per_row = ia[row+1] - ia[row];

    if (loc_nnz_per_row > max_nnz_per_row) 
      max_nnz_per_row = loc_nnz_per_row;

    if (loc_nnz_per_row < min_nnz_per_row) 
      min_nnz_per_row = loc_nnz_per_row;
  }

  std::cout << "\t\t\t\tmin_nnz_per_row = " << min_nnz_per_row << std::endl;
  std::cout << "\t\t\t\tave_nnz_per_row = " << ave_nnz_per_row << std::endl;
  std::cout << "\t\t\t\tmax_nnz_per_row = " << max_nnz_per_row << std::endl;
  std::cout << std::endl;

  //
  // Execute Matrix Sort
  //

  oneapi::mkl::sparse::matrix_handle_t A = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&A);

  oneapi::mkl::sparse::set_csr_data(q, A, a_nrows, a_ncols, a_index, a_rowptr,
                                    a_colind, a_val, {}).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // Sort the sparse matrix
    oneapi::mkl::sparse::sort_matrix(q, A, {}).wait();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of CSR sort: " << (time * 1e-3f) / repeat
            << " us" << std::endl;

  // Release the matrix_handle
  oneapi::mkl::sparse::release_matrix_handle(q, &A, {}).wait();

  q.memcpy(ia_result, a_rowptr, (a_nrows+1) * sizeof(intType));
  q.memcpy(ja_result, a_colind, a_nnz * sizeof(intType));
  q.memcpy(a_result, a_val, a_nnz * sizeof(fp));
  q.wait();

  int error;

  error = memcmp(ia_result, ia, (a_nrows+1) * sizeof(intType));
  if (error) {
    std::cout << "Error: row index arrays mismatch" << std::endl;
    goto print_error;
  }

  error = memcmp(ja_result, ja, a_nnz * sizeof(intType));
  if (error) {
    std::cout << "Error: column index arrays mismatch" << std::endl;
    goto print_error;
  }

  error = memcmp(a_result, a, a_nnz * sizeof(fp));
  if (error) {
    std::cout << "Error: value arrays mismatch" << std::endl;
    goto print_error;
  }

  print_error:
  if (error == 0)
      std::cout << "csrsort_example test PASSED" << std::endl;
  else
      std::cout << "csrsort_example test FAILED: wrong result" << std::endl;

  sycl::free(a_rowptr, q);
  sycl::free(a_colind, q);
  sycl::free(a_val, q);
  free(m);
  free(ia);
  free(ja);
  free(a);
  free(ia_result);
  free(ja_result);
  free(a_result);
  return 0;
}

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 5) {
    std::cout << "The function sorts the column indices of each row in a sparse MxN matrix" << std::endl;
    std::cout << "The sparse matrix is represented in CSR (Compressed Sparse Row) storage format" << std::endl;
    std::cout << "Usage " << argv[0] << " <M> <N> <nnz> <repeat>" << std::endl;
    std::cout << "nnz is the number of non-zero elements" << std::endl;
    return 1;
  }

  int m, n, nnz;

  m = atoi(argv[1]);
  n = atoi(argv[2]);
  nnz = atoi(argv[3]);
  repeat = atoi(argv[4]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int status = sort_sparse_matrix<float, int>(q, m, n, nnz, repeat);
  if (status != 0) return status;

  return status;
}
