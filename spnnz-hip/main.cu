#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <hipsparse/hipsparse.h>
#include "utils.h"

#define CHECK_HIP(func)                                                      \
{                                                                            \
    hipError_t status = (func);                                              \
    if (status != hipSuccess) {                                              \
        printf("HIP API failed at line %d with error: %s (%d)\n",            \
               __LINE__, hipGetErrorString(status), status);                 \
        return EXIT_FAILURE;                                                 \
    }                                                                        \
}

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, "HIPSPARSE API failed at line %d with error: %s\n",         \
               __LINE__, #token_); \
        break

#define CHECK_HIPSPARSE(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);    \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            return local_error;                                                           \
        }                                                                                 \
    }                                                                                     

int main(int argc, char *argv[])
{
  int repeat = 1;

  /*
    https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-t-nnz
    hipsparseStatus_t
    hipsparseSnnz(hipsparseHandle_t    handle,
             hipsparseDirection_t      dirA,
             int                       m,
             int                       n,
             const hipsparseMatDescr_t descrA,
             const float*              A,
             int                       lda,
             int*                      nnzPerRowColumn,
             int*                      nnzTotalDevHostPtr)
   */

  if (argc != 5) {
    printf("This function computes the number of nonzero elements per row or column");
    printf(" and the total number of nonzero elements in a dense matrix.\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    printf("nnz is the number of non-zero elements\n");
    return 1;
  }

  int64_t m, n, h_nnz;

  m = atol(argv[1]);
  n = atol(argv[2]);
  h_nnz = atol(argv[3]);
  repeat = atoi(argv[4]);

  // Host problem definition
  const int64_t num_rows = m;
  const int64_t num_cols = n;
  const int64_t lda = num_cols;
  const int64_t dense_size = m * n;
  const int64_t dense_size_bytes = dense_size * sizeof(float);
  const int64_t nnzRowCol_size_bytes = num_rows * sizeof(int);

  float *h_dense = (float*) malloc (dense_size_bytes);

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, num_rows, num_cols, h_nnz);

  // device results
  int* nnzPerRowColumn = (int*) malloc (nnzRowCol_size_bytes);

  // allocate the device arrays
  int *d_nnzPerRowColumn;
  float *d_dense;
  CHECK_HIP( hipMalloc((void**) &d_nnzPerRowColumn, nnzRowCol_size_bytes)) 
  CHECK_HIP( hipMalloc((void**) &d_dense, dense_size_bytes))
  CHECK_HIP( hipMemcpy(d_dense, h_dense, dense_size_bytes,
                       hipMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // HIPSPARSE APIs
  hipsparseHandle_t   handle = NULL;
  hipsparseMatDescr_t descr = 0;

  int nnzTotal = 0;

  CHECK_HIPSPARSE( hipsparseCreate(&handle) )
  CHECK_HIPSPARSE( hipsparseCreateMatDescr(&descr) )
  CHECK_HIPSPARSE( hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL) )
  CHECK_HIPSPARSE( hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO) )

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    CHECK_HIPSPARSE( hipsparseSnnz(handle,
                                   HIPSPARSE_DIRECTION_COLUMN,
                                   num_rows,
                                   num_cols,
                                   descr,
                                   d_dense,
                                   lda,
                                   d_nnzPerRowColumn,
                                   &nnzTotal) )
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of hipsparseSnnz : %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroyMatDescr(descr) )
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  CHECK_HIP( hipMemcpy(nnzPerRowColumn, d_nnzPerRowColumn, nnzRowCol_size_bytes,
                       hipMemcpyDeviceToHost) )
  int correct = 1; 
  if (h_nnz != nnzTotal) {
    printf("nnz: %d != %d\n", (int)h_nnz, nnzTotal);
    correct = 0;
    goto print_error;
  }

  for (int64_t i = 0; i < num_rows; i++) {
    int nnz = 0; // nnz per row
    for (int64_t j = 0; j < num_cols; j++) {
      if (h_dense[i*num_cols+j] != 0) nnz++; 
    }
    if (nnz != nnzPerRowColumn[i]) {
      printf("@row %ld %d != %d\n", i, nnz, nnzPerRowColumn[i]);
      correct = 0;
      goto print_error;
    }
  }

  print_error:
  if (correct)
      printf("sparse_nnz_example test PASSED\n");
  else
      printf("sparse_nnz_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_HIP( hipFree(d_nnzPerRowColumn) )
  CHECK_HIP( hipFree(d_dense) )
  free(h_dense);
  free(nnzPerRowColumn);
  return EXIT_SUCCESS;
}
