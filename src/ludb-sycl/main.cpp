/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This example demonstrates how to use the oneMKL library API
 * for lower-upper (LU) decomposition of a matrix. LU decomposition
 * factors a matrix as the product of upper triangular matrix and
 * lower trianglular matrix.
 *
 * https://en.wikipedia.org/wiki/LU_decomposition
 *
 * This sample uses 10000 matrices of size NxN and performs
 * LU decomposition of them using batched decomposition API
 * of oneMKL library. To test the correctness of upper and lower
 * matrices generated, they are multiplied and compared with the
 * original input matrix.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// oneAPI libraries and helpers
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

// configurable parameters
// dimension of matrix
#define N 48
#define BATCH_SIZE 10000
#define CHECK_ERROR

//#define DOUBLE_PRECISION 

#ifdef DOUBLE_PRECISION
#define DATA_TYPE double
#define MAX_ERROR 1e-15
#else
#define DATA_TYPE float
#define MAX_ERROR 1e-6
#endif /* DOUBLE_PRCISION */

// helper functions
template <typename T>
inline void getrf_batch_wrapper(sycl::queue &q, int n, T *a[],
                                int lda, int *ipiv, int *info, int batch_size) {
  std::int64_t m_int64 = n;
  std::int64_t n_int64 = n;
  std::int64_t lda_int64 = lda;
  std::int64_t group_count = 1;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<T>(
      q, &m_int64, &n_int64, &lda_int64, group_count, &group_sizes);

  T **a_shared = sycl::malloc_shared<T *>(batch_size, q);
  q.memcpy(a_shared, a, batch_size * sizeof(T *));

  T *scratchpad = sycl::malloc_device<T>(scratchpad_size, q);
  std::int64_t *ipiv_int64 = sycl::malloc_device<std::int64_t>(batch_size * n, q);
  std::int64_t **ipiv_int64_ptr = sycl::malloc_shared<std::int64_t *>(batch_size, q);

  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  oneapi::mkl::lapack::getrf_batch(q, &m_int64, &n_int64, a_shared, &lda_int64,
                           ipiv_int64_ptr, group_count, &group_sizes, scratchpad,
                           scratchpad_size);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
      ipiv[idx] = static_cast<int>(ipiv_int64[idx]);
    });
  });

  q.wait();
  sycl::free(scratchpad, q);
  sycl::free(ipiv_int64, q);
  sycl::free(ipiv_int64_ptr, q);
  sycl::free(a_shared, q);
}

// wrapper
void getrfBatched(sycl::queue &q, int n, DATA_TYPE *A[],
                  int lda, int *P, int *info, int batchSize) {
  return getrf_batch_wrapper(q, n, A, lda, P, info, batchSize);
}

// wrapper around malloc
// clears the allocated memory to 0
// terminates the program if malloc fails
void* xmalloc(size_t size) {
  void* ptr = malloc(size);
  if (ptr == NULL) {
    printf("> ERROR: malloc for size %zu failed..\n", size);
    exit(EXIT_FAILURE);
  }
  memset(ptr, 0, size);
  return ptr;
}

// initalize identity matrix
void initIdentityMatrix(DATA_TYPE* mat) {
  // clear the matrix
  memset(mat, 0, N * N * sizeof(DATA_TYPE));

  // set all diagonals to 1
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] = 1.0;
  }
}

// initialize matrix with all elements as 0
void initZeroMatrix(DATA_TYPE* mat) {
  memset(mat, 0, N * N * sizeof(DATA_TYPE));
}

// fill random value in column-major matrix
void initRandomMatrix(DATA_TYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[(j * N) + i] =
          (DATA_TYPE)1.0 + ((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX);
    }
  }

  // diagonal dominant matrix to insure it is invertible matrix
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] += (DATA_TYPE)N;
  }
}

// print column-major matrix
void printMatrix(DATA_TYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%20.16f ", mat[(j * N) + i]);
    }
    printf("\n");
  }
  printf("\n");
}

// matrix mulitplication
void matrixMultiply(DATA_TYPE* res, DATA_TYPE* mat1, DATA_TYPE* mat2) {
  initZeroMatrix(res);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res[(j * N) + i] += mat1[(k * N) + i] * mat2[(j * N) + k];
      }
    }
  }
}

// check matrix equality
bool checkRelativeError(DATA_TYPE* mat1, DATA_TYPE* mat2, DATA_TYPE maxError) {
  DATA_TYPE err = (DATA_TYPE)0.0;
  DATA_TYPE refNorm = (DATA_TYPE)0.0;
  DATA_TYPE relError = (DATA_TYPE)0.0;
  DATA_TYPE relMaxError = (DATA_TYPE)0.0;

  for (int i = 0; i < N * N; i++) {
    refNorm = abs(mat1[i]);
    err = abs(mat1[i] - mat2[i]);

    if (refNorm != 0.0 && err > 0.0) {
      relError = err / refNorm;
      relMaxError = relMaxError > relError ? relMaxError : relError;
    }

    if (relMaxError > maxError) return false;
  }
  return true;
}

// decode lower and upper matrix from single matrix
// returned by getrfBatched()
void getLUdecoded(DATA_TYPE* mat, DATA_TYPE* L, DATA_TYPE* U) {
  // init L as identity matrix
  initIdentityMatrix(L);

  // copy lower triangular values from mat to L (skip diagonal)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      L[(j * N) + i] = mat[(j * N) + i];
    }
  }

  // init U as all zero
  initZeroMatrix(U);

  // copy upper triangular values from mat to U
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      U[(j * N) + i] = mat[(j * N) + i];
    }
  }
}

// generate permutation matrix from pivot vector
void getPmatFromPivot(DATA_TYPE* Pmat, int* P) {
  int pivot[N];

  // pivot vector in base-1
  // convert it to base-0
  for (int i = 0; i < N; i++) {
    P[i]--;
  }

  // generate permutation vector from pivot
  // initialize pivot with identity sequence
  for (int k = 0; k < N; k++) {
    pivot[k] = k;
  }

  // swap the indices according to pivot vector
  for (int k = 0; k < N; k++) {
    int q = P[k];

    // swap pivot(k) and pivot(q)
    int s = pivot[k];
    int t = pivot[q];
    pivot[k] = t;
    pivot[q] = s;
  }

  // generate permutation matrix from pivot vector
  initZeroMatrix(Pmat);
  for (int i = 0; i < N; i++) {
    int j = pivot[i];
    Pmat[(j * N) + i] = (DATA_TYPE)1.0;
  }
}

int main(int argc, char **argv) try {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // host variables
  size_t matSize = N * N * sizeof(DATA_TYPE);

  DATA_TYPE* h_AarrayInput;
  DATA_TYPE* h_AarrayOutput;
  DATA_TYPE* h_ptr_array[BATCH_SIZE];

  int* h_pivotArray;
  int* h_infoArray;

  // device variables
  DATA_TYPE* d_Aarray;
  DATA_TYPE** d_ptr_array;

  int* d_pivotArray;
  int* d_infoArray;

  int err_count = 0;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // seed the rand() function with time
  srand(12345);
  std::cout << "\nRunning on " << q.get_device().get_info<sycl::info::device::name>()<<"\n";
  printf("> initializing..\n");

#ifdef DOUBLE_PRECISION
  printf("> using DOUBLE precision..\n");
#else
  printf("> using SINGLE precision..\n");
#endif

  printf("> pivot ENABLED..\n");

  // allocate memory for host variables
  h_AarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);
  h_AarrayOutput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);

  h_pivotArray = (int*)xmalloc(N * BATCH_SIZE * sizeof(int));
  h_infoArray = (int*)xmalloc(BATCH_SIZE * sizeof(int));

  // allocate memory for device variables

  CHECK_ERROR(d_Aarray = (DATA_TYPE *)sycl::malloc_device(BATCH_SIZE * matSize, q));
  CHECK_ERROR(d_pivotArray = sycl::malloc_device<int>(N * BATCH_SIZE, q));
  CHECK_ERROR(d_infoArray = sycl::malloc_device<int>(BATCH_SIZE, q));
  CHECK_ERROR(d_ptr_array = (DATA_TYPE **)sycl::malloc_device(BATCH_SIZE * sizeof(DATA_TYPE *), q));

  // fill matrix with random data
  printf("> generating random matrices..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initRandomMatrix(h_AarrayInput + (i * N * N));
  }

  // create pointer array for matrices
  for (int i = 0; i < BATCH_SIZE; i++) h_ptr_array[i] = d_Aarray + (i * N * N);

  // copy pointer array to device memory
  CHECK_ERROR(q.memcpy(d_ptr_array, h_ptr_array, BATCH_SIZE * sizeof(DATA_TYPE *)));

  long time = 0;
  // perform LU decomposition
  printf("> performing LU decomposition..\n");
  for (int i = 0; i <= repeat; i++) {
    // copy data to device from host
    //printf("> copying data from host memory to GPU memory..\n");
    CHECK_ERROR(q.memcpy(d_Aarray, h_AarrayInput, BATCH_SIZE * matSize).wait());

    auto start = std::chrono::steady_clock::now();
    getrfBatched(q, N, d_ptr_array, N, d_pivotArray, d_infoArray, BATCH_SIZE);
    auto end = std::chrono::steady_clock::now();
    if (i != 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average kernel execution time : %f (us)\n", (time * 1e-3f) / repeat);


  // copy data to host from device
  printf("> copying data from GPU memory to host memory..\n");
  CHECK_ERROR(q.memcpy(h_AarrayOutput, d_Aarray, BATCH_SIZE * matSize));
  CHECK_ERROR(q.memcpy(h_infoArray, d_infoArray, BATCH_SIZE * sizeof(int)));
  CHECK_ERROR(q.memcpy(h_pivotArray, d_pivotArray, N * BATCH_SIZE * sizeof(int)));
  q.wait();

  // verify the result
  printf("> verifying the result..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (h_infoArray[i] == 0) {
      DATA_TYPE* A = h_AarrayInput + (i * N * N);
      DATA_TYPE* LU = h_AarrayOutput + (i * N * N);
      DATA_TYPE L[N * N];
      DATA_TYPE U[N * N];
      getLUdecoded(LU, L, U);

      // test P * A = L * U
      int* P = h_pivotArray + (i * N);
      DATA_TYPE Pmat[N * N];
      getPmatFromPivot(Pmat, P);

      // perform matrix multiplication
      DATA_TYPE PxA[N * N];
      DATA_TYPE LxU[N * N];
      matrixMultiply(PxA, Pmat, A);
      matrixMultiply(LxU, L, U);

      // check for equality of matrices
      if (!checkRelativeError(PxA, LxU, (DATA_TYPE)MAX_ERROR)) {
        printf("> ERROR: accuracy check failed for matrix number %05d..\n",
               i + 1);
        err_count++;
      }

    } else if (h_infoArray[i] > 0) {
      printf(
          "> execution for matrix %05d is successful, but U is singular and "
          "U(%d,%d) = 0..\n",
          i + 1, h_infoArray[i] - 1, h_infoArray[i] - 1);
    } else  // (h_infoArray[i] < 0)
    {
      printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
             i + 1, -h_infoArray[i],
             *(h_AarrayInput + (i * N * N) + (-h_infoArray[i])));
    }
  }

  // free device variables
  CHECK_ERROR(sycl::free(d_ptr_array, q));
  CHECK_ERROR(sycl::free(d_infoArray, q));
  CHECK_ERROR(sycl::free(d_pivotArray, q));
  CHECK_ERROR(sycl::free(d_Aarray, q));

  // free host variables
  if (h_infoArray) free(h_infoArray);
  if (h_pivotArray) free(h_pivotArray);
  if (h_AarrayOutput) free(h_AarrayOutput);
  if (h_AarrayInput) free(h_AarrayInput);

  if (err_count > 0) {
    printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count,
           MAX_ERROR);
    return (EXIT_FAILURE);
  }

  printf("> TEST SUCCESSFUL, with precision: %g\n", MAX_ERROR);
  return (EXIT_SUCCESS);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
