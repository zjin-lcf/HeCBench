/*
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Modifications Copyright (C) 2023 Intel Corporation​
 *
 * -- Copyright notice and Licensing terms:
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 *
 * SPDX-License-Identifier: BSD-4-Clause
 */

/*
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.3 - December 2, 2018
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 * -- Copyright notice and Licensing terms:
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */

// FP32 VERSION - Uses single precision on GPU for DGX Spark GB10
#define NUMBER_OF_STREAMS 4
#define CHUNK_SIZE 512
#define NN 128
#define NM 128
#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))

#define _GNU_SOURCE

#define DEVICE_DEBUG

#include <cblas.h>
#include <cublas.h>
#include <cuda.h>
#ifdef DEVICE_DEBUG
#include <iostream>
#endif

extern "C" {

void dpcpp_dgemm(const int ORDER, const int TRANSA, const int TRANSB,
                 const int M, const int N, const int K, const double ALPHA,
                 const double *A, const int LDA, const double *B, const int LDB,
                 const double BETA, double *C, const int LDC);

void dpcpp_dtrsm(int HPL_ORDER, int HPL_SIDE, int HPL_UPLO, int HPL_TRANS,
                 int HPL_DIAG, const int, const int, const double,
                 const double *, const int, double *, const int);
}

void dpcpp_dgemm(const int ORDER, const int TRANSA, const int TRANSB,
                 const int M, const int N, const int K, const double ALPHA,
                 const double *A, const int LDA, const double *B, const int LDB,
                 const double BETA, double *C, const int LDC) {

  if ((M == 0) || (K == 0) || (N == 0)) {
    return;
  }

#ifdef DEVICE_DEBUG
  static int call_count = 0;
  if (call_count++ < 5) {
    std::cout << "dgemm called with M=" << M << ", N=" << N << ", K=" << K << std::endl;
  }
#endif

  if ((N) < NN || (M) < NM || (K) < 128) {

#ifdef DEVICE_DEBUG
    std::cout << "dgemm-Running on CPU" << std::endl;
#endif

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A,
                LDA, B, LDB, BETA, C, LDC);
    return;
  }

#ifdef DEVICE_DEBUG
  std::cout << "dgemm-Running on GPU (FP32)" << std::endl;
#endif

  float *devPtrA, *devPtrB, *devPtrC;

  // Allocate device memory for FP32
  cudaMalloc((void **)&devPtrA, M * K * sizeof(float));
  cudaMalloc((void **)&devPtrB, K * N * sizeof(float));
  cudaMalloc((void **)&devPtrC, M * N * sizeof(float));

  // Convert and copy A (column-major: M x K stored in LDA x K)
  float *hostA = new float[K * LDA];
  for (int i = 0; i < K * LDA; i++) hostA[i] = (float)A[i];
  cudaMemcpy(devPtrA, hostA, K * LDA * sizeof(float), cudaMemcpyHostToDevice);
  delete[] hostA;

  // Convert and copy B (column-major: K x N stored in LDB x N)
  float *hostB = new float[N * LDB];
  for (int i = 0; i < N * LDB; i++) hostB[i] = (float)B[i];
  cudaMemcpy(devPtrB, hostB, N * LDB * sizeof(float), cudaMemcpyHostToDevice);
  delete[] hostB;

  // Convert and copy C (column-major: M x N stored in LDC x N)
  float *hostC = new float[N * LDC];
  for (int i = 0; i < N * LDC; i++) hostC[i] = (float)C[i];
  cudaMemcpy(devPtrC, hostC, N * LDC * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  // Use single precision GEMM
  float alpha_f = (float)ALPHA;
  float beta_f = (float)BETA;
  cublasSgemm('N', 'N', M, N, K, alpha_f, devPtrA, LDA, devPtrB, LDB, beta_f,
              devPtrC, LDC);
  cudaDeviceSynchronize();

  // Copy result back and convert to double
  cudaMemcpy(hostC, devPtrC, N * LDC * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N * LDC; i++) C[i] = (double)hostC[i];
  delete[] hostC;

  cudaDeviceSynchronize();
  cudaFree(devPtrA);
  cudaFree(devPtrB);
  cudaFree(devPtrC);
}

void dpcpp_dtrsm

    (const int ORDER, const int SIDE, const int UPLO, const int TRANS,
     const int DIAG, const int M, const int N, const double ALPHA,
     const double *A, const int LDA, double *B, const int LDB) {

  if ((M == 0) || (N == 0)) {
    return;
  }

  if ((M) < 256 || (N) < 256) {
#ifdef DEVICE_DEBUG
    std::cout << "dtrsm-Running on CPU" << std::endl;
#endif
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                M, N, ALPHA, A, LDA, B, LDB);
    return;
  }

#ifdef DEVICE_DEBUG
  std::cout << "dtrsm-Running on GPU (FP32)" << std::endl;
#endif

  float *devPtrA, *devPtrB;

  cudaMalloc((void **)&devPtrA, M * LDA * sizeof(float));
  float *hostA = new float[M * LDA];
  for (int i = 0; i < M * LDA; i++) hostA[i] = (float)A[i];
  cudaMemcpy(devPtrA, hostA, M * LDA * sizeof(float), cudaMemcpyHostToDevice);
  delete[] hostA;

  cudaMalloc((void **)&devPtrB, N * LDB * sizeof(float));
  float *hostB = new float[N * LDB];
  for (int i = 0; i < N * LDB; i++) hostB[i] = (float)B[i];
  cudaMemcpy(devPtrB, hostB, N * LDB * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  float alpha_f = (float)ALPHA;
  cublasStrsm('L', 'L', 'N', 'U', M, N, alpha_f, devPtrA, LDA, devPtrB, LDB);
  cudaDeviceSynchronize();

  cudaMemcpy(hostB, devPtrB, N * LDB * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N * LDB; i++) B[i] = (double)hostB[i];
  delete[] hostB;

  cudaDeviceSynchronize();
  cudaFree(devPtrA);
  cudaFree(devPtrB);
}

// Fortran-style BLAS wrappers with trailing underscores
extern "C" {

void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc) {
  // Call our CUDA-accelerated version
  dpcpp_dgemm(0, 0, 0, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

void dtrsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, double *b, const int *ldb) {
  // Call our CUDA-accelerated version
  dpcpp_dtrsm(0, 0, 0, 0, 0, *m, *n, *alpha, a, *lda, b, *ldb);
}

} // extern "C"
