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

#define NUMBER_OF_STREAMS 4
#define CHUNK_SIZE 512
#define NN 64
#define NM 128
#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))

#define _GNU_SOURCE

// #define DEVICE_DEBUG

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

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

  if ((N) < NN || (M) < NM || (K) < 128) {

#ifdef DEVICE_DEBUG
    std::cout << "dgemm-Running on CPU" << std::endl;
#endif

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A,
                LDA, B, LDB, BETA, C, LDC);
    return;
  }

#ifdef DEVICE_DEBUG
  std::cout << "dgemm-Running on GPU" << std::endl;
#endif

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

  double *devPtrA, *devPtrB, *devPtrC;
  int status;

  devPtrA = sycl::malloc_device<double>(K * LDA, q);
  q.memcpy(devPtrA, &A[0], K * LDA * sizeof(double));

  devPtrB = sycl::malloc_device<double>(N * LDB, q);
  q.memcpy(devPtrB, &B[0], N * LDB * sizeof(double));

  devPtrC = sycl::malloc_device<double>(N * LDC, q);
  q.memcpy(devPtrC, &C[0], N * LDC * sizeof(double));

  oneapi::mkl::blas::column_major::gemm(
      q,
      oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, M, N,
      K, ALPHA, devPtrA, LDA, devPtrB, LDB, BETA, devPtrC, LDC)
      .wait();
  q.memcpy(&C[0], devPtrC, N * LDC * sizeof(double)).wait();
  sycl::free(devPtrA, q);
  sycl::free(devPtrB, q);
  sycl::free(devPtrC, q);
}

void dpcpp_dtrsm

    (const int ORDER, const int SIDE, const int UPLO, const int TRANS,
     const int DIAG, const int M, const int N, const double ALPHA,
     const double *A, const int LDA, double *B, const int LDB) {

  if ((M == 0) || (N == 0)) {
    return;
  }

  double *devPtrA, *devPtrB;
  int status;

  if ((M) < 512 || (N) < 2 * (M)) {
#ifdef DEVICE_DEBUG
    std::cout << "dtrsm-Running on CPU" << std::endl;
#endif
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                M, N, ALPHA, A, LDA, B, LDB);
    return;
  }

#ifdef DEVICE_DEBUG
  std::cout << "dtrsm-Running on GPU" << std::endl;
#endif

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

  devPtrA = sycl::malloc_device<double>(M * LDA, q);

  q.memcpy(devPtrA, A, M * LDA * sizeof(double));

  devPtrB = sycl::malloc_device<double>(N * LDB, q);

  q.memcpy(devPtrB, B, N * LDB * sizeof(double));

  oneapi::mkl::blas::column_major::trsm(
      q, oneapi::mkl::side::left,
      oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::diag::unit, M, N, ALPHA, devPtrA, LDA, devPtrB, LDB)
      .wait();

  q.memcpy(B, devPtrB, N * LDB * sizeof(double)).wait();

  sycl::free(devPtrA, q);
  sycl::free(devPtrB, q);
}
