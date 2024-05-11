/*******************************************************************************
* Copyright 2018-2020 Intel Corporation.
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

/*
*
*  Content:
*       This example demonstrates use of DPCPP API mkl::blas::gemm to perform General 
*       Matrix-Matrix Multiplication on a SYCL device (HOST, CPU, GPU).
*
*       C = alpha * op(A) * op(B) + beta * C
*
*       where op() is defined by one of mkl::transpose::{nontrans,trans,conjtrans} 
*
*
*       The supported floating point data types for gemm matrix data are:
*           half
*           float
*           double
*           std::complex<float>
*           std::complex<double>
*
*
*******************************************************************************/

// stl includes
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <vector>
#include <omp.h>
#include "mkl.h"
#include "mkl_omp_offload.h"

//
// Main example for Gemm consisting of 
// initialization of A, B and C matrices as well as 
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
  template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name) 
{
  std::cout << std::endl;
  std::cout << "\t\t\t" << M_name << " = [ " << M[0*ldM + 0] << ", " << M[1*ldM + 0]         << ", ...\n";
  std::cout << "\t\t\t    [ "                << M[0*ldM + 1] << ", " << M[1*ldM + 1] << ", ...\n";
  std::cout << "\t\t\t    [ "                << "...\n";
  std::cout << std::endl;
}

//
// helpers for initializing templated scalar data type values.
//
template <typename fp> void rand_matrix(fp *M, int n_row, int n_col)
{
  for (int i = 0; i < n_row; i++)
    for (int j = 0; j < n_col; j++)
      M[i * n_col + j] = rand() % 5;
}

template <typename fp>
void run_gemm_example(int repeat) {

  // matrix data sizes
  MKL_INT m = 79;
  MKL_INT n = 83; 
  MKL_INT k = 91;

  // set scalar fp values     
  fp alpha = fp(2.0); 
  fp beta  = fp(0.5);

  // prepare matrix data
  fp* a = (float *)mkl_malloc((m * k) * sizeof(float), 64);
  fp* b = (float *)mkl_malloc((k * n) * sizeof(float), 64);
  fp* c = (float *)mkl_malloc((m * n) * sizeof(float), 64);

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);

  //
  // Execute Gemm
  //

  #pragma omp target data map(to:a[0:m*k], b[0:k*n]) map(tofrom:c[0:m*n]) device(0)
  {
    // run gemm on gpu, use standard oneMKL interface within a variant dispatch construct
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) 
    {
      #pragma omp target variant dispatch device(0) use_device_ptr(a, b, c)
      sgemm("N", "N", &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average sgemm execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  //
  // Post Processing
  //
  std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

  // output the top 2x2 block of A matrix
  print_2x2_matrix_values(a, k, "A");

  // output the top 2x2 block of B matrix
  print_2x2_matrix_values(b, n, "B");

  // output the top 2x2 block of C matrix
  print_2x2_matrix_values(c, n, "C");

  mkl_free(a);
  mkl_free(b);
  mkl_free(c);
}

//
// Main entry point for example.  
//
int main (int argc, char ** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  std::cout << "\tRunning with single precision real data type:" << std::endl;
  run_gemm_example<float>(repeat);
  return 0;
}
