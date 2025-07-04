#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <memory>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "timer.h"
#include "lutGEMM"

void random_seed(){
  srand(123);
}
bool rand_bool(){
  return rand()>(RAND_MAX/2);
}
double rand_fp64(double max=1.0){
  double sign[] = {-1.0,1.0};
  return (double)sign[rand_bool()]*rand()/RAND_MAX*rand()/RAND_MAX*max;
}

float rand_fp32(float max=1.0){
  return rand_fp64()*max;
}

template <typename Ta, typename Tc, typename Ts>
bool mkl_gemm(
    sycl::queue &q,
    oneapi::mkl::transpose transA,
    oneapi::mkl::transpose transB,
    int m, int n, int k,
    Ta *A, Ta *B, Tc *C,
    int lda, int ldb, int ldc,
    Ts alpha, Ts beta,
    oneapi::mkl::blas::compute_mode mode =
      oneapi::mkl::blas::compute_mode::standard)
{
  sycl::event status;
  try {
    status = oneapi::mkl::blas::column_major::gemm(
      q,
      transA, transB,
      m, n, k,
      alpha, A, lda,
      B, ldb, beta,
      C, ldc);
  } catch(sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
              << e.what() << std::endl;
    return false;
  }
  status.wait();
  return true;
}

inline int cublas_gemm_ex(sycl::queue &q, sycl::half *A, sycl::half *B,
                          sycl::half *C, int m, int n, int k) {
    static sycl::half alpha = 1;
    static sycl::half beta = 0;
    return mkl_gemm(q,
                    oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans,
                    n, // number of rows of matrix A and C
                    m, // number of columns of matrix B and C
                    k, // number of columns of A and rows of B
                    B,
                    A,
                    C,
                    n, // lda
                    k, // ldb
                    n, // ldc
                    alpha,
                    beta,
                    oneapi::mkl::blas::compute_mode::standard);
}

