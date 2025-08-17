#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <memory>
#include <cublas_v2.h>
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

inline
cublasStatus_t cublas_gemm_ex(__half *A,  __half *B,  __half *C, int m, int n, int k) {
    static __half alpha = 1;
    static __half beta  = 0;
    static cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    
    cudaDataType_t AType, BType, CType;
    cublasComputeType_t  ComputeType;
    AType = BType = CType = CUDA_R_16F;
    ComputeType = CUBLAS_COMPUTE_16F;
    cublasStatus_t status = 
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     n, m, k, 
                     &alpha,
                     B, BType, n,
                     A, AType, k,
                     &beta,
                     C, CType, n,
                     ComputeType,
                     CUBLAS_GEMM_DEFAULT);
   cublasDestroy(handle);
   return status;
}
