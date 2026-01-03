#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <memory>
#include <hipblas/hipblas.h>
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
hipblasStatus_t cublas_gemm_ex(__half *A,  __half *B,  __half *C, int m, int n, int k) {
    static __half alpha = 1;
    static __half beta  = 0;
    static hipblasHandle_t handle = nullptr;
    hipblasCreate(&handle);
    
    hipDataType AType, BType, CType;
    hipblasComputeType_t  ComputeType;
    AType = BType = CType = HIP_R_16F;
    ComputeType = HIPBLAS_COMPUTE_16F;
    hipblasStatus_t status = 
        hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     n, m, k, 
                     &alpha,
                     B, BType, n,
                     A, AType, k,
                     &beta,
                     C, CType, n,
                     ComputeType,
                     HIPBLAS_GEMM_DEFAULT);
   hipblasDestroy(handle);
   return status;
}
