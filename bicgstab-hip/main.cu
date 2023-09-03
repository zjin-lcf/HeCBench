/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#include "utils.h"

#define CHECK_HIP(func)                                                       \
{                                                                              \
    hipError_t status = (func);                                               \
    if (status != hipSuccess) {                                               \
        printf("HIP API failed at line %d with error: %s (%d)\n",             \
               __LINE__, hipGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_HIPBLAS(func)                                                     \
{                                                                              \
    hipblasStatus_t status = (func);                                            \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                     \
        printf("HIPBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
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


#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

typedef struct VecStruct {
    hipsparseDnVecDescr_t vec;
    double*              ptr;
} Vec;


double gpu_BiCGStab(
                 int                  verbose,
                 hipblasHandle_t       hipblasHandle,
                 hipsparseHandle_t     hipsparseHandle,
                 int                  m,
                 hipsparseSpMatDescr_t matA,
                 hipsparseSpMatDescr_t matM_lower,
                 hipsparseSpMatDescr_t matM_upper,
                 Vec                  d_B,
                 Vec                  d_X,
                 Vec                  d_R0,
                 Vec                  d_R,
                 Vec                  d_P,
                 Vec                  d_P_aux,
                 Vec                  d_S,
                 Vec                  d_S_aux,
                 Vec                  d_V,
                 Vec                  d_T,
                 Vec                  d_tmp,
                 void*                d_bufferMV,
                 int                  maxIterations,
                 double               tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // Create opaque data structures that holds analysis data between calls
    double              coeff_tmp;
    size_t              bufferSizeL, bufferSizeU;
    void*               d_bufferL, *d_bufferU;
    hipsparseSpSVDescr_t spsvDescrL, spsvDescrU;
    CHECK_HIPSPARSE( hipsparseSpSV_createDescr(&spsvDescrL) )
    CHECK_HIPSPARSE( hipsparseSpSV_bufferSize(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, HIP_R_64F,
                        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) )
    CHECK_HIP( hipMalloc(&d_bufferL, bufferSizeL) )
    CHECK_HIPSPARSE( hipsparseSpSV_analysis(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, HIP_R_64F,
                        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) )

    // Calculate UPPER buffersize
    CHECK_HIPSPARSE( hipsparseSpSV_createDescr(&spsvDescrU) )
    CHECK_HIPSPARSE( hipsparseSpSV_bufferSize(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        &bufferSizeU) )
    CHECK_HIP( hipMalloc(&d_bufferU, bufferSizeU) )
    CHECK_HIPSPARSE( hipsparseSpSV_analysis(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        d_bufferU) )
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_HIP( hipMemcpy(d_R0.ptr, d_B.ptr, m * sizeof(double),
                           hipMemcpyDeviceToDevice) )
    //    (b) compute R = -A * X0 + R
    CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R0.vec,
                                 HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) )
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_R0.ptr, 1, d_R0.ptr, 1,
                             &delta) )
    delta_prev = delta;
    // R = R0
    CHECK_HIP( hipMemcpy(d_R.ptr, d_R0.ptr, m * sizeof(double),
                           hipMemcpyDeviceToDevice) )
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_R0.ptr, 1, &nrm_R) )
    double threshold = tolerance * nrm_R;
    if (verbose) printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual

    for (int i = 1; i <= maxIterations; i++) {
        if (verbose) printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        CHECK_HIP( hipMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                               hipMemcpyDeviceToDevice) )
        if (i > 1) {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_R0.ptr, 1, d_R.ptr, 1,
                                     &delta) )
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &minus_omega, d_V.ptr, 1,
                                      d_P.ptr, 1) )
            //    (b) P = beta * P
            CHECK_HIPBLAS( hipblasDscal(hipblasHandle, m, &beta, d_P.ptr, 1) )
            //    (c) P = R + P
            CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &one, d_R.ptr, 1,
                                      d_P.ptr, 1) )
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        CHECK_HIP( hipMemset(d_tmp.ptr,   0x0, m * sizeof(double)) )
        CHECK_HIP( hipMemset(d_P_aux.ptr, 0x0, m * sizeof(double)) )
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_P.vec, d_tmp.vec,
                                           HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL, d_bufferL) )
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_P_aux.vec, HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU, d_bufferU) )
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P_aux.vec, &zero, d_V.vec,
                                     HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) denominator = R'0 * V
        double denominator;
        CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_R0.ptr, 1, d_V.ptr, 1,
                                 &denominator) )
        alpha = delta / denominator;
        if (verbose) PRINT_INFO(delta)
        if (verbose) PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &alpha, d_P_aux.ptr, 1,
                                  d_X.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CHECK_HIP( hipMemcpy(d_S.ptr, d_R.ptr, m * sizeof(double),
                               hipMemcpyDeviceToDevice) )
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &minus_alpha, d_V.ptr, 1,
                                  d_S.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        double nrm_S;
        CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_S.ptr, 1, &nrm_S) )
        if (verbose) PRINT_INFO(nrm_S)
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        hipMemset(d_tmp.ptr, 0x0, m * sizeof(double));
        hipMemset(d_S_aux.ptr, 0x0, m * sizeof(double));
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_S.vec, d_tmp.vec,
                                           HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL, d_bufferL) )
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_S_aux.vec, HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU, d_bufferU) )
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_S_aux.vec, &zero, d_T.vec,
                                     HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_T.ptr, 1, d_S.ptr, 1,
                                 &omega_num) )
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_T.ptr, 1, d_T.ptr, 1,
                                 &omega_den) )
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        if (verbose) PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &omega, d_S_aux.ptr, 1,
                                  d_X.ptr, 1) )
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CHECK_HIP( hipMemcpy(d_R.ptr, d_S.ptr, m * sizeof(double),
                               hipMemcpyDeviceToDevice) )
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &minus_omega, d_T.ptr, 1,
                                  d_R.ptr, 1) )
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_R.ptr, 1, &nrm_R) )
        if (verbose) PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    if (verbose) printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_HIP( hipMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           hipMemcpyDeviceToDevice) )
    // R = -A * X + R
    CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, HIP_R_64F,
                                 HIPSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // check ||R||
    CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_R.ptr, 1, &nrm_R) )
    //--------------------------------------------------------------------------
    CHECK_HIPSPARSE( hipsparseSpSV_destroyDescr(spsvDescrL) )
    CHECK_HIPSPARSE( hipsparseSpSV_destroyDescr(spsvDescrU) )
    CHECK_HIP( hipFree(d_bufferL) )
    CHECK_HIP( hipFree(d_bufferU) )
    return nrm_R;
}

//==============================================================================
//==============================================================================

int main(int argc, char** argv) {
    const double tolerance     = 0.0000000001;
    if (argc != 4) {
        printf("Usage: %s <matrix.mtx> <maximum number of iterations> <verbose output>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *file_path = argv[1]; 
    const int maxIterations = atoi(argv[2]);
    const int verbose = atoi(argv[3]);

    int base = 0;
    int num_rows, num_cols, nnz, num_lines, is_symmetric;
    mtx_header(file_path, &num_lines, &num_rows, &num_cols, &nnz, &is_symmetric);
    printf("\nmatrix name: %s\n"
           "num. rows:   %d\n"
           "num. cols:   %d\n"
           "nnz:         %d\n"
           "structure:   %s\n\n",
           file_path, num_rows, num_cols, nnz,
           (is_symmetric) ? "symmetric" : "unsymmetric");
    if (num_rows != num_cols) {
        printf("the input matrix must be square\n");
        return EXIT_FAILURE;
    }
    if (!is_symmetric) {
        printf("the input matrix must be symmetric\n");
        return EXIT_FAILURE;
    }
    int     m           = num_rows;
    int     num_offsets = m + 1;
    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
    printf("Matrix parsing...\n");
    mtx_parsing(file_path, num_lines, num_rows, nnz, h_A_rows,
                h_A_columns, h_A_values, base);
    printf("Testing BiCGStab\n");
    for (int i = 0; i < num_rows; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_M_values;
    Vec     d_B, d_X, d_R, d_R0, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CHECK_HIP( hipMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) )
    CHECK_HIP( hipMalloc((void**) &d_A_columns, nnz * sizeof(int)) )
    CHECK_HIP( hipMalloc((void**) &d_A_values,  nnz * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_M_values,  nnz * sizeof(double)) )

    CHECK_HIP( hipMalloc((void**) &d_B.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_X.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_R.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_R0.ptr,    m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_P.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_P_aux.ptr, m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_S.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_S_aux.ptr, m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_V.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_T.ptr,     m * sizeof(double)) )
    CHECK_HIP( hipMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) )

    // copy the CSR matrices and vectors into device memory
    CHECK_HIP( hipMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           hipMemcpyHostToDevice) )
    CHECK_HIP( hipMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           hipMemcpyHostToDevice) )
    CHECK_HIP( hipMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           hipMemcpyHostToDevice) )
    CHECK_HIP( hipMemcpy(d_M_values, h_A_values, nnz * sizeof(double),
                           hipMemcpyHostToDevice) )
    CHECK_HIP( hipMemcpy(d_X.ptr, h_X, m * sizeof(double),
                           hipMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    hipblasHandle_t   hipblasHandle   = NULL;
    hipsparseHandle_t hipsparseHandle = NULL;
    CHECK_HIPBLAS( hipblasCreate(&hipblasHandle) )
    CHECK_HIPSPARSE( hipsparseCreate(&hipsparseHandle) )
    // Create dense vectors
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_B.vec,     m, d_B.ptr, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_X.vec,     m, d_X.ptr, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_R.vec,     m, d_R.ptr, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_R0.vec,    m, d_R0.ptr, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_P.vec,     m, d_P.ptr, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_P_aux.vec, m, d_P_aux.ptr,
                                        HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_S.vec,     m, d_S.ptr, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_S_aux.vec, m, d_S_aux.ptr,
                                        HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_V.vec,   m, d_V.ptr,   HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, HIP_R_64F) )

    hipsparseIndexBase_t  baseIdx = HIPSPARSE_INDEX_BASE_ZERO;
    // IMPORTANT: Upper/Lower triangular decompositions of A
    //            (matM_lower, matM_upper) must use two distinct descriptors
    hipsparseSpMatDescr_t matA, matM_lower, matM_upper;
    hipsparseMatDescr_t   matLU;
    int*                 d_M_rows      = d_A_rows;
    int*                 d_M_columns   = d_A_columns;
    hipsparseFillMode_t   fill_lower    = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDiagType_t   diag_unit     = HIPSPARSE_DIAG_TYPE_UNIT;
    hipsparseFillMode_t   fill_upper    = HIPSPARSE_FILL_MODE_UPPER;
    hipsparseDiagType_t   diag_non_unit = HIPSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CHECK_HIPSPARSE( hipsparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      baseIdx, HIP_R_64F) )
    // M_lower
    CHECK_HIPSPARSE( hipsparseCreateCsr(&matM_lower, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      baseIdx, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseSpMatSetAttribute(matM_lower,
                                              HIPSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) )
    CHECK_HIPSPARSE( hipsparseSpMatSetAttribute(matM_lower,
                                              HIPSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)) )
    // M_upper
    CHECK_HIPSPARSE( hipsparseCreateCsr(&matM_upper, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      baseIdx, HIP_R_64F) )
    CHECK_HIPSPARSE( hipsparseSpMatSetAttribute(matM_upper,
                                              HIPSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)) )
    CHECK_HIPSPARSE( hipsparseSpMatSetAttribute(matM_upper,
                                              HIPSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) )
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    const double alpha = 0.75;
    size_t bufferSizeMV;
    void*  d_bufferMV;
    double beta = 0.0;
    CHECK_HIPSPARSE( hipsparseSpMV_bufferSize(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, d_X.vec, &beta, d_B.vec, HIP_R_64F,
                        HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) )
    CHECK_HIP( hipMalloc(&d_bufferMV, bufferSizeMV) )

    CHECK_HIPSPARSE( hipsparseSpMV(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, d_X.vec, &beta, d_B.vec, HIP_R_64F,
                        HIPSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // X0 = 0
    CHECK_HIP( hipMemset(d_X.ptr, 0x0, m * sizeof(double)) )
    //--------------------------------------------------------------------------
    // Perform Incomplete-LU factorization of A (csrilu0) -> M_lower, M_upper
    csrilu02Info_t infoM        = NULL;
    int            bufferSizeLU = 0;
    void*          d_bufferLU;
    CHECK_HIPSPARSE( hipsparseCreateMatDescr(&matLU) )
    CHECK_HIPSPARSE( hipsparseSetMatType(matLU, HIPSPARSE_MATRIX_TYPE_GENERAL) )
    CHECK_HIPSPARSE( hipsparseSetMatIndexBase(matLU, baseIdx) )
    CHECK_HIPSPARSE( hipsparseCreateCsrilu02Info(&infoM) )

    CHECK_HIPSPARSE( hipsparseDcsrilu02_bufferSize(
                        hipsparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM, &bufferSizeLU) )
    CHECK_HIP( hipMalloc(&d_bufferLU, bufferSizeLU) )
    CHECK_HIPSPARSE( hipsparseDcsrilu02_analysis(
                        hipsparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) )
    int structural_zero;
    CHECK_HIPSPARSE( hipsparseXcsrilu02_zeroPivot(hipsparseHandle, infoM,
                                                &structural_zero) )
    // M = L * U
    CHECK_HIPSPARSE( hipsparseDcsrilu02(
                        hipsparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) )
    // Find numerical zero
    int numerical_zero;
    CHECK_HIPSPARSE( hipsparseXcsrilu02_zeroPivot(hipsparseHandle, infoM,
                                                &numerical_zero) )

    CHECK_HIPSPARSE( hipsparseDestroyCsrilu02Info(infoM) )
    CHECK_HIPSPARSE( hipsparseDestroyMatDescr(matLU) )
    CHECK_HIP( hipFree(d_bufferLU) )
    //--------------------------------------------------------------------------
    // ### Run BiCGStab computation ###
    printf("BiCGStab loop:\n");

    auto start = std::chrono::steady_clock::now();

    double nrm_R = gpu_BiCGStab(verbose, hipblasHandle, hipsparseHandle, m,
                                matA, matM_lower, matM_upper,
                                d_B, d_X, d_R0, d_R, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T,
                                d_tmp, d_bufferMV, maxIterations, tolerance);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total execution time of BiCGStab: %f (s)\n", time * 1e-9f);
    printf("Final error norm = %e\n", nrm_R);
    //--------------------------------------------------------------------------
    // ### Free resources ###
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_B.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_X.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_R.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_R0.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_P.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_P_aux.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_S.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_S_aux.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_V.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_T.vec) )
    CHECK_HIPSPARSE( hipsparseDestroyDnVec(d_tmp.vec) )
    CHECK_HIPSPARSE( hipsparseDestroySpMat(matA) )
    CHECK_HIPSPARSE( hipsparseDestroySpMat(matM_lower) )
    CHECK_HIPSPARSE( hipsparseDestroySpMat(matM_upper) )
    CHECK_HIPSPARSE( hipsparseDestroy(hipsparseHandle) )
    CHECK_HIPBLAS( hipblasDestroy(hipblasHandle) )

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_X);

    CHECK_HIP( hipFree(d_X.ptr) )
    CHECK_HIP( hipFree(d_B.ptr) )
    CHECK_HIP( hipFree(d_R.ptr) )
    CHECK_HIP( hipFree(d_R0.ptr) )
    CHECK_HIP( hipFree(d_P.ptr) )
    CHECK_HIP( hipFree(d_P_aux.ptr) )
    CHECK_HIP( hipFree(d_S.ptr) )
    CHECK_HIP( hipFree(d_S_aux.ptr) )
    CHECK_HIP( hipFree(d_V.ptr) )
    CHECK_HIP( hipFree(d_T.ptr) )
    CHECK_HIP( hipFree(d_tmp.ptr) )
    CHECK_HIP( hipFree(d_A_values) )
    CHECK_HIP( hipFree(d_A_columns) )
    CHECK_HIP( hipFree(d_A_rows) )
    CHECK_HIP( hipFree(d_M_values) )
    CHECK_HIP( hipFree(d_bufferMV) )
    return EXIT_SUCCESS;
}
