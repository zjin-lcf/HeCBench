#include <chrono>
#include "helper.h"

/// Sample wrapper executing fp8 matmul with cublasLtMatmul, with addition of per-tensor scaling, amax calculations, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtFp8Matmul(const int repeat,
                 cublasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *beta, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const float *c_scale, /* device pointer */
                 const __nv_bfloat16 *C,
                 int ldc,
                 const float *d_scale, /* device pointer */
                 __nv_fp8_e4m3 *D,
                 //__nv_bfloat16 *D,
                 float *amax_d, /* device pointer */
                 void *workspace,
                 size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // set scaling factors
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

    // set fast accumulation
    int8_t fast_accum = 1;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));
    //checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        return;
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha, A, Adesc,
                                     B, Bdesc, beta,
                                     C, Cdesc,
                                     D, Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns_fp8 = (time / repeat);
    printf("Average cublasLtMatmul execution time %10.3f (us) | ", ns_fp8 * 1e-3f);
    printf("Average cublasLtMatmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns_fp8 * 1e-3f);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


int main(int argc, char *argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);

   const int shapes[5][3] = {{16384, 8192, 1280},
                             {16384, 1024, 8192},
                             {16384, 8192, 7168},
                             {16384, 3584, 8192},
                             {8192, 8192, 8192}};

   for (int i = 0; i < 5; i++) {

     int m = shapes[i][0], n = shapes[i][1], k = shapes[i][2];
     printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

     TestBench<__nv_fp8_e4m3,
               __nv_bfloat16, // C type
               __nv_fp8_e4m3, // output type
               //__nv_bfloat16, // output type
               float> props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

     props.run([&props, repeat] {
          LtFp8Matmul(repeat,
                      props.ltHandle,
                      props.m,
                      props.n,
                      props.k,
                      &props.alpha,
                      &props.beta,
                      props.AscaleDev, props.Adev, props.k, // mxk
                      props.BscaleDev, props.Bdev, props.k, // kxn
                      props.CscaleDev, props.Cdev, props.m, // mxn
                      props.DscaleDev, props.Ddev,
                      props.DamaxDev,
                      props.workspace,
                      props.workspaceSize);
      });
    }

    return 0;
}
