#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#define GPU_CHECK(ans)                                                                  \
    {                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                            \
    }
inline void
gpuAssert(hipError_t code, const char* file, int line, bool abort = true)
{
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

hipblasLtHandle_t handle;

template <typename scalar_t>
int mlp_gemm_lt(
    hipblasLtHandle_t ltHandle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    float *alpha, /* host pointer */
    const scalar_t* A,
    int lda,
    const scalar_t* B,
    int ldb,
    float *beta, /* host pointer */
    scalar_t* C,
    int ldc,
    void *workspace,
    size_t workspaceSize,
    hipStream_t stream,
    bool use_bias,
    bool use_relu,
    const void* bias) {

  hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

  hipblasLtMatmulDesc_t operationDesc = {};
  hipblasLtMatrixLayout_t Adesc = {}, Bdesc = {}, Cdesc = {};
  hipblasLtMatmulPreference_t preference = {};

  int returnedResults                             = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult = {};
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  // Create operation descriptor; see hipblasLtMatmulDescAttributes_t
  // for details about defaults; here we just set the transforms for
  // A and B.
  //status = hipblasLtMatmulDescInit(&operationDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  status = hipblasLtMatmulDescCreate(&operationDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (use_bias) {
    status = hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != HIPBLAS_STATUS_SUCCESS) {
      goto CLEANUP;
    }
    if (use_relu) {
      epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_BIAS;
    }
  } else {
    if (use_relu) {
      epilogue = HIPBLASLT_EPILOGUE_RELU;
    }
  }

  status = hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    goto CLEANUP;
  }

  // Initialize matrix descriptors. Not setting any extra attributes.
  status = hipblasLtMatrixLayoutCreate(
    &Adesc, HIP_R_32F, transa == HIPBLAS_OP_N ? m : k, transa == HIPBLAS_OP_N ? k : m, lda);
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = hipblasLtMatrixLayoutCreate(
    &Bdesc, HIP_R_32F, transb == HIPBLAS_OP_N ? k : n, transb == HIPBLAS_OP_N ? n : k, ldb);
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_32F, m, n, ldc);
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;

  // Create preference handle; In general, extra attributes can be
  // used here to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C. However, for simplicity
  // here we assume A,B,C are always well aligned (e.g., directly
  // come from hipMalloc)
  status = hipblasLtMatmulPreferenceCreate(&preference);
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;
  status = hipblasLtMatmulPreferenceSetAttribute(
    preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;

  // We just need the best available heuristic to try and run matmul.
  // There is no guarantee that this will work. For example, if A is
  // badly aligned, you can request more (e.g. 32) algos and try to
  // run them one by one until something works.
  status = hipblasLtMatmulAlgoGetHeuristic(
    ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
  if (status != HIPBLAS_STATUS_SUCCESS) goto CLEANUP;

  if (returnedResults == 0) {
    status = HIPBLAS_STATUS_NOT_SUPPORTED;
    goto CLEANUP;
  }
  status = hipblasLtMatmul(ltHandle,
                           operationDesc,
                           alpha,
                           A,
                           Adesc,
                           B,
                           Bdesc,
                           beta,
                           C,
                           Cdesc,
                           C,
                           Cdesc,
                           &heuristicResult.algo,
                           workspace,
                           workspaceSize,
                           stream);

CLEANUP:
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : 1;
}


// Does a simple MLP fprop (GEMM+bias).
// Can handle num_layers number of layers, each with its own shape. Output of layer i is assumed
// to be input of layer i+1. output_features, WPtr and BPtr are arrays of length num_layers, and
// must be in the same order i.e. WPtr[i] and BPtr[i] are respectively the weight and bias of layer
// 'i'.
template <typename T>
int mlp_fp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T** BPtr,
    T* Y,
    T* reserved_space,
    int use_bias,
    int use_relu,
    void* lt_workspace)
{
  T *weight, *input, *output, *bias = nullptr;
  T *reserved_space_x, *reserved_space_y;
  reserved_space_x = NULL;
  reserved_space_y = reserved_space;

  for (int layer = 0; layer < num_layers; layer++) {
    weight = WPtr[layer];
    input = (layer == 0) ? X : reserved_space_x;
    output = (layer == num_layers - 1) ? Y : reserved_space_y;
    if (use_bias) {
      bias = BPtr[layer];
    }
    int ifeat = (layer == 0) ? input_features : output_features[layer - 1];
    int ofeat = output_features[layer];

    float one = 1.f;
    float zero = 0.f;

    int hipblaslt_status;
    hipblaslt_status = mlp_gemm_lt(
      handle,
      HIPBLAS_OP_T,
      HIPBLAS_OP_N,
      ofeat,
      batch_size,
      ifeat,
      &one,
      weight,
      ifeat,
      input,
      ifeat,
      &zero,
      output,
      ofeat,
      lt_workspace,
      1 << 22,
      0, //stream,
      use_bias == 1,
      use_relu == 1,
      bias);
    if (hipblaslt_status) return 1;

    // Set current output as next layer input
    reserved_space_x = reserved_space_y;
    // Set next layer output
    reserved_space_y += ofeat * batch_size;
  }

  return 0;
}

/*
template int mlp_fp<half>(
    half* X,
    int input_features,
    int batch_size,
    half** WPtr,
    int num_layers,
    int* output_features,
    half** BPtr,
    half* Y,
    half* reserved_space,
    int use_bias,
    int use_relu,
    void* lt_workspace);
*/

template int mlp_fp<float>(
    float* X,
    int input_features,
    int batch_size,
    float** WPtr,
    int num_layers,
    int* output_features,
    float** BPtr,
    float* Y,
    float* reserved_space,
    int use_bias,
    int use_relu,
    void* lt_workspace);

