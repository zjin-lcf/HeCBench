/*
Kernels for matmul forward pass.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
__global__ void matmul_forward_kernel1(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc * C;
        const float* inp_bt = inp + bt * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but i can't seem to find a hipBLAS function that does this
__global__ void add_bias(float* out, const float* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// kernel 4: semi-efficient handwritten kernel
// see trimat_forward.cu for some intermediate development steps
__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

__global__ void __launch_bounds__(16*16) 
matmul_forward_kernel4(float* out, const float* inp,
                       const float* weight, const float* bias, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    // buffers to cache chunks of the input matrices
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // adjust our pointers for the current block
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[8][8] = {};
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    int si_start = 4*(16 * threadIdx.y + threadIdx.x);
    for (int so = 0; so < C; so += 32) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        for (int si = si_start; si < si_start + 32; si += 4) {
            float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int sqrt_block_size = 16;
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    hipCheck(hipDeviceSynchronize());
}

// kernel 2 calls hipBLAS, which should be very efficient
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    // for reference API is:
    // hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,
    //                        hipblasOperation_t transa, hipblasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // hipBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, hipBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because hipBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get hipBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call hipBLAS with A = weight, B = inp
    // => need to call hipBLAS with transa = HIPBLAS_OP_T, transb = HIPBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int sqrt_block_size = 16;
    hipblasCheck(hipblasSgemm(hipblas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));
    // and now we still have to add the bias... (ew)
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
    }
    hipCheck(hipDeviceSynchronize());
}

// uses hipblasLt to fuse the bias and gelu
void matmul_forward3(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);
    int has_gelu = 0;

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (hipBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    hipblasLtMatmulDesc_t operationDesc;
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatrixLayout_t weightLayout;
    hipblasLtMatrixLayout_t inputLayout;
    hipblasLtMatrixLayout_t outputLayout;
    hipblasLtMatrixLayout_t biasLayout;
    hipblasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    hipblasOperation_t opNoTranspose = HIPBLAS_OP_N;
    hipblasOperation_t opTranspose = HIPBLAS_OP_T;
    hipblasLtEpilogue_t epilogueBias = HIPBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = HIPBLASLT_EPILOGUE_GELU_BIAS;
    } else if (has_bias) {
        epilogueBias = HIPBLASLT_EPILOGUE_BIAS;
    } else if (has_gelu) {
        epilogueBias = HIPBLASLT_EPILOGUE_GELU;
    }
    hipblasCheck(hipblasLtMatmulDescCreate(&operationDesc, hipblas_compute_type, HIP_R_32F));
    hipblasCheck(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    hipblasCheck(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
    hipblasCheck(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    hipblasCheck(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    hipblasCheck(hipblasLtMatrixLayoutCreate(&weightLayout, HIP_R_32F, C, OC, C));
    hipblasCheck(hipblasLtMatrixLayoutCreate(&inputLayout, HIP_R_32F, C, B*T, C));
    hipblasCheck(hipblasLtMatrixLayoutCreate(&outputLayout, HIP_R_32F, OC, B*T, OC));
    hipblasCheck(hipblasLtMatrixLayoutCreate(&biasLayout, HIP_R_32F, OC, 1, OC));

    // create a preference handle with specified max workspace
    hipblasCheck(hipblasLtMatmulPreferenceCreate(&preference));
    hipblasCheck(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &hipblaslt_workspace_size, sizeof(hipblaslt_workspace_size)));

    // find a suitable algorithm
    hipblasCheck(hipblasLtMatmulAlgoGetHeuristic(hipblaslt_handle, operationDesc,
        weightLayout, inputLayout, outputLayout, outputLayout,
        preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No hipBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
            B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    const float alpha = 1.0f, beta = 0.0f;
    hipblasCheck(hipblasLtMatmul(hipblaslt_handle, operationDesc,
        &alpha, weight, weightLayout, inp, inputLayout, &beta,
        out, outputLayout, out, outputLayout, &heuristic.algo,
        hipblaslt_workspace, hipblaslt_workspace_size, 0));

    // wait for matmul
    hipCheck(hipDeviceSynchronize());

    // cleanups
    hipblasCheck(hipblasLtMatmulPreferenceDestroy(preference));
    hipblasCheck(hipblasLtMatmulDescDestroy(operationDesc));
    hipblasCheck(hipblasLtMatrixLayoutDestroy(weightLayout));
    hipblasCheck(hipblasLtMatrixLayoutDestroy(inputLayout));
    hipblasCheck(hipblasLtMatrixLayoutDestroy(outputLayout));
    hipblasCheck(hipblasLtMatrixLayoutDestroy(biasLayout));
}

// handwritten, relatively efficient non-tensorcore matmul kernel
void matmul_forward4(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    int sqrt_block_size = 16;

    dim3 gridDim(ceil_div(B * T, 8*sqrt_block_size), ceil_div(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    hipCheck(hipDeviceSynchronize());
}

// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC);
            break;
        case 3:
            matmul_forward3(out, inp, weight, bias, B, T, C, OC);
            break;
        case 4:
            matmul_forward4(out, inp, weight, bias, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 4;
    int T = 1024;
    int C = 768;
    int OC = 768 * 3;

    // set up the device
    int deviceIdx = 0;
    hipCheck(hipSetDevice(deviceIdx));
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup hipBLAS and hipBLASLt
    hipblasCheck(hipblasCreate(&hipblas_handle));
    hipblasCheck(hipblasLtCreate(&hipblaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = 0;
    printf("enable_tf32: %d\n", enable_tf32);
    hipblas_compute_type = enable_tf32 ? HIPBLAS_COMPUTE_32F_FAST_TF32 : HIPBLAS_COMPUTE_32F;
    hipblasMath_t hipblas_math_mode = enable_tf32 ? HIPBLAS_TF32_TENSOR_OP_MATH : HIPBLAS_DEFAULT_MATH;
    hipblasCheck(hipblasSetMathMode(hipblas_handle, hipblas_math_mode));
    // setup the (global) hipBLASLt workspace
    hipCheck(hipMalloc(&hipblaslt_workspace, hipblaslt_workspace_size));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    hipCheck(hipMalloc(&d_out, B * T * OC * sizeof(float)));
    hipCheck(hipMalloc(&d_inp, B * T * C * sizeof(float)));
    hipCheck(hipMalloc(&d_weight, C * OC * sizeof(float)));
    hipCheck(hipMalloc(&d_bias, OC * sizeof(float)));
    hipCheck(hipMemcpy(d_inp, inp, B * T * C * sizeof(float), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_weight, weight, C * OC * sizeof(float), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_bias, bias, OC * sizeof(float), hipMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC);
    validate_result(d_out, out, "out", B * T * OC, 1e-1f);

    printf("All results match. Starting benchmarks.\n\n");

    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                          kernel_num, d_out, d_inp, d_weight, d_bias,
                                          B, T, C, OC);

    float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
    printf("time %.4f ms | tflops %.2f\n", elapsed_time, tflops);

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    hipCheck(hipFree(d_out));
    hipCheck(hipFree(d_inp));
    hipCheck(hipFree(d_weight));
    hipCheck(hipFree(d_bias));
    hipCheck(hipFree(hipblaslt_workspace));
    hipblasCheck(hipblasDestroy(hipblas_handle));
    hipblasCheck(hipblasLtDestroy(hipblaslt_handle));
    return 0;
}
