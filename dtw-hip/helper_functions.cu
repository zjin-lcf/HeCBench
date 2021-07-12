/** Helper functions shared across CUDA kernels
 *  @file helper_functions.cu
 *  @author Alex Kyllo
 *  @date 2021-03
 */
#include "helper_functions.cuh"


/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */
__global__ void fill_matrix_inf(float *A, uint width, uint height, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x)
    {
        A[i] = val;
        if (i % width == 0) A[i] = 0.0;
    }
}
