#include "morphology.h"


enum class MorphOpType {
    ERODE,
    DILATE,
};


template <MorphOpType opType>
inline __device__ unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
inline __device__ unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
    return min(lhs, rhs);
}

template <>
inline __device__ unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
    return max(lhs, rhs);
}


template <MorphOpType opType>
inline __device__ unsigned char borderValue()
{
}

template <>
inline __device__ unsigned char borderValue<MorphOpType::ERODE>()
{
    return BLACK;
}

template <>
inline __device__ unsigned char borderValue<MorphOpType::DILATE>()
{
    return WHITE;
}


// NOTE: step-efficient parallel scan
template <MorphOpType opType>
__device__ void reversedScan(unsigned char* __restrict__ buffer,
        unsigned char* __restrict__ opArray,
        const int selSize,
        const int tid)
{
    opArray[tid] = buffer[tid];
    __syncthreads();

    for (int offset = 1; offset < selSize; offset *= 2) {
        if (tid <= selSize - 1 - offset) {
            opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
        }
        __syncthreads();
    }
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
__device__ void scan(unsigned char* __restrict__ buffer,
        unsigned char* __restrict__ opArray,
        const int selSize,
        const int tid)
{
    opArray[tid] = buffer[tid];
    __syncthreads();

    for (int offset = 1; offset < selSize; offset *= 2) {
        if (tid >= offset) {
            opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid - offset]);
        }
        __syncthreads();
    }
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
__device__ void twoWayScan(unsigned char* __restrict__ buffer,
        unsigned char* __restrict__ opArray,
        const int selSize,
        const int tid)
{
    opArray[tid] = buffer[tid];
    opArray[tid + selSize] = buffer[tid + selSize];
    __syncthreads();

    for (int offset = 1; offset < selSize; offset *= 2) {
        if (tid >= offset) {
            opArray[tid + selSize - 1] = 
                elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
        }
        if (tid <= selSize - 1 - offset) {
            opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
        }
        __syncthreads();
    }
}


template <MorphOpType opType>
__global__ void vhgw_horiz(unsigned char* __restrict__ dst,
        unsigned char* __restrict__ src,
        const int width,
        const int height,
        const int selSize
        )
{
    HIP_DYNAMIC_SHARED(unsigned char, sMem);
    unsigned char* buffer = sMem;
    unsigned char* opArray = buffer + 2 * selSize;

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx >= width || tidy >= height) {
        return;
    }

    buffer[threadIdx.x] = src[tidy * width + tidx];
    if (tidx + selSize < width) {
        buffer[threadIdx.x + selSize] = src[tidy * width + tidx + selSize];
    }
    __syncthreads();

    twoWayScan<opType>(buffer, opArray, selSize, threadIdx.x);

    if (tidx + selSize/2 < width - selSize/2) {
        dst[tidy * width + tidx + selSize/2] = 
            elementOp<opType>(opArray[threadIdx.x], opArray[threadIdx.x + selSize - 1]);
    }
}

template <MorphOpType opType>
__global__ void vhgw_vert(unsigned char* __restrict__ dst,
        unsigned char* __restrict__ src,
        const int width,
        const int height,
        const int selSize
        )
{
    HIP_DYNAMIC_SHARED(unsigned char, sMem);
    unsigned char* buffer = sMem;
    unsigned char* opArray = buffer + 2 * selSize;

    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy >= height || tidx >= width) {
        return;
    }

    buffer[threadIdx.y] = src[tidy * width + tidx];
    if (tidy + selSize < height) {
        buffer[threadIdx.y + selSize] = src[(tidy + selSize) * width + tidx];
    }
    __syncthreads();

    twoWayScan<opType>(buffer, opArray, selSize, threadIdx.y);

    if (tidy + selSize/2 < height - selSize/2) {
        dst[(tidy + selSize/2) * width + tidx] = 
            elementOp<opType>(opArray[threadIdx.y], opArray[threadIdx.y + selSize - 1]);
    }

    if (tidy < selSize/2 || tidy >= height - selSize/2) {
        dst[tidy * width + tidx] = borderValue<opType>();
    }
}


template <MorphOpType opType>
void morphology(unsigned char* img_d,
        unsigned char* tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    dim3 blockSize;
    blockSize.x = hsize;
    blockSize.y = 1;
    dim3 gridSize;
    gridSize.x = roundUp(width, blockSize.x);
    gridSize.y = roundUp(height, blockSize.y);
    size_t sMemSize = 4 * hsize * sizeof(unsigned char);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vhgw_horiz<opType>), dim3(gridSize), dim3(blockSize), sMemSize, 0, tmp_d, img_d, width, height, hsize);

    blockSize.x = 1;
    blockSize.y = vsize;
    gridSize.x = roundUp(width, blockSize.x);
    gridSize.y = roundUp(height, blockSize.y);
    sMemSize = 4 * vsize * sizeof(unsigned char);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vhgw_vert<opType>), dim3(gridSize), dim3(blockSize), sMemSize, 0, img_d, tmp_d, width, height, vsize);
}


extern "C"
void erode(unsigned char* img_d,
        unsigned char* tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    morphology<MorphOpType::ERODE>(img_d, tmp_d, width, height, hsize, vsize);
}

extern "C"
void dilate(unsigned char* img_d,
        unsigned char* tmp_d,
        const int width,
        const int height,
        const int hsize,
        const int vsize)
{
    morphology<MorphOpType::DILATE>(img_d, tmp_d, width, height, hsize, vsize);
}
