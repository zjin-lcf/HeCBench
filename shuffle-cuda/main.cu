/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <cuda.h>
#include <iostream>

#define BUF_SIZE 256
#define WARP_MASK 0x7
#define WARP_SUM 28
#define PATTERN 0xDEADBEEF

#define CUDACHECK(code)                                                         \
  do {                                                                         \
    cudaerr = code;                                                             \
    if (cudaerr != cudaSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)cudaerr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

//==================================================================================
// Broadcast
//==================================================================================
__global__ void bcast_shfl(const int arg, int *out) {
  int value = ((threadIdx.x & WARP_MASK) == 0) ? arg : 0;

  int out_v = __shfl(
      value, 0); // Synchronize all threads in warp, and get "value" from lane 0

  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;
  out[oi] = out_v;
}

__global__ void bcast_shfl_xor(int *out) {
  int value = (threadIdx.x & WARP_MASK);

  for (int mask = 1; mask < WARP_MASK; mask *= 2)
    value += __shfl_xor(value, mask);

  size_t oi = blockDim.x * blockIdx.x + threadIdx.x;

  out[oi] = value;
}

//==================================================================================
// Matrix transpose
//==================================================================================
__global__ void transpose_shfl(float* out, const float* in) {
  unsigned b_start = blockDim.x * blockIdx.x;
  unsigned b_offs = b_start + threadIdx.x;
  unsigned s_offs = blockDim.x - threadIdx.x - 1;

  float val = in[b_offs];
  out[b_offs] = __shfl(val, s_offs);
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, 
    unsigned int numGroups, unsigned int subGroupSize) {
  for (unsigned i = 0; i < numGroups; ++i) {
    for (unsigned j = 0; j < subGroupSize; j++) {
      output[i * subGroupSize + j] = input[i * subGroupSize + subGroupSize - j - 1];
    }
  }
}

int main() {

  cudaError_t cudaerr = cudaSuccess;
  size_t errors = 0;

  std::cout << "Broadcast using shuffle functions\n";

  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);

  int *d_out;
  CUDACHECK(cudaMalloc((void **)&d_out, sizeof(int) * BUF_SIZE));
  bcast_shfl_xor <<< dim3(1), dim3(BUF_SIZE) >>> (d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost));

  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != WARP_SUM) {
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
    }
  }

  bcast_shfl <<< dim3(1), dim3(BUF_SIZE) >>> (PATTERN, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaMemcpy(out, d_out, sizeof(int) * BUF_SIZE, cudaMemcpyDeviceToHost));

  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != PATTERN) {
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
    }
  }

  free(out);
  CUDACHECK(cudaFree(d_out));

  if (errors != 0) {
    std::cout << "FAILED: " << errors << " errors\n";
  } else {
    std::cout << "PASSED\n";
  }

  std::cout << "matrix transpose using shuffle functions\n";

  const int numGroups = 8;
  const int subGroupSize = 8;
  const int total = numGroups * subGroupSize;

  float* Matrix = (float*)malloc(total * sizeof(float));
  float* TransposeMatrix = (float*)malloc(total * sizeof(float));
  float* cpuTransposeMatrix = (float*)malloc(total * sizeof(float));

  // initialize the input data
  for (int i = 0; i < total; i++) {
    Matrix[i] = (float)i * 10.0f;
  }

  float *gpuMatrix;
  float *gpuTransposeMatrix;
  // allocate the memory on the device side
  CUDACHECK(cudaMalloc((void **)&gpuMatrix, total * sizeof(float)));
  CUDACHECK(cudaMalloc((void **)&gpuTransposeMatrix, total * sizeof(float)));

  // Memory transfer from host to device
  CUDACHECK(cudaMemcpy(gpuMatrix, Matrix, total * sizeof(float),
        cudaMemcpyHostToDevice));

  // Lauching kernel from host
  transpose_shfl <<< dim3(numGroups), dim3(subGroupSize) >>> (
      gpuTransposeMatrix, gpuMatrix);
  CUDACHECK(cudaGetLastError());

  // Memory transfer from device to host
  CUDACHECK(cudaMemcpy(TransposeMatrix, gpuTransposeMatrix,
        total * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, numGroups, subGroupSize);

  // verify the results
  errors = 0;
  float eps = 1.0E-6;
  for (int i = 0; i < total; i++) {
    if (std::fabs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      std::cout << "ITEM: " << i <<
        " cpu: " << cpuTransposeMatrix[i] <<
        " gpu: " << TransposeMatrix[i] << "\n";
      errors++;
    }
  }

  if (errors > 0) {
    std::cout << "FAIL: " << errors << " errors \n";
  }
  else {
    std::cout << "PASSED\n";
  }

  // free the resources
  CUDACHECK(cudaFree(gpuMatrix));
  CUDACHECK(cudaFree(gpuTransposeMatrix));

  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return 0;
}
