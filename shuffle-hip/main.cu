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

#include <hip/hip_runtime.h>
#include <iostream>

#define BUF_SIZE 256
#define WARP_MASK 0x7
#define WARP_SUM 28
#define PATTERN 0xDEADBEEF

#define HIPCHECK(code)                                                         \
  do {                                                                         \
    hiperr = code;                                                             \
    if (hiperr != hipSuccess) {                                                \
      std::cerr << "ERROR on line " << __LINE__ << ": " << (unsigned)hiperr    \
                << "\n";                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

//==================================================================================
// Broadcast
//==================================================================================
__global__ void bcast_shfl(const int arg, int *out) {
  int value = ((hipThreadIdx_x & WARP_MASK) == 0) ? arg : 0;

  int out_v = __shfl(
      value, 0); // Synchronize all threads in warp, and get "value" from lane 0

  size_t oi = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  out[oi] = out_v;
}

__global__ void bcast_shfl_xor(int *out) {
  int value = (hipThreadIdx_x & WARP_MASK);

  for (int mask = 1; mask < WARP_MASK; mask *= 2)
    value += __shfl_xor(value, mask);

  size_t oi = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  out[oi] = value;
}

//==================================================================================
// Matrix transpose
//==================================================================================
__global__ void transpose_shfl(float* out, const float* in) {
  unsigned b_start = hipBlockDim_x * hipBlockIdx_x;
  unsigned b_offs = b_start + hipThreadIdx_x;
  unsigned s_offs = hipBlockDim_x - hipThreadIdx_x - 1;

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

  hipError_t hiperr = hipSuccess;
  size_t errors = 0;

  std::cout << "Broadcast using shuffle functions\n";

  int *out = (int *)malloc(sizeof(int) * BUF_SIZE);

  int *d_out;
  HIPCHECK(hipMalloc((void **)&d_out, sizeof(int) * BUF_SIZE));
  hipLaunchKernelGGL(bcast_shfl_xor, dim3(1), dim3(BUF_SIZE), 0, 0, d_out);
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipMemcpy(out, d_out, sizeof(int) * BUF_SIZE, hipMemcpyDeviceToHost));

  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != WARP_SUM) {
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
    }
  }

  hipLaunchKernelGGL(bcast_shfl, dim3(1), dim3(BUF_SIZE), 0, 0, PATTERN, d_out);
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipMemcpy(out, d_out, sizeof(int) * BUF_SIZE, hipMemcpyDeviceToHost));

  for (int i = 0; i < BUF_SIZE; i++) {
    if (out[i] != PATTERN) {
      std::cout << "ERROR @ " << i << ":  " << out[i] << "\n";
      ++errors;
    }
  }

  free(out);
  HIPCHECK(hipFree(d_out));

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
  HIPCHECK(hipMalloc((void **)&gpuMatrix, total * sizeof(float)));
  HIPCHECK(hipMalloc((void **)&gpuTransposeMatrix, total * sizeof(float)));

  // Memory transfer from host to device
  HIPCHECK(hipMemcpy(gpuMatrix, Matrix, total * sizeof(float),
        hipMemcpyHostToDevice));

  // Lauching kernel from host
  hipLaunchKernelGGL(transpose_shfl, dim3(numGroups), dim3(subGroupSize), 0, 0,
      gpuTransposeMatrix, gpuMatrix);
  HIPCHECK(hipGetLastError());

  // Memory transfer from device to host
  HIPCHECK(hipMemcpy(TransposeMatrix, gpuTransposeMatrix,
        total * sizeof(float), hipMemcpyDeviceToHost));

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
  HIPCHECK(hipFree(gpuMatrix));
  HIPCHECK(hipFree(gpuTransposeMatrix));

  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return 0;
}
