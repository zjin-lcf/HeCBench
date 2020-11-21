#include <cstdlib>
#include <cstdio>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
#define SLICE_SIZE 784


// A C model derived from the OpenCL kernel 
void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

__global__ void 
softMax (const int numSlice, const int sliceSize, const float* src, float* dest) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += exp(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] = exp(src[i * sliceSize + j] - max_) / sum;
  }
}

int main() {
   
  int numSlice = 10000;
  int sliceSize = SLICE_SIZE;
  int numElem = numSlice * sliceSize;

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_cpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i*sliceSize+j] = rand() % 13; 

  float *d_input, *d_output;
  hipMalloc((void**)&d_input, sizeof(float) * numElem);
  hipMalloc((void**)&d_output, sizeof(float) * numElem);
  hipMemcpy(d_input, input, sizeof(float) * numElem, hipMemcpyHostToDevice);

  dim3 global_work_size ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  dim3 local_work_size (BLOCK_SIZE);

  for (int n = 0; n < 100; n++) {
    hipLaunchKernelGGL(softMax, global_work_size, local_work_size, 0, 0, numSlice, sliceSize, d_input, d_output);
  }

  hipMemcpy(output_gpu, d_output, sizeof(float) * numElem, hipMemcpyDeviceToHost);

  // verification
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-3) {
      printf("@index %d cpu: %f gpu: %f\n", i, output_cpu[i], output_gpu[i]);
      break;
    }
  }

  free(input);
  free(output_cpu);
  free(output_gpu);
  hipFree(d_input);
  hipFree(d_output);
  return 0;
}

