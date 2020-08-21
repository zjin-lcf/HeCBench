#include <cstdlib>
#include <cstdio>
#include <cmath>

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

#pragma omp target data map(to: input[0:numElem]) map(from: output_gpu[0:numElem])
{
  for (int n = 0; n < 100; n++) {
    #pragma omp target teams distribute parallel for simd thread_limit(BLOCK_SIZE)
    for (int i = 0; i < numSlice; i++) {
      float max_ = input[i * sliceSize];
      for (int j = 0; j < sliceSize; j++) {
        max_ = (max_ < input[i * sliceSize + j]) ? input[i * sliceSize + j] : max_;
      }
      float sum = 0;
      for (int j = 0; j < sliceSize; j++) {
        sum += expf(input[i * sliceSize + j] - max_);
      }
      for (int j = 0; j < sliceSize; j++) {
        output_gpu[i * sliceSize + j] = expf(input[i * sliceSize + j] - max_) / sum;
      }
    }
  }
}

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
  return 0;

}

