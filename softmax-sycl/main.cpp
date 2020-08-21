#include <cstdlib>
#include <cstdio>
#include "common.h"

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

  {
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_input (input, numElem);
  buffer<float, 1> d_output (output_gpu, numElem);

  range<1> global_work_size ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  range<1> local_work_size (BLOCK_SIZE);

  for (int n = 0; n < 100; n++) {
    q.submit([&](handler &h) {
      auto src = d_input.get_access<sycl_read>(h);
      auto dest = d_output.get_access<sycl_discard_write>(h);
      h.parallel_for(nd_range<1>(global_work_size, local_work_size), [=](nd_item<1> item) {
        int i = item.get_global_id(0);
	if (i >= numSlice) return;
        float max_ = src[i * sliceSize];
        for (int j = 0; j < sliceSize; j++) {
          max_ = cl::sycl::max(max_, src[i * sliceSize + j]);
        }
        float sum = 0;
        for (int j = 0; j < sliceSize; j++) {
          sum += cl::sycl::exp(src[i * sliceSize + j] - max_);
        }
        for (int j = 0; j < sliceSize; j++) {
          dest[i * sliceSize + j] = cl::sycl::exp(src[i * sliceSize + j] - max_) / sum;
        }
      });
    });
  }
  q.wait();

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

