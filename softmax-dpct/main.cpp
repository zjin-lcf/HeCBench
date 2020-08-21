#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

void 
softMax (const int numSlice, const int sliceSize, const float* src, float* dest,
         sycl::nd_item<3> item_ct1) {
  unsigned i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = sycl::max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += sycl::exp(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] = sycl::exp(src[i * sliceSize + j] - max_) / sum;
  }
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

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
  d_input = sycl::malloc_device<float>(numElem, q_ct1);
  d_output = sycl::malloc_device<float>(numElem, q_ct1);
  q_ct1.memcpy(d_input, input, sizeof(float) * numElem).wait();

  sycl::range<3> global_work_size(
      (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  sycl::range<3> local_work_size(BLOCK_SIZE, 1, 1);

  for (int n = 0; n < 100; n++) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto dpct_global_range = global_work_size * local_work_size;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(local_work_size.get(2), local_work_size.get(1),
                             local_work_size.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            softMax(numSlice, sliceSize, d_input, d_output, item_ct1);
          });
    });
  }

  q_ct1.memcpy(output_gpu, d_output, sizeof(float) * numElem).wait();

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
  sycl::free(d_input, q_ct1);
  sycl::free(d_output, q_ct1);
  return 0;

}

