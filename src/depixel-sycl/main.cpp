#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "kernels.h"

#define nthreads 256

int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  int size = width * height;
  size_t size_output_bytes = size * sizeof(uint);
  size_t size_image_bytes = size * sizeof(sycl::float3);

  std::mt19937 gen(19937);
  // reduce the upper bound can increase the kernel execution time of eliminate_crosses
  std::uniform_real_distribution<float> dis(0.f, 0.4f); 

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::float3 *h_img = (sycl::float3*) malloc(size_image_bytes);

  uint *h_out = (uint*) malloc(size_output_bytes);

  sycl::float3 *d_img = sycl::malloc_device<sycl::float3>(size, q);

  uint *d_tmp = sycl::malloc_device<uint>(size, q);

  uint *d_out = sycl::malloc_device<uint>(size, q);

  // assume that size is a multiple of nthreads
  sycl::range<1> gws (size);
  sycl::range<1> lws (nthreads);

  float sum = 0;
  float total_time = 0;

  for (int n = 0; n < repeat; n++) {

    for (int i = 0; i < size; i++) {
      h_img[i].x() = dis(gen);
      h_img[i].y() = dis(gen);
      h_img[i].z() = dis(gen);
    }

    q.memcpy(d_img, h_img, size_image_bytes).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class check>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        check_connect(item, d_img, d_tmp, width, height);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class remove>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        eliminate_crosses(item, d_tmp, d_out, width, height);
      });
    });
  
    q.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    total_time += time.count();

    q.memcpy(h_out, d_out, size_output_bytes).wait();

    float lsum = 0;
    for (int i = 0; i < size; i++)
      lsum += (h_out[i] & 0xff) / 256.f + 
             ((h_out[i] >> 8) & 0xff) / 256.f + 
             ((h_out[i] >> 16) & 0xff) / 256.f + 
             ((h_out[i] >> 24) & 0xff) / 256.f;

    sum += lsum / size;
  }

  printf("Image size: %d (width) x %d (height)\ncheckSum: %f\n",
         width, height, sum);
  printf("Average kernel time over %d iterations: %f (s)\n",
         repeat, total_time / repeat);

  sycl::free(d_out, q);
  sycl::free(d_img, q);
  sycl::free(d_tmp, q);
  free(h_out);
  free(h_img);

  return 0;
}
