#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include "common.h"
#include "kernels.h"

#define nthreads 256

int main(int argc, char** argv) {

  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  size_t size = width * height;
  size_t size_output_bytes = size * sizeof(uint);
  size_t size_image_bytes = size * sizeof(float3);

  std::mt19937 gen(19937);
  // reduce the upper bound can increase the kernel execution time of eliminate_crosses
  std::uniform_real_distribution<float> dis(0.f, 0.4f); 

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  float3 *h_img = (float3*) malloc(size_image_bytes);

  uint *h_out = (uint*) malloc(size_output_bytes);

  float3 *d_img = sycl::malloc_device<float3>(size, q);

  uint *d_tmp = sycl::malloc_device<uint>(size, q);

  uint *d_out = sycl::malloc_device<uint>(size, q);

  // assume that size is a multiple of nthreads
  range<1> gws (size);
  range<1> lws (nthreads);

  float sum = 0;
  float total_time = 0;

  for (int n = 0; n < repeat; n++) {

    for (size_t i = 0; i < size; i++) {
      h_img[i].x() = dis(gen);
      h_img[i].y() = dis(gen);
      h_img[i].z() = dis(gen);
    }

    auto start = std::chrono::steady_clock::now();

    auto e_h2d = q.memcpy(d_img, h_img, size_image_bytes);
    auto e_k1 = q.submit([&] (handler &cgh) {
      cgh.depends_on(e_h2d);
      cgh.parallel_for<class check>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        check_connect(item, d_img, d_tmp, width, height);
      });
    });

    auto e_k2 = q.submit([&] (handler &cgh) {
      cgh.depends_on(e_k1);
      cgh.parallel_for<class remove>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        eliminate_crosses(item, d_tmp, d_out, width, height);
      });
    });
  
    q.memcpy(h_out, d_out, size_output_bytes, e_k2).wait();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    total_time += time.count();

    float lsum = 0;
    for (size_t i = 0; i < size; i++)
      lsum += (h_out[i] & 0xff) / 256.f + 
             ((h_out[i] >> 8) & 0xff) / 256.f + 
             ((h_out[i] >> 16) & 0xff) / 256.f + 
             ((h_out[i] >> 24) & 0xff) / 256.f;

    sum += lsum / size;
  }

  printf("Image size: %d (width) x %d (height)\ncheckSum: %f\n",
         width, height, sum);
  printf("Average device time over %d iterations: %f (s)\n",
         repeat, total_time / repeat);

  sycl::free(d_out, q);
  sycl::free(d_img, q);
  sycl::free(d_tmp, q);
  free(h_out);
  free(h_img);

  return 0;
}
