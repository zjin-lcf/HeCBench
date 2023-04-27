#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

int PowTwoDivider(int n)
{
  if (n == 0) return 0;
  int divider = 1;
  while ((n & divider) == 0) divider <<= 1;
  return divider;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int image_pitch = width * sizeof(float);
  const int numPix = width * height;
  const int image_size = numPix * sizeof(float);

  float *image = (float*) malloc (image_size);

  // image image with random values
  srand(123);
  for (int i = 0; i < numPix; i++) {
    uint x = rand() % 256;
    uint y = rand() % 256;
    uint z = rand() % 256;
    uint w = rand() % 256;
    *(uint*)(&image[i]) = (w << 24) | (z << 16) | (y << 8) | x;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_image = sycl::malloc_device<float>(numPix, q);

  int blocks = std::min(PowTwoDivider(height), 64);
  sycl::range<1> lwsX (blocks);
  sycl::range<1> gwsX ((height + blocks-1) / blocks * blocks);

  blocks = std::min(PowTwoDivider(width), 64);
  sycl::range<1> lwsY (blocks);
  sycl::range<1> gwsY ((width + blocks-1) / blocks * blocks);

  long total_time = 0;
  for (int i = 0; i < repeat; i++) {
    q.memcpy(d_image, image, image_size).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class convertX>(
        sycl::nd_range<1>(gwsX, lwsX), [=] (sycl::nd_item<1> item) {
        toCoef2DX(item, d_image, image_pitch, width, height);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class convertY>(
        sycl::nd_range<1>(gwsY, lwsY), [=] (sycl::nd_item<1> item) {
        toCoef2DY(item, d_image, image_pitch, width, height);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  printf("Average kernel execution time %f (s)\n", total_time * 1e-9f / repeat);

  q.memcpy(image, d_image, image_size).wait();
  sycl::free(d_image, q);

  float sum = 0.f;
  for (int i = 0; i < numPix; i++) {
    const sycl::uchar *t = (const sycl::uchar*)(&image[i]);
    sum += (t[0] + t[1] + t[2] + t[3]) / 4;
  }
  printf("Checksum: %f\n", sum / numPix);

  free(image);
  return 0;
}
