#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include "common.h"
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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_image (numPix);

  int blocks = std::min(PowTwoDivider(height), 64);
  range<1> lwsX (blocks);
  range<1> gwsX ((height + blocks-1) / blocks * blocks);

  blocks = std::min(PowTwoDivider(width), 64);
  range<1> lwsY (blocks);
  range<1> gwsY ((width + blocks-1) / blocks * blocks);

  double total_time = 0.0;
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto acc = d_image.get_access<sycl_discard_write>(cgh);
      cgh.copy(image, acc);
    }).wait();

    auto start = std::chrono::steady_clock::now();
    q.submit([&] (handler &cgh) {
      auto img = d_image.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class convertX>(nd_range<1>(gwsX, lwsX), [=] (nd_item<1> item) {
        toCoef2DX(item, img.get_pointer(), image_pitch, width, height);
      });
    });

    q.submit([&] (handler &cgh) {
      auto img = d_image.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class convertY>(nd_range<1>(gwsY, lwsY), [=] (nd_item<1> item) {
        toCoef2DY(item, img.get_pointer(), image_pitch, width, height);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  printf("Average kernel execution time %f (s)\n", total_time * 1e-9f / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_image.get_access<sycl_read>(cgh);
    cgh.copy(acc, image);
  }).wait();

  float sum = 0.f;
  for (int i = 0; i < numPix; i++) {
    const uchar *t = (const uchar*)(&image[i]);
    sum += (t[0] + t[1] + t[2] + t[3]) / 4;
  }
  printf("Checksum: %f\n", sum / numPix);

  free(image);
  return 0;
}
