#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

float sigmoid (float x) {
  return (1.f / (1.f + sycl::exp(-x)));
}

void parallelPitched2DAccess (sycl::nd_item<2> &item, char* devPtr,
                              size_t pitch,int width, int height)
{
  int r = item.get_global_id(0);
  int c = item.get_global_id(1);
  if (r < height && c < width) {
    float* row = (float*)(devPtr + r * pitch);
    row[c] = sigmoid(row[c]);
  }
}

void parallelSimple2DAccess (sycl::nd_item<2> &item, float* elem, int width, int height)
{
  int r = item.get_global_id(0);
  int c = item.get_global_id(1);
  if (r < height && c < width) {
    elem[r * width + c] = sigmoid(elem[r * width + c]);
  }
}

void parallelPitched3DAccess (sycl::nd_item<3> &item, char* devPtr, int pitch,
                              int width, int height, int depth)
{
  int z = item.get_global_id(0);
  int y = item.get_global_id(1);
  int x = item.get_global_id(2);
  if (z < depth && y < height && x < width) {
    size_t slicePitch = pitch * height;
    char* slice = devPtr + z * slicePitch;
    float* row = (float*)(slice + y * pitch);
    row[x] = sigmoid(row[x]);
  }
}

void parallelSimple3DAccess (sycl::nd_item<3> &item, float* elem,
                             int width, int height, int depth)
{
  int z = item.get_global_id(0);
  int y = item.get_global_id(1);
  int x = item.get_global_id(2);
  if (z < depth && y < height && x < width) {
    float element = elem[z * height * width + y * width + x];
    elem[z * height * width + y * width + x] = sigmoid(element);
  }
}

// Host code
void malloc2D (sycl::queue &q, int repeat, int width, int height) {
  printf("Dimension: (%d %d)\n", width, height);

  sycl::range<2> gws ((height + 15)/16*16, (width + 15)/16*16);
  sycl::range<2> lws (16, 16);

  // size of a row in bytes
  size_t pitch = (width * sizeof(float) + 63) & ~(0x3F);

  char* devPitchedPtr = (char*) sycl::malloc_device(pitch * height, q);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      parallelPitched2DAccess(item, devPitchedPtr, pitch, width, height);
    });
  }).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        parallelPitched2DAccess(item, devPitchedPtr, pitch, width, height);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  float* devPtr = sycl::malloc_device<float>(width * height, q);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      parallelSimple2DAccess(item, devPtr, width, height);
    });
  }).wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        parallelSimple2DAccess(item, devPtr, width, height);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time (pitched vs simple): %f %f (us)\n",
          (time * 1e-3f) / repeat, (time2 * 1e-3f) / repeat);

  sycl::free(devPitchedPtr, q);
  sycl::free(devPtr, q);
}


// Host code
void malloc3D (sycl::queue &q, int repeat, int width, int height, int depth) {
  printf("Dimension: (%d %d %d)\n", width, height, depth);
  sycl::range<3> gws ((depth + 3)/4*4, (height + 7)/8*8, (width + 15)/16*16);
  sycl::range<3> lws (4, 8, 16);

  // size of a row in bytes
  size_t pitch = (width * sizeof(float) + 63) & ~(0x3F);

  char* devPitchedPtr = (char*) sycl::malloc_device(pitch * height * depth, q);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
      parallelPitched3DAccess(item, devPitchedPtr, pitch, width, height, depth);
    });
  }).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        parallelPitched3DAccess(item, devPitchedPtr, pitch, width, height, depth);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  float *devPtr = sycl::malloc_device<float>(width * height * depth, q);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
      parallelSimple3DAccess(item, devPtr, width, height, depth);
    });
  }).wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        parallelSimple3DAccess(item, devPtr, width, height, depth);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time (pitched vs simple): %f %f (us)\n",
          (time * 1e-3f) / repeat, (time2 * 1e-3f) / repeat);

  sycl::free(devPitchedPtr, q);
  sycl::free(devPtr, q);
}


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // width, height and depth
  const int w[] = {227, 256, 720, 768, 854, 1280, 1440, 1920, 2048, 3840, 4096};
  const int h[] = {227, 256, 480, 576, 480, 720, 1080, 1080, 1080, 2160, 2160};
  const int d[] = {1, 3};

  for (int i = 0; i < 11; i++)
    malloc2D(q, repeat, w[i], h[i]);

  for (int i = 0; i < 11; i++)
    for (int j = 0; j < 2; j++)
      malloc3D(q, repeat, w[i], h[i], d[j]);

  return 0;
}
