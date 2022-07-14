#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include "common.h"

#define VERTICES 600
#define BLOCK_SIZE_X 256

#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng (123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x() = distribution(rng);
    point[i].y() = distribution(rng);
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x() = cosf(t);
    vertex[i].y() = sinf(t);
  }

  // kernel results
  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float2, 1> d_point (point, nPoints);
  buffer<float2, 1> d_vertex (vertex, vertices);
  buffer<int, 1> d_bitmap_ref (nPoints);
  buffer<int, 1> d_bitmap_opt (nPoints);

  //kernel parameters
  range<1> lws (BLOCK_SIZE_X);
  range<1> gws ((nPoints+BLOCK_SIZE_X-1) / BLOCK_SIZE_X * BLOCK_SIZE_X);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_ref.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class reference>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_base(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_bitmap_ref.get_access<sycl_read>(cgh);
    cgh.copy(acc, bitmap_ref);
  }).wait();

  // performance tuning with tile sizes
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<1>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<2>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt3>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<4>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt4>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<8>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt5>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<16>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt6>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<32>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto bm = d_bitmap_opt.get_access<sycl_discard_write>(cgh);
      auto p = d_point.get_access<sycl_read>(cgh);
      auto v = d_vertex.get_access<sycl_read>(cgh);
      cgh.parallel_for<class opt7>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pnpoly_opt<64>(item, bm.get_pointer(), p.get_pointer(), v.get_pointer(), nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_bitmap_opt.get_access<sycl_read>(cgh);
    cgh.copy(acc, bitmap_opt);
  }).wait();

  // compare against reference kernel for verification
  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 

  // double check
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
