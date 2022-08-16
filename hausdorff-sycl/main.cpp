#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "common.h"
#include "reference.h"

inline float hd (const float2 ap, const float2 bp)
{
  return (ap.x() - bp.x()) * (ap.x() - bp.x())
       + (ap.y() - bp.y()) * (ap.y() - bp.y());
}

void computeDistance(nd_item<1> &item,
                     const float2* __restrict Apoints,
                     const float2* __restrict Bpoints,
                           float*  __restrict distance,
                     const int numA, const int numB)
{
  int i = item.get_global_id(0);
  if (i >= numA) return;

  float d = std::numeric_limits<float>::max();
  float2 p = Apoints[i];
  for (int j = 0; j < numB; j++)
  {
    float t = hd(p, Bpoints[j]);
    d = std::min(t, d);
  }
  
  auto atomic_obj_ref = ext::oneapi::atomic_ref<float, 
    ext::oneapi::memory_order::relaxed,
    ext::oneapi::memory_scope::device,
    access::address_space::global_space> (distance[0]);
  atomic_obj_ref.fetch_max(d);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points in space A>", argv[0]);
    printf(" <number of points in space B> <repeat>\n");
    return 1;
  }
  const int num_Apoints = atoi(argv[1]);
  const int num_Bpoints = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t num_Apoints_bytes = sizeof(float2) * num_Apoints;
  const size_t num_Bpoints_bytes = sizeof(float2) * num_Bpoints;

  float2 *h_Apoints = (float2*) malloc (num_Apoints_bytes);
  float2 *h_Bpoints = (float2*) malloc (num_Bpoints_bytes);
  
  srand(123);
  for (int i = 0; i < num_Apoints; i++) {
    h_Apoints[i].x() = (float)rand() / (float)RAND_MAX;
    h_Apoints[i].y() = (float)rand() / (float)RAND_MAX;
  }
  
  for (int i = 0; i < num_Bpoints; i++) {
    h_Bpoints[i].x() = (float)rand() / (float)RAND_MAX;
    h_Bpoints[i].y() = (float)rand() / (float)RAND_MAX;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);
  buffer<float2, 1> d_Apoints (h_Apoints, num_Apoints);
  buffer<float2, 1> d_Bpoints (h_Bpoints, num_Bpoints);
  buffer<float, 1> d_distance (2);

  range<1> gwsA ((num_Apoints + 255) / 256 * 256);
  range<1> gwsB ((num_Bpoints + 255) / 256 * 256);
  range<1> lws (256);

  float h_distance[2] = {-1.f, -1.f}; 

  double time = 0.0;

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto acc = d_distance.get_access<sycl_discard_write>(cgh);
      cgh.copy(h_distance, acc);
    }).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      auto a = d_Apoints.get_access<sycl_read>(cgh);
      auto b = d_Bpoints.get_access<sycl_read>(cgh);
      auto d = d_distance.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class distanceAB>(nd_range<1>(gwsA, lws), [=] (nd_item<1> item) {
        computeDistance(item, a.get_pointer(), b.get_pointer(), d.get_pointer(),
                        num_Apoints, num_Bpoints);
      });
    });

    q.submit([&] (handler &cgh) {
      auto a = d_Bpoints.get_access<sycl_read>(cgh);
      auto b = d_Apoints.get_access<sycl_read>(cgh);
      auto d = d_distance.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class distanceBA>(nd_range<1>(gwsB, lws), [=] (nd_item<1> item) {
        computeDistance(item, a.get_pointer(), b.get_pointer(), d.get_pointer() + 1,
                        num_Bpoints, num_Apoints);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of kernels: %f (ms)\n", (time * 1e-6f) / repeat);

  q.submit([&] (handler &cgh) {
    auto acc = d_distance.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_distance);
  }).wait();

  printf("Verifying the result may take a while..\n");
  float r_distance = hausdorff_distance(h_Apoints, h_Bpoints, num_Apoints, num_Bpoints);
  float t_distance = std::max(h_distance[0], h_distance[1]);

  bool error = (fabsf(t_distance - r_distance)) > 1e-3f;
  printf("%s\n", error ? "FAIL" : "PASS");

  free(h_Apoints);
  free(h_Bpoints);
  return 0;
}
