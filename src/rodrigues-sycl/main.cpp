#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

using float3 = sycl::float3;
using float4 = sycl::float4;

void rotate (sycl::nd_item<1> &item, const int n, const float angle, const float3 w,
             float3 *d)
{
  int i = item.get_global_id(0);
  if (i >= n) return;

  float s, c;
  s = sycl::sincos(angle, &c);

  const float3 p = d[i];
  const float mc = 1.f - c;

  // Rodrigues' formula:
  float m1 = c+(w.x())*(w.x())*(mc);
  float m2 = (w.z())*s+(w.x())*(w.y())*(mc);
  float m3 =-(w.y())*s+(w.x())*(w.z())*(mc);

  float m4 =-(w.z())*s+(w.x())*(w.y())*(mc);
  float m5 = c+(w.y())*(w.y())*(mc);
  float m6 = (w.x())*s+(w.y())*(w.z())*(mc);

  float m7 = (w.y())*s+(w.x())*(w.z())*(mc);
  float m8 =-(w.x())*s+(w.y())*(w.z())*(mc);
  float m9 = c+(w.z())*(w.z())*(mc);

  float ox = p.x()*m1 + p.y()*m2 + p.z()*m3;
  float oy = p.x()*m4 + p.y()*m5 + p.z()*m6;
  float oz = p.x()*m7 + p.y()*m8 + p.z()*m9;
  d[i] = {ox, oy, oz};
}

void rotate2 (sycl::nd_item<1> &item, const int n, const float angle, const float3 w,
              float4 *d)
{
  int i = item.get_global_id(0);
  if (i >= n) return;

  float s, c;
  s = sycl::sincos(angle, &c);

  const float4 p = d[i];
  const float mc = 1.f - c;

  // Rodrigues' formula:
  float m1 = c+(w.x())*(w.x())*(mc);
  float m2 = (w.z())*s+(w.x())*(w.y())*(mc);
  float m3 =-(w.y())*s+(w.x())*(w.z())*(mc);

  float m4 =-(w.z())*s+(w.x())*(w.y())*(mc);
  float m5 = c+(w.y())*(w.y())*(mc);
  float m6 = (w.x())*s+(w.y())*(w.z())*(mc);

  float m7 = (w.y())*s+(w.x())*(w.z())*(mc);
  float m8 =-(w.x())*s+(w.y())*(w.z())*(mc);
  float m9 = c+(w.z())*(w.z())*(mc);

  float ox = p.x()*m1 + p.y()*m2 + p.z()*m3;
  float oy = p.x()*m4 + p.y()*m5 + p.z()*m6;
  float oz = p.x()*m7 + p.y()*m8 + p.z()*m9;
  d[i] = {ox, oy, oz, 0.f};
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  // axis of rotation
  const float wx = -0.3, wy = -0.6, wz = 0.15;
  const float norm = 1.f / sqrtf(wx*wx + wy*wy + wz*wz);
  const float3 w = {wx*norm, wy*norm, wz*norm};

  float angle = 0.5f;

  float3 *h = (float3*) malloc (sizeof(float3) * n);
  float4 *h2 = (float4*) malloc (sizeof(float4) * n);

  srand(123);
  for (int i = 0; i < n; i++) {
    float a = rand();
    float b = rand();
    float c = rand();
    float d = sqrtf(a*a + b*b + c*c);
    h[i] = {a/d, b/d, c/d};
    h2[i] = {a/d, b/d, c/d, 0.f};
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::range<1> gws ((n + 255) / 256 * 256);
  sycl::range<1> lws  (256);

  float3 *d = sycl::malloc_device<float3>(n, q);
  q.memcpy(d, h, sizeof(float3) * n);

  float4 *d2 = sycl::malloc_device<float4>(n, q);
  q.memcpy(d2, h2, sizeof(float4) * n);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rr>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        rotate(item, n, angle, w, d);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (float3): %f (us)\n", (time * 1e-3f) / repeat);

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class rr2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        rotate2(item, n, angle, w, d2);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (float4): %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d, q);
  sycl::free(d2, q);
  free(h);
  free(h2);
  return 0;
}
