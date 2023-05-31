#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

using float3 = sycl::float3;
using float4 = sycl::float4;

#ifndef SYCL_Geometric
inline float dot(const float3 &a, const float3 &b)
{
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

inline float3 normalize(const float3 &v)
{
  float invLen = sycl::rsqrt(dot(v, v));
  return v * invLen;
}

inline float3 cross(const float3 &a, const float3 &b)
{
  return float3(a.y()*b.z() - a.z()*b.y(), a.z()*b.x() - a.x()*b.z(), a.x()*b.y() - a.y()*b.x());
}

inline float length(const float3 &v)
{
  return sycl::sqrt(dot(v, v));
}
#endif

inline float4 normalEstimate(const float3 *points, int idx, int width, int height)
{
  float3 query_pt = points[idx];
  if (sycl::isnan(query_pt.z()))
    return float4 (0.f,0.f,0.f,0.f);

  int xIdx = idx % width;
  int yIdx = idx / width;

  // are we at a border? are our neighbor valid points?
  bool west_valid  = (xIdx > 1)        && !sycl::isnan (points[idx-1].z()) &&
                     sycl::fabs (points[idx-1].z() - query_pt.z()) < 200.f;
  bool east_valid  = (xIdx < width-1)  && !sycl::isnan (points[idx+1].z()) &&
                     sycl::fabs (points[idx+1].z() - query_pt.z()) < 200.f;
  bool north_valid = (yIdx > 1)        && !sycl::isnan (points[idx-width].z()) &&
                     sycl::fabs (points[idx-width].z() - query_pt.z()) < 200.f;
  bool south_valid = (yIdx < height-1) && !sycl::isnan (points[idx+width].z()) &&
                     sycl::fabs (points[idx+width].z() - query_pt.z()) < 200.f;

  float3 horiz, vert;
  if (west_valid & east_valid)
    horiz = points[idx+1] - points[idx-1];
  if (west_valid & !east_valid)
    horiz = points[idx] - points[idx-1];
  if (!west_valid & east_valid)
    horiz = points[idx+1] - points[idx];
  if (!west_valid & !east_valid)
    return float4 (0.f,0.f,0.f,1.f);

  if (south_valid & north_valid)
    vert = points[idx-width] - points[idx+width];
  if (south_valid & !north_valid)
    vert = points[idx] - points[idx+width];
  if (!south_valid & north_valid)
    vert = points[idx-width] - points[idx];
  if (!south_valid & !north_valid)
    return float4 (0.f,0.f,0.f,1.f);

#ifdef SYCL_Geometric
  float3 normal = sycl::cross (horiz, vert);
  float curvature = sycl::length (normal);
#else
  float3 normal = cross (horiz, vert);
  float curvature = length (normal);
#endif

  curvature = sycl::fabs(horiz.z()) > 0.04f || sycl::fabs(vert.z()) > 0.04f ||
              !west_valid || !east_valid || !north_valid || !south_valid;

#ifdef SYCL_Geometric
  float3 mc = sycl::normalize (normal);
  if ( sycl::dot (query_pt, mc) > 0.f )
#else
  float3 mc = normalize (normal);
  if ( dot (query_pt, mc) > 0.f )
#endif
    mc = mc * -1.f;
  return float4 (mc.x(), mc.y(), mc.z(), curvature);
}

void ne (
  sycl::nd_item<1> &item,
  const float3 *__restrict points,
        float4 *__restrict normal_points,
  const int width,
  const int height,
  const int numPts)
{
  int idx = item.get_global_id(0);
  if (idx < numPts)
    normal_points[idx] = normalEstimate(points, idx, width, height);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int numPts = width * height;
  const int size = numPts * sizeof(float3);
  const int normal_size = numPts * sizeof(float4);
  float3 *points = (float3*) malloc (size);
  float4 *normal_points = (float4*) malloc (normal_size);

  srand(123);
  for (int i = 0; i < numPts; i++) {
    points[i].x() = rand() % width;
    points[i].y() = rand() % height;
    points[i].z() = rand() % 256;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float3 *d_points = sycl::malloc_device<float3>(numPts, q);
  q.memcpy(d_points, points, size);

  float4 *d_normal_points = sycl::malloc_device<float4>(numPts, q);

  sycl::range<1> gws ((numPts + 255)/256*256);
  sycl::range<1> lws  (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class normal_estimate>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        ne(item, d_points, d_normal_points, width, height, numPts);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(normal_points, d_normal_points, normal_size).wait();

  float sx, sy, sz, sw;
  sx = sy = sz = sw = 0.f;
  for (int i = 0; i < numPts; i++) {
    sx += normal_points[i].x();
    sy += normal_points[i].y();
    sz += normal_points[i].z();
    sw += normal_points[i].w();
  }
  printf("Checksum: x=%f y=%f z=%f w=%f\n", sx, sy, sz, sw);

  sycl::free(d_normal_points, q);
  sycl::free(d_points, q);
  free(normal_points);
  free(points);
  return 0;
}
