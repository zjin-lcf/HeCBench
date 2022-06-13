#include "distance.h"

__global__ void compute_haversine_distance(
  const double4 *__restrict__ p,
        double*__restrict__ distance,
  const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    auto ay = p[i].x * DEGREE_TO_RADIAN;  // a_lat
    auto ax = p[i].y * DEGREE_TO_RADIAN;  // a_lon
    auto by = p[i].z * DEGREE_TO_RADIAN;  // b_lat
    auto bx = p[i].w * DEGREE_TO_RADIAN;  // b_lon

    // haversine formula
    auto x        = (bx - ax) / 2.0;
    auto y        = (by - ay) / 2.0;
    auto sinysqrd = sin(y) * sin(y);
    auto sinxsqrd = sin(x) * sin(x);
    auto scale    = cos(ay) * cos(by);
    distance[i] = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
  }
}

void distance_device(const double4* loc, double* dist, const int n, const int iteration) {

  dim3 grids ((n+255)/256);
  dim3 threads (256);

  double4 *d_loc;
  double *d_dist;
  hipMalloc((void**)&d_loc, sizeof(double4)*n);
  hipMemcpy(d_loc, loc, sizeof(double4)*n, hipMemcpyHostToDevice);
  hipMalloc((void**)&d_dist, sizeof(double)*n);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iteration; i++) {
    hipLaunchKernelGGL(compute_haversine_distance, grids, threads, 0, 0, d_loc, d_dist, n);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);

  hipMemcpy(dist, d_dist, sizeof(double)*n, hipMemcpyDeviceToHost);
  hipFree(d_loc);
  hipFree(d_dist);
}
