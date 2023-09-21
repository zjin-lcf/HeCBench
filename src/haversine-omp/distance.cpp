#include "distance.h"

void distance_device(const double4* loc, double* dist, const int n, const int iteration) {
  
  #pragma omp target data map(to: loc[0:n]) map(from: dist[0:n])
  {
    for (int i = 0; i < iteration; i++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int p = 0; p < n; p++) {
        auto ay = loc[p].x * DEGREE_TO_RADIAN;  // a_lat
        auto ax = loc[p].y * DEGREE_TO_RADIAN;  // a_lon
        auto by = loc[p].z * DEGREE_TO_RADIAN;  // b_lat
        auto bx = loc[p].w * DEGREE_TO_RADIAN;  // b_lon

        // haversine formula
        auto x        = (bx - ax) / 2.0;
        auto y        = (by - ay) / 2.0;
        auto sinysqrd = sin(y) * sin(y);
        auto sinxsqrd = sin(x) * sin(x);
        auto scale    = cos(ay) * cos(by);
        dist[p] = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
      }
    }
  }
}
