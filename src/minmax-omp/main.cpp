#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <random>
#include <iostream>
#include "vec_2d.hpp"
#include "utils.hpp"

// the value type is not templated
struct MinPair { float val; int64_t idx; };

#pragma omp declare reduction( \
    minpair : MinPair : \
    omp_out = (omp_in.val < omp_out.val) ? omp_in : omp_out \
) initializer(omp_priv = {INFINITY, -1})

// the value type is not templated
struct MaxPair { float val; int64_t idx; };

#pragma omp declare reduction( \
    maxpair : MaxPair : \
    omp_out = (omp_in.val > omp_out.val) ? omp_in : omp_out \
) initializer(omp_priv = {-INFINITY, -1})

// the value type is not templated
struct MinMaxPair { float min_val; float max_val; int64_t min_idx; int64_t max_idx; };

#pragma omp declare reduction( \
    minmaxpair : MinMaxPair : \
    omp_out = MinMaxPair{ \
        /* min_val */ (omp_in.min_val < omp_out.min_val ? omp_in.min_val : omp_out.min_val), \
        /* max_val */ (omp_in.max_val > omp_out.max_val ? omp_in.max_val : omp_out.max_val), \
        /* min_idx */ (omp_in.min_val < omp_out.min_val ? omp_in.min_idx : omp_out.min_idx), \
        /* max_idx */ (omp_in.max_val > omp_out.max_val ? omp_in.max_idx : omp_out.max_idx) \
    } \
) initializer(omp_priv = {INFINITY, -INFINITY, -1, -1})

template <typename T>
void eval (const T bounding_box_size, const int repeat) {

  // use std::get to fetch members in the pair
  auto pair = generate_points<T>(bounding_box_size);
  std::size_t N = std::get<0>(pair); 
  std::vector<vec_2d<T>> points_vector = std::get<1>(pair); 

  printf("Total number of points: %zu\n", N);

  vec_2d<T> min_point[2], max_point[2];

  auto points = points_vector.data();

  #pragma omp target data map(to: points[0:N])
  {
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {

      /*#pragma omp target teams distribute parallel for reduction(min:min_idx)
      for (size_t i = 0; i < N; i++) {
          if (norm2(points[i]) < norm2(points[min_idx])) {
              min_idx = i;
          }
      }*/

      MinPair min_result;
      min_result.val = INFINITY;
      min_result.idx = -1;
      
      #pragma omp target teams distribute parallel for \
       reduction(minpair:min_result)
      for (std::size_t i = 0; i < N; i++) {
          float val = norm2(points[i]);
          if (val < min_result.val) {
              min_result.val = val;
              min_result.idx = i;
          }
      }

      /*#pragma omp target teams distribute parallel for reduction(max:max_idx)
      for (size_t i = 0; i < N; i++) {
          if (norm2(points[i]) > norm2(points[max_idx])) {
              max_idx = i;
          }
      }*/
      MaxPair max_result;
      max_result.val = -INFINITY;
      max_result.idx = -1;
      
      #pragma omp target teams distribute parallel for \
       reduction(maxpair:max_result)
      for (std::size_t i = 0; i < N; i++) {
        float val = norm2(points[i]);
        if (val > max_result.val) {
            max_result.val = val;
            max_result.idx = i;
        }
      }

      min_point[0] = points[min_result.idx];
      max_point[0] = points[max_result.idx];
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    printf("Average execution time of omp:min() + omp:max(): %f (us)\n",
           (time * 1e-3f) / repeat);

    // Single-pass minmax
    start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {

      /*#pragma omp target teams distribute parallel for reduction(min:min_idx) reduction(max:max_idx)
      for (size_t i = 0; i < N; i++) {
          auto p = norm2(points[i]);
          if (p < norm2(points[min_idx])) {
              min_idx = i;
          }
          if (p > norm2(points[max_idx])) {
              max_idx = i;
          }
      }*/

      MinMaxPair result;
      result.min_val = INFINITY;
      result.max_val = -INFINITY;
      result.min_idx = -1;
      result.max_idx = -1;
      
      #pragma omp target teams distribute parallel for \
       reduction(minmaxpair:result)
      for (std::size_t i = 0; i < N; i++) {
        float val = norm2(points[i]);
      
        if (val < result.min_val) {
            result.min_val = val;
            result.min_idx = i;
        }
      
        if (val > result.max_val) {
            result.max_val = val;
            result.max_idx = i;
        }
      }

      min_point[1] = points[result.min_idx];
      max_point[1] = points[result.max_idx];
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    printf("Average execution time of omp:minmax(): %f (us)\n",
           (time * 1e-3f) / repeat);
  }

  // CPU reference
  int ref_min = 0, ref_max = 0;

  for (std::size_t i = 1; i < N; i++) {
    if (norm2(points[i]) < norm2(points[ref_min])) ref_min = i;
    if (norm2(points[i]) > norm2(points[ref_max])) ref_max = i;
  }

  vec_2d<T> r_min_point = points[ref_min];
  vec_2d<T> r_max_point = points[ref_max];

  bool ok = (min_point[0] == r_min_point) && (max_point[0] == r_max_point);
  ok &= (min_point[1] == r_min_point) && (max_point[1] == r_max_point);

  printf("%s\n", ok ? "PASS" : "FAIL");
}


int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <bounding-box size> <repeat>\n", argv[0]);
    return 1;
  }

  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  eval((float)size, repeat);

  return 0;
}
