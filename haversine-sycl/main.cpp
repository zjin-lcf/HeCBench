#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "distance.h"

void distance_device(queue &q, const double4* loc, double* dist, const int n, const int iteration) {

  range<1> gws ((n+255)/256*256);
  range<1> lws (256);

  buffer<double4, 1> d_loc (loc, n);
  buffer<double, 1> d_dist (dist, n);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iteration; i++) {
    q.submit([&] (handler &cgh) {
      auto in = d_loc.get_access<sycl_read>(cgh);
      auto out = d_dist.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class haversine>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) {
          auto ay = in[i].x() * DEGREE_TO_RADIAN;  // a_lat
          auto ax = in[i].y() * DEGREE_TO_RADIAN;  // a_lon
          auto by = in[i].z() * DEGREE_TO_RADIAN;  // b_lat
          auto bx = in[i].w() * DEGREE_TO_RADIAN;  // b_lon

          // haversine formula
          auto x        = (bx - ax) / 2.0;
          auto y        = (by - ay) / 2.0;
          auto sinysqrd = sycl::sin(y) * sycl::sin(y);
          auto sinxsqrd = sycl::sin(x) * sycl::sin(x);
          auto scale    = sycl::cos(ay) * sycl::cos(by);
          out[i] = 2.0 * EARTH_RADIUS_KM * sycl::asin(sycl::sqrt(sinysqrd + sinxsqrd * scale));
        }
      });
    });
  }
  
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iteration);
}

void verify(int size, const double *output, const double *expected_output) {
  double error_rate = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(output[i] - expected_output[i]) > error_rate) {
      error_rate = fabs(output[i] - expected_output[i]);
    }
  }
  printf("The maximum error in distance is %f\n", error_rate); 
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <file> <repeat>\n", argv[0]);
    return 1;
  }
  const char* filename = argv[1];
  const int repeat = atoi(argv[2]);

  printf("Reading city locations from file %s...\n", filename);
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    perror ("Error opening the file");
    exit(-1);
  }

  int num_cities = 2097152; // 2 ** 21
  int num_ref_cities = 6; // bombay, melbourne, waltham, moscow, glasgow, morocco
  int index_map[] ={436483, 1952407, 627919, 377884, 442703, 1863423};
  int N = num_cities * num_ref_cities;
  int city = 0;
  double lat, lon;

  double4* input  = (double4*) aligned_alloc(4096, N*sizeof(double4));
  double*  output = (double*) aligned_alloc(4096, N*sizeof(double));
  double*  expected_output = (double*) malloc(N*sizeof(double));

  while (fscanf(fp, "%lf %lf\n", &lat, &lon) != EOF) { 
    input[city].x() = lat;
    input[city].y() = lon;
    city++;
    if (city == num_cities) break;  
  }
  fclose(fp);

  // duplicate for "num_ref_cities"
  for (int c = 1;  c < num_ref_cities; c++) {
    std::copy(input, input+num_cities, input+c*num_cities);
  }
  // each reference city is compared with 'num_cities' cities
  for (int c = 0;  c < num_ref_cities; c++) {
    int index = index_map[c] - 1;
    for(int j = c*num_cities; j < (c+1)*num_cities; ++j) {
      input[j].z() = input[index].x();
      input[j].w() = input[index].y();
    }
  }

  // run on the host for verification
  for (int i = 0; i < N; i++)
  {
    double a_lat = input[i].x();
    double a_lon = input[i].y();
    double b_lat = input[i].z();
    double b_lon = input[i].w();

    auto ax = a_lon * DEGREE_TO_RADIAN;
    auto ay = a_lat * DEGREE_TO_RADIAN;
    auto bx = b_lon * DEGREE_TO_RADIAN;
    auto by = b_lat * DEGREE_TO_RADIAN;

    // haversine formula
    auto x        = (bx - ax) / 2.0;
    auto y        = (by - ay) / 2.0;
    auto sinysqrd = sin(y) * sin(y);
    auto sinxsqrd = sin(x) * sin(x);
    auto scale    = cos(ay) * cos(by);
    expected_output[i] = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  distance_device(q, input, output, N, repeat);

  verify(N, output, expected_output);

  free(input);
  free(output);
  free(expected_output);
  return 0;
}
