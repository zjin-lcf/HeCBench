#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <omp.h>

using half = _Float16;

#include "reference.h"

inline half float_to_half(float f)
{
  return half(f);
}
inline float half_to_float(half h)
{
  return float(h);
}

// Operation: (feature > 0) ? gradient : 0
template <int VL=2>
void ReluGrad_impl(const half* __restrict__ gradient,
                   const half* __restrict__ feature,
                         half* __restrict__ backprop,
                   const int count)
{
  int v_count = count / VL;

  #pragma omp target teams distribute parallel for num_threads(256)
  for (int index = 0; index < v_count; index++) {
    int base = index * VL;
    float g[VL], f[VL];

    #pragma unroll
    for (int i = 0; i < VL; i++) {
      g[i] = half_to_float(gradient[base + i]);
      f[i] = half_to_float(feature[base + i]);
    }

    #pragma unroll
    for (int i = 0; i < VL; i++) {
      backprop[base + i] = float_to_half((f[i] > 0.f) ? g[i] : 0.f);
    }
  }
}

template <int VL=2>
void Relu_impl(int count, const int* input, int* output)
{
  int v_count = count / VL;

  #pragma omp target teams distribute parallel for num_threads(256)
  for (int index = 0; index < v_count; index++) {
    int base = index * VL;
    int inp[VL];
    int r0[VL], r1[VL], r2[VL], r3[VL];
    signed char b0[VL], b1[VL], b2[VL], b3[VL];

    #pragma unroll
    for (int i = 0; i < VL; i++) {
      inp[i] = input[base + i];
    }

    #pragma unroll
    for (int i = 0; i < VL; i++) {
      b0[i] = (signed char)( inp[i] & 0xFF);
      b1[i] = (signed char)((inp[i] >>  8) & 0xFF);
      b2[i] = (signed char)((inp[i] >> 16) & 0xFF);
      b3[i] = (signed char)((inp[i] >> 24) & 0xFF);

      r0[i] = (b0[i] > 0) ? b0[i] : 0;
      r1[i] = (b1[i] > 0) ? b1[i] : 0;
      r2[i] = (b2[i] > 0) ? b2[i] : 0;
      r3[i] = (b3[i] > 0) ? b3[i] : 0;
    }

    #pragma unroll
    for (int i = 0; i < VL; i++) {
      output[base + i] = (r3[i] << 24) | (r2[i] << 16) | (r1[i] << 8) | r0[i];
    }
  }
}


int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <count> <repeat>\n", argv[0]);
    return 1;
  }

  const int count  = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t hsize = count * sizeof(half);

  half *h_gradient = (half*) malloc(hsize);
  half *h_feature  = (half*) malloc(hsize);
  half *h_backprop = (half*) malloc(hsize);
  half *r_backprop = (half*) malloc(hsize);

  std::mt19937 engine(19937);
  std::uniform_real_distribution<float> real_dist(-1.f, 1.f);

  for (int i = 0; i < count; i++) {
    h_feature[i]  = float_to_half(real_dist(engine));
    h_gradient[i] = float_to_half(1.f);
  }

  // Reference result computed on the host (cast to our half type)
  ReluGrad_reference(count, h_gradient, h_feature, r_backprop);

  const int vec_len[] = {1, 2, 4, 8};

  #pragma omp target data map(to: h_gradient[0:count], h_feature[0:count]) \
                          map(alloc: h_backprop[0:count])
  {
    for (int vl : vec_len) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        switch (vl) {
          case 1:
            ReluGrad_impl<1>(h_gradient, h_feature, h_backprop, count);
            break;
          case 2:
            ReluGrad_impl<2>(h_gradient, h_feature, h_backprop, count);
            break;
          case 4:
            ReluGrad_impl<4>(h_gradient, h_feature, h_backprop, count);
            break;
          case 8:
            ReluGrad_impl<8>(h_gradient, h_feature, h_backprop, count);
            break;
        }
      }
      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of ReluGrad_impl (VL=%d): %f (us)\n",
             vl, (time * 1e-3f) / repeat);

      #pragma omp target update from (h_backprop[0:count])
      int fail = 0;
      for (int i = 0; i < count; i++) {
        if (fabsf(half_to_float(h_backprop[i]) -
                  half_to_float(r_backprop[i])) > 1e-3f) {
          fail = 1; break;
        }
      }
      printf("%s\n", fail ? "FAIL" : "PASS");
    }
  }

  free(h_gradient);
  free(h_feature);
  free(h_backprop);
  free(r_backprop);

  // Relu — integer kernels
  size_t isize = count * sizeof(int);
  int *h_in  = (int*) malloc(isize);
  int *h_out = (int*) malloc(isize);
  int *r_out = (int*) malloc(isize);

  std::uniform_int_distribution<unsigned char> int_dist(0, 255);
  for (int i = 0; i < count; i++) {
    h_in[i] = (unsigned)int_dist(engine)       |
              (unsigned)int_dist(engine) <<  8  |
              (unsigned)int_dist(engine) << 16  |
              (unsigned)int_dist(engine) << 24;
  }

  Relu_reference(count, h_in, r_out);

  #pragma omp target data map(to: h_in[0:count]) map(alloc: h_out[0:count])
  {
    for (int vl : vec_len) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        switch (vl) {
          case 1:
            Relu_impl<1>(count, h_in, h_out);
            break;
          case 2:
            Relu_impl<2>(count, h_in, h_out);
            break;
          case 4:
            Relu_impl<4>(count, h_in, h_out);
            break;
          case 8:
            Relu_impl<8>(count, h_in, h_out);
            break;
        }
      }
      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of Relu_impl (VL=%d): %f (us)\n",
             vl, (time * 1e-3f) / repeat);

      #pragma omp target update from (h_out[0:count])

      printf("%s\n", memcmp(h_out, r_out, isize) ? "FAIL" : "PASS");
    }
  }

  free(h_in);
  free(h_out);
  free(r_out);

  return 0;
}
