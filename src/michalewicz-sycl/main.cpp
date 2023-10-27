#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

inline
float michalewicz(const float *xValues, const int dim) {
  float result = 0;
  for (int i = 0; i < dim; ++i) {
      float a = sycl::sin(xValues[i]);
      float b = sycl::sin(((i + 1) * xValues[i] * xValues[i]) / (float)M_PI);
      float c = sycl::pow(b, 20.f); // m = 10
      result += a * c;
  }
  return -1.0f * result;
}

// https://www.sfu.ca/~ssurjano/michal.html
void Error(float value, int dim) {
  printf("Global minima = %f\n", value);
  float trueMin = 0.0;
  if (dim == 2)
    trueMin = -1.8013;
  else if (dim == 5)
    trueMin = -4.687658;
  else if (dim == 10)
    trueMin = -9.66015;
  printf("Error = %f\n", fabsf(trueMin - value));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  // generate random numbers
  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.0, 4.0);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  // dimensions
  const int dims[] = {2, 5, 10}; 

  for (int d = 0; d < 3; d++) {

    const int dim = dims[d];

    const size_t size = n * dim;
    const size_t size_bytes = size * sizeof(float);
    
    float *values = (float*) malloc (size_bytes);
    
    for (int i = 0; i < size; i++) {
      values[i] = dis(gen);
    }
    
    float *d_values = sycl::malloc_device<float>(size, q);
    q.memcpy(d_values, values, size_bytes);

    float *d_minValue = sycl::malloc_device<float>(1, q);
    float minValue;

    sycl::range<1> gws ((n + 255) / 256 * 256);
    sycl::range<1> lws (256);

    q.memset(d_minValue, 0, sizeof(float)).wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          size_t i = item.get_global_id(0);
          if (i < n) {
            auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                      sycl::memory_scope::device,\
                      sycl::access::address_space::generic_space>(*d_minValue);
            ao.fetch_min(michalewicz(d_values + i * dim, dim));
          }
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel (dim = %d): %f (us)\n",
           dim, (time * 1e-3f) / repeat);

    q.memcpy(&minValue, d_minValue, sizeof(float)).wait();
    Error(minValue, dim);

    sycl::free(d_values, q);
    sycl::free(d_minValue, q);
    free(values);
  }

  return 0;
}
