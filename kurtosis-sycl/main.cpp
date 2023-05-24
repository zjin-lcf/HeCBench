#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>   // std::transform_reduce
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>

#include "StoreTypedefs.h"
#include "StoreElement.h"
#include "kurtosis.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage ./main <elemCount> <repeat>\n";
    return 1;
  }

  const int elemCount = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  storeElement *elem;
  elem = sycl::malloc_shared<storeElement>(elemCount, q);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(1.f, 2.f);
  for (int i = 0; i < elemCount; i++)
    elem[i] = {i, 0, (unsigned long long int)i, dis(gen)};

  auto start = std::chrono::steady_clock::now();

  kurtosisResult* result;
  const size_t s = kurtosis(q, elem, elemCount, repeat, (void**)&result);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Total device compute time: " << time * 1e-9 << " (s)\n";

  std::cout << "Results:" << std::endl;
  std::cout << s << " "
            << result->count << " "
            << result->m2 << " "
            << result->m3 << " "
            << result->m4 << std::endl;

  sycl::free(elem, q);
  delete result;
  return 0;
}
