#include <iostream>
#include <iomanip>
#include "bench_merkle_tree.hpp"

#ifdef DEBUG
const size_t BENCH_ROUND = 1;
#else
const size_t BENCH_ROUND = 4;
#endif

int
main(int argc, char** argv)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::cout << "\nMerklize ( approach 1 ) using Rescue Prime on F(2**64 - "
               "2**32 + 1) elements\n\n";

  std::cout << std::setw(11) << "leaves"
            << "\t\t" << std::setw(15) << "total" << std::endl;

#ifdef DEBUG
  for (uint dim = 256; dim <= 256; dim <<= 1) {
#else
  for (uint dim = (1ul << 20); dim <= (1ul << 24); dim <<= 1) {
#endif
    // time in nanoseconds --- beware !
    double tm = 0;
    for (size_t i = 0; i < BENCH_ROUND; i++) {
      tm +=
        static_cast<double>(benchmark_merklize_approach_1(q, dim, 1ul << 5));
    }
    tm /= static_cast<double>(BENCH_ROUND);

    std::cout << std::setw(11) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << tm * 1e-6 << " ms" << std::endl;
  }

  return 0;
}
