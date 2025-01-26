#include <chrono>
#include <cstdio>
#include <sycl/sycl.hpp>
#include "kiss.h"

template <class T, class Rng>
void kernel (T * const out, const std::uint64_t n, const std::uint32_t seed,
             const sycl::nd_item<3> &item)
{
  const std::uint32_t tid = item.get_global_id(2);

  // generate initial local seed per thread
  const std::uint32_t local_seed =
      kiss::hashers::MurmurHash<std::uint32_t>::hash(seed+tid);

  Rng rng {local_seed};

  // grid-stride loop
  const auto grid_stride = item.get_local_range(2) * item.get_group_range(2);
  for(std::uint64_t i = tid; i < n; i += grid_stride)
  {
      // generate random element and write to output
      out[i] = rng.template next<T>();
  }
}

template<class T, class Rng>
inline
void uniform_distribution(
    sycl::queue &q,
    T * const out,
    const std::uint64_t n,
    const std::uint32_t seed) noexcept
{
  // execute kernel
  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 4096 * 256),
                        sycl::range<3>(1, 1, 256)),
      [=](sycl::nd_item<3> item) {
        kernel<T, Rng>(out, n, seed, item);
  });
}

// This example shows the easy generation of gigabytes of uniform random values
// in only a few milliseconds.

int main(int argc, char* argv[])
{
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      return 1;
    }
    const int repeat = atoi(argv[1]);

    // define the data types to be generated
    using data_t = std::uint64_t;
    using rng_t = kiss::Kiss<data_t>;

    static constexpr std::uint64_t n = 1UL << 28;

    static constexpr std::uint32_t seed = 42;

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    data_t * data_h = nullptr;
    data_h = sycl::malloc_host<data_t>(n, q);

    data_t * data_d = nullptr;
    data_d = sycl::malloc_device<data_t>(n, q);

    q.memset(data_d, 0, sizeof(data_t) * n).wait();

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      uniform_distribution<data_t, rng_t>(q, data_d, n, seed);
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(data_h, data_d, sizeof(data_t) * n).wait();

    printf("\nThe first 10 random numbers:\n");
    for(std::uint64_t i = 0; i < 10; i++)
    {
      printf("%lu\n", data_h[i]);
    }

    sycl::free(data_h, q);
    sycl::free(data_d, q);
    return 0;
}
