#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <sycl/sycl.hpp>

using sycl::ext::oneapi::bfloat16;
using sycl::half;
typedef unsigned char uchar;

template <typename Td, typename Ts>
void convert(sycl::queue &q, int nelems, int niters)
{
  Ts *src = sycl::malloc_device<Ts>(nelems, q);
  Td *dst = sycl::malloc_device<Td>(nelems, q);

  const int ls = std::min(nelems, 256);
  const int gs = (nelems + ls - 1) / ls * ls;
  sycl::range<1> gws (gs);
  sycl::range<1> lws (ls);

  // Warm-up run
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      const int i = item.get_global_id(0);
      if (i < nelems) {
        dst[i] = static_cast<Td>(src[i]);
      }
    });
  }).wait();

  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < nelems) {
          dst[i] = static_cast<Td>(src[i]);
        }
      });
    });
  }
  q.wait();
  const auto end = std::chrono::high_resolution_clock::now();
  const double time = std::chrono::duration_cast<std::chrono::microseconds>
                (end - start).count() / niters / 1.0e6f;
  const double size = (sizeof(Td) + sizeof(Ts)) * nelems / 1e9;
  std::printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);

  sycl::free(src, q);
  sycl::free(dst, q);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::printf("bfloat16 -> half\n");
  convert<half, bfloat16>(q, nelems, niters);
  std::printf("bfloat16 -> float\n");
  convert<float, bfloat16>(q, nelems, niters);
  std::printf("bfloat16 -> int\n");
  convert<int, bfloat16>(q, nelems, niters);
  std::printf("bfloat16 -> char\n");
  convert<char, bfloat16>(q, nelems, niters);
  std::printf("bfloat16 -> unsigned char\n");
  convert<uchar, bfloat16>(q, nelems, niters);

  std::printf("half -> half\n");
  convert<half, half>(q, nelems, niters);
  std::printf("half -> float\n");
  convert<float, half>(q, nelems, niters);
  std::printf("half -> int\n");
  convert<int, half>(q, nelems, niters);
  std::printf("half -> char\n");
  convert<char, half>(q, nelems, niters);
  std::printf("half -> uchar\n");
  convert<uchar, half>(q, nelems, niters);

  std::printf("float -> float\n");
  convert<float, float>(q, nelems, niters);
  std::printf("float -> half\n");
  convert<half, float>(q, nelems, niters);
  std::printf("float -> int\n");
  convert<int, float>(q, nelems, niters);
  std::printf("float -> char\n");
  convert<char, float>(q, nelems, niters);
  std::printf("float -> uchar\n");
  convert<uchar, float>(q, nelems, niters);

  std::printf("int -> int\n");
  convert<int, int>(q, nelems, niters);
  std::printf("int -> float\n");
  convert<float, int>(q, nelems, niters);
  std::printf("int -> half\n");
  convert<half, int>(q, nelems, niters);
  std::printf("int -> char\n");
  convert<char, int>(q, nelems, niters);
  std::printf("int -> uchar\n");
  convert<uchar, int>(q, nelems, niters);

  std::printf("char -> int\n");
  convert<int, char>(q, nelems, niters);
  std::printf("char -> float\n");
  convert<float, char>(q, nelems, niters);
  std::printf("char -> half\n");
  convert<half, char>(q, nelems, niters);
  std::printf("char -> char\n");
  convert<char, char>(q, nelems, niters);
  std::printf("char -> uchar\n");
  convert<uchar, char>(q, nelems, niters);

  std::printf("uchar -> int\n");
  convert<int, uchar>(q, nelems, niters);
  std::printf("uchar -> float\n");
  convert<float, uchar>(q, nelems, niters);
  std::printf("uchar -> half\n");
  convert<half, uchar>(q, nelems, niters);
  std::printf("uchar -> char\n");
  convert<char, uchar>(q, nelems, niters);
  std::printf("uchar -> uchar\n");
  convert<uchar, uchar>(q, nelems, niters);

  return 0;
}
