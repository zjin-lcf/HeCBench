#include <algorithm>
#include <chrono>
#include <cstdio>
#include <CL/sycl.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>

using sycl::ext::oneapi::experimental::bfloat16;

using namespace std;
using namespace sycl;

template <typename Td, typename Ts>
void convert(queue &q, int nelems, int niters)
{
  Ts *src = (Ts*) malloc_shared (nelems * sizeof(Ts), q);
  Td *dst = (Td*) malloc_shared (nelems * sizeof(Td), q);

  const size_t ls = std::min((size_t)nelems, (size_t)256);
  const size_t gs = nelems % ls ? (nelems / ls + 1) * ls : nelems;
  range<1> gws (gs);
  range<1> lws (ls);

  // Warm-up run
  q.submit([&](handler &cgh) {
    cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < nelems) {
        dst[i] = static_cast<Td>(src[i]);
      }
    });
  }).wait();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    q.submit([&](handler &cgh) {
      cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < nelems) {
          dst[i] = static_cast<Td>(src[i]);
        }
      });
    });
  }
  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::microseconds>
                (end - start).count() / niters / 1.0e6f;
  double size = (sizeof(Td) + sizeof(Ts)) * nelems / 1e9;
  printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);

  free(src, q);
  free(dst, q);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int niters = atoi(argv[1]);
  const int nelems = 1024 * 1024 * 256;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  printf("bfloat16 -> half\n");
  convert<half, bfloat16>(q, nelems, niters); 
  printf("bfloat16 -> float\n");
  convert<float, bfloat16>(q, nelems, niters); 
  printf("bfloat16 -> int\n");
  convert<int, bfloat16>(q, nelems, niters); 
  printf("bfloat16 -> char\n");
  convert<char, bfloat16>(q, nelems, niters); 
  printf("bfloat16 -> uchar\n");
  convert<uchar, bfloat16>(q, nelems, niters); 

  printf("half -> half\n");
  convert<half, half>(q, nelems, niters); 
  printf("half -> float\n");
  convert<float, half>(q, nelems, niters); 
  printf("half -> int\n");
  convert<int, half>(q, nelems, niters); 
  printf("half -> char\n");
  convert<char, half>(q, nelems, niters); 
  printf("half -> uchar\n");
  convert<uchar, half>(q, nelems, niters); 

  printf("float -> float\n");
  convert<float, float>(q, nelems, niters); 
  printf("float -> half\n");
  convert<half, float>(q, nelems, niters); 
  printf("float -> int\n");
  convert<int, float>(q, nelems, niters); 
  printf("float -> char\n");
  convert<char, float>(q, nelems, niters); 
  printf("float -> uchar\n");
  convert<uchar, float>(q, nelems, niters); 

  printf("int -> int\n");
  convert<int, int>(q, nelems, niters); 
  printf("int -> float\n");
  convert<float, int>(q, nelems, niters); 
  printf("int -> half\n");
  convert<half, int>(q, nelems, niters); 
  printf("int -> char\n");
  convert<char, int>(q, nelems, niters); 
  printf("int -> uchar\n");
  convert<uchar, int>(q, nelems, niters); 

  printf("char -> int\n");
  convert<int, char>(q, nelems, niters); 
  printf("char -> float\n");
  convert<float, char>(q, nelems, niters); 
  printf("char -> half\n");
  convert<half, char>(q, nelems, niters); 
  printf("char -> char\n");
  convert<char, char>(q, nelems, niters); 
  printf("char -> uchar\n");
  convert<uchar, char>(q, nelems, niters); 

  printf("uchar -> int\n");
  convert<int, uchar>(q, nelems, niters); 
  printf("uchar -> float\n");
  convert<float, uchar>(q, nelems, niters); 
  printf("uchar -> half\n");
  convert<half, uchar>(q, nelems, niters); 
  printf("uchar -> char\n");
  convert<char, uchar>(q, nelems, niters); 
  printf("uchar -> uchar\n");
  convert<uchar, uchar>(q, nelems, niters); 

  return 0;
}
