#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include "utils.h"
#include "block_load.h"
#include "block_store.h"

#define NUM 4

void reference (const float * __restrict__ A,
                unsigned char *out, const unsigned int n,
                const sycl::nd_item<3> &item)
{
  for (unsigned int idx = item.get_global_id(2);
       idx < n; idx += item.get_local_range(2) * item.get_group_range(2)) {
    out[idx] = int(A[idx]);
  }
}

template<int TH, int NUM_BLOCK>
void kernel(const float * __restrict__ A,
            unsigned char *out, const int n,
            const sycl::nd_item<3> &item)
{
  const int bid = item.get_group(2);
  const int base_idx = (bid * NUM_BLOCK);

  float vals[NUM];
  unsigned char qvals[NUM];

  typedef BlockLoad<float, TH, NUM> LoadFloat;
  typedef BlockStore<unsigned char, TH, NUM> StoreChar;

  sycl::multi_ptr<typename LoadFloat::TempStorage[1], sycl::access::address_space::local_space> p1 =
      sycl::ext::oneapi::group_local_memory_for_overwrite<typename LoadFloat::TempStorage[1]>(item.get_group());
  typename LoadFloat::TempStorage *loadf = *p1;

  sycl::multi_ptr<typename StoreChar::TempStorage[1], sycl::access::address_space::local_space> p2 =
      sycl::ext::oneapi::group_local_memory_for_overwrite<typename StoreChar::TempStorage[1]>(item.get_group());
  typename StoreChar::TempStorage *storec = *p2;

  for (unsigned int i = base_idx; i < n; i += item.get_group_range(2)*NUM_BLOCK)
  {
      unsigned int valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;

      LoadFloat(*loadf, item).Load(&(A[i]), vals, valid_items);

      #pragma unroll
      for(int j = 0; j < NUM; j++)
          qvals[j] = (int)vals[j];

      StoreChar(*storec, item).Store(&(out[i]), qvals, valid_items);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t n = (size_t)nrows * ncols;
  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(unsigned char);

  float *A = (float*) malloc (A_size);
  unsigned char *out = (unsigned char*) malloc (out_size);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{128.0, 127.0};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen); 
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_A;
  d_A = (float *)sycl::malloc_device(A_size, q);
  q.memcpy(d_A, A, A_size).wait();

  unsigned char *d_out;
  d_out = (unsigned char *)sycl::malloc_device(out_size, q);

  const int block_size = 256;

  int cu = q.get_device().get_info<sycl::info::device::max_compute_units>();
  sycl::range<3> gws (1, 1, 16 * cu * block_size);
  sycl::range<3> lws (1, 1, block_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) {
            reference(d_A, d_out, n, item);
          });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the reference kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) {
            kernel<block_size, block_size * NUM>(
                d_A, d_out, n, item);
          });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the blockAccess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(out, d_out, out_size).wait();

  bool error = false;
  for (unsigned int i = 0; i < n; i++) {
    unsigned char t = int(A[i]);
    if (out[i] != t) {
      printf("@%u: %u != %u\n", i, out[i], t);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  
  sycl::free(d_A, q);
  sycl::free(d_out, q);
  free(A);
  free(out);
  return 0;
}
