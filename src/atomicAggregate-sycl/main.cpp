#include <chrono>
#include <stdio.h>
#include <sycl/sycl.hpp>

// reference
// https://stackoverflow.com/questions/59879285/whats-the-alternative-for-match-any-sync-on-compute-capability-6

#define warpSize 32

inline int ffs(int x) {
  return (x == 0) ? 0 : sycl::ctz(x) + 1;
}

// increment the value at ptr by 1 and return the old value
int atomicAggInc(int* ptr, sycl::nd_item<1> &item) {
  int mask;
  auto sg = item.get_sub_group();
  for (int i = 0; i < warpSize; i++) {
    unsigned long long tptr = sycl::select_from_group(sg, (unsigned long long)ptr, i);
    auto gb = sycl::ext::oneapi::group_ballot(sg, (tptr == (unsigned long long)ptr));
    unsigned my_mask;
    gb.extract_bits(my_mask, 0);
    if (i == (item.get_local_id(0) & (warpSize - 1))) mask = my_mask;
  }

  int leader = ffs(mask) - 1; // select a leader
  int res = 0;
  unsigned lane_id = item.get_local_id(0) % warpSize;
  if (lane_id == leader) {                 // leader does the update
    sycl::atomic_ref<int, 
       sycl::memory_order::relaxed,
       sycl::memory_scope::device,
       sycl::access::address_space::global_space> ao (*ptr);
    res = ao.fetch_add(sycl::popcount(mask));
  }
  res = sycl::select_from_group(sg, res, leader); // get leader’s old value
  return res + sycl::popcount(mask & ((1 << lane_id) - 1)); // compute old value
}

void k(int *d, sycl::nd_item<1> &item) {
  int *ptr = d + item.get_local_id(0) % 32;
  atomicAggInc(ptr, item);
}

const int ds = 32;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int *d_d, *h_d;
  h_d = new int[ds];

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  d_d = (int *)sycl::malloc_device(ds * sizeof(d_d[0]), q);
  q.memset(d_d, 0, ds * sizeof(d_d[0]));

  q.wait();

  sycl::range<1> gws (256 * 32 * 256);
  sycl::range<1> lws (256);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class kernel>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) 
                       [[sycl::reqd_sub_group_size(warpSize)]] {
        k(d_d, item);
      });
    });
  }

  q.wait();

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Total kernel time: %f (s)\n", time.count());

  q.memcpy(h_d, d_d, ds * sizeof(d_d[0])).wait();

  bool ok = true;
  for (int i = 0; i < ds; i++) {
    if (h_d[i] != 256 * 256 * repeat) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  sycl::free(d_d, q);
  delete [] h_d;
  return 0;
}
