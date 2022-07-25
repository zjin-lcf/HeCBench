#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

#define syncthreads() item.barrier(access::fence_space::local_space)
#define threadfence() item.mem_fence(access::fence_space::global_space)

void sum (
    nd_item<1> &item,
    float *__restrict partialSum,
    bool *__restrict isLastBlockDone,
    const float*__restrict array,
    const int N,
    unsigned int *__restrict count,
    volatile float*__restrict result)
{

  // Each block sums a subset of the input array.
  unsigned int bid = item.get_group(0);
  unsigned int num_blocks = item.get_group_range(0);
  unsigned int block_size = item.get_local_range(0);
  unsigned int lid = item.get_local_id(0);
  unsigned int gid = bid * block_size + lid;
  
  auto ao = ext::oneapi::atomic_ref<float, 
            ext::oneapi::memory_order::relaxed,
            ext::oneapi::memory_scope::work_group,
            access::address_space::local_space> (*partialSum);

  if (lid == 0) *partialSum = 0;
  syncthreads();

  if (gid < N)
    ao.fetch_add(array[gid]);

  syncthreads();

  if (lid == 0) {

    // Thread 0 of each block stores the partial sum
    // to global memory. The compiler will use 
    // a store operation that bypasses the L1 cache
    // since the "result" variable is declared as
    // volatile. This ensures that the threads of
    // the last block will read the correct partial
    // sums computed by all other blocks.
    result[bid] = *partialSum;

    // Thread 0 makes sure that the incrementation
    // of the "count" variable is only performed after
    // the partial sum has been written to global memory.
    threadfence();

    // Thread 0 signals that it is done.
    auto count_ao = ext::oneapi::atomic_ref<unsigned int, 
                    ext::oneapi::memory_order::relaxed,
                    ext::oneapi::memory_scope::device,
                    access::address_space::global_space> (*count);
    unsigned int value = count_ao.fetch_add(1u);

    // Thread 0 determines if its block is the last
    // block to be done.
    *isLastBlockDone = (value == (num_blocks - 1));
  }

  // Synchronize to make sure that each thread reads
  // the correct value of isLastBlockDone.
  syncthreads();

  if (*isLastBlockDone) {

    // The last block sums the partial sums
    // stored in result[0 .. num_blocks-1]
    if (lid == 0) *partialSum = 0;
    syncthreads();

    for (int i = lid; i < num_blocks; i += block_size)
      ao.fetch_add(result[i]);

    syncthreads();

    if (lid == 0) {

      // Thread 0 of last block stores the total sum
      // to global memory and resets the count
      // varialble, so that the next kernel call
      // works properly.
      result[0] = *partialSum;
      *count = 0;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);

  float* h_array = (float*) malloc (N * sizeof(float));
  float h_sum;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  const int blocks = 256;
  const int grids = (N + blocks - 1) / blocks;
  range<1> gws (grids * blocks);
  range<1> lws (blocks);

  buffer<float, 1> d_result (grids);

  buffer<float, 1> d_array (N);

  buffer<unsigned int, 1> d_count (1);

  q.submit([&] (handler &cgh) {
    auto count = d_count.get_access<sycl_discard_write>(cgh);
    cgh.fill(count, 0u);
  });

  bool ok = true;
  double time = 0.0;

  for (int n = 0; n < repeat; n++) {

    for (int i = 0; i < N; i++)
      h_array[i] = -1.f;

    q.submit([&] (handler &cgh) {
      auto acc = d_array.get_access<sycl_discard_write>(cgh);
      cgh.copy(h_array, acc);
    }).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      auto array = d_array.get_access<sycl_read>(cgh);
      auto count = d_count.get_access<sycl_read_write>(cgh);
      auto result = d_result.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> lsum (1, cgh);
      accessor<bool, 1, sycl_read_write, access::target::local> isLastBlockDone (1, cgh);
      cgh.parallel_for<class reduce>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        sum (item, lsum.get_pointer(), isLastBlockDone.get_pointer(),
             array.get_pointer(), N, count.get_pointer(), result.get_pointer());
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    q.submit([&] (handler &cgh) {
      auto acc = d_result.get_access<sycl_read>(cgh, range<1>(1));
      cgh.copy(acc, &h_sum);
    }).wait();

    if (h_sum != -1.f * N) {
      ok = false;
      break;
    }
  }

  if (ok) printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
