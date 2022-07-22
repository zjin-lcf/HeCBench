/*
   Shared memory speeds up performance when we need to access data frequently. 
   Here, the 1D stencil kernel adds all its neighboring data within a radius.

   The C model is added to verify the stencil result on a GPU
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"

#define RADIUS 7
#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int size = length;
  int pad_size = (length + RADIUS);

  // Alloc space for host copies of a, b, c and setup input values
  int* a = (int *)malloc(pad_size*sizeof(int)); 
  int* b = (int *)malloc(size*sizeof(int));

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  {
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();

    buffer<int, 1> d_a(a, pad_size, props);
    buffer<int, 1> d_b(b, size, props);

    size_t global_work_size = length;
    range<1> gws (global_work_size);
    range<1> lws (BLOCK_SIZE);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&](handler& cgh) { 
        auto in = d_a.get_access<sycl_read>(cgh);
        auto out = d_b.get_access<sycl_discard_write>(cgh);
        accessor <int, 1, sycl_read_write, access::target::local> temp (BLOCK_SIZE + 2 * RADIUS, cgh);
        cgh.parallel_for<class stencil1D>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int gindex = item.get_global_id(0);
          int lindex = item.get_local_id(0) + RADIUS;

          // Read input elements into shared memory
          temp[lindex] = in[gindex];

          // At both end of a block, the sliding window moves beyond the block boundary.
          if (item.get_local_id(0) < RADIUS) {
            temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
            temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
          }

          // Synchronize (ensure all the threads will be completed before continue)
          item.barrier(access::fence_space::local_space);

          // Apply the 1D stencil
          int result = 0;
          for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
            result += temp[lindex + offset];

          // Store the result
          out[gindex] = result; 

        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  // verification
  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Cleanup
  free(a);
  free(b); 
  return 0;
}
