#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }

  // specify the number of test cases
  const int iteration = atoi(argv[1]);

  // number of elements to reverse
  const int len = 256;

  // device result
  int test[len];

  // expected results after reverse operations even/odd times
  int error = 0;
  int gold_odd[len];
  int gold_even[len];

  for (int i = 0; i < len; i++) {
    gold_odd[i] = len-i-1;
    gold_even[i] = i;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);
  
  buffer<int, 1> d_test (len);
  range<1> gws (len);
  range<1> lws (len);

  srand(123);
  for (int i = 0; i < iteration; i++) {

    const int count = rand() % 10000 + 100;  // bound the reverse range

    q.submit([&](handler &cgh) {
      auto acc = d_test.get_access<sycl_discard_write>(cgh);
      cgh.copy(gold_even, acc);
    });
      
    for (int j = 0; j < count; j++)
      q.submit([&](handler &cgh) {
        accessor <int, 1, sycl_read_write, access::target::local> s (len, cgh);
        auto acc = d_test.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class blockReverse>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
          int t = item.get_local_id(0);
          s[t] = acc[t];
          item.barrier(access::fence_space::local_space);
          acc[t] = s[len-t-1];
        });
      });

    q.submit([&](handler &cgh) {
      auto acc = d_test.get_access<sycl_read>(cgh);
      cgh.copy(acc, test);
    }).wait();

    if (count % 2 == 0)
      error = memcmp(test, gold_even, len*sizeof(int));
    else
      error = memcmp(test, gold_odd, len*sizeof(int));
    
    if (error) break;
  }
  
  printf("%s\n", error ? "FAIL" : "PASS");

  return 0;
}
