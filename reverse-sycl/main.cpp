#include <assert.h>
#include "common.h"

int main() {
  const int len = 256;
  const int iteration = 1 << 20;
  int d[len];
  for (int i = 0; i < len; i++) d[i] = i;

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> dd (d, len);
  for (int i = 0; i <= iteration; i++) {
    q.submit([&](handler &h) {
      accessor <int, 1, sycl_read_write, access::target::local> s (len, h);
      auto d = dd.get_access<sycl_read_write>(h);
      h.parallel_for(nd_range<1>(range<1>(len), range<1>(len)), [=](nd_item<1> item) {
        int t = item.get_local_id(0);
        int tr = len-t-1;
        s[t] = d[t];
        item.barrier(access::fence_space::local_space);
        d[t] = s[tr];
      });
    });
  }
  q.wait();
  }
  for (int i = 0; i < len; i++) assert(d[i] == len-i-1);
  return 0;
}
