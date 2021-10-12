#include <cmath>
#include "common.h"

int main (void) {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);
  float in[8];

  buffer<float, 1> d_in (in, 8);

  range<3> a({1,2,3});
  range<3> b(1,2,3);
  // get the value of the first dimension
  printf("%d %d\n", a.get(0), b.get(0));

  q.submit([&] (handler &cgh) {
    auto in = d_in.get_access<sycl_discard_write>(cgh);
    accessor<float, 2, sycl_read_write, access::target::local> temp({2,4}, cgh);
    cgh.parallel_for<class k>(nd_range<1>(range<1>(1), range<1>(1)), [=] (nd_item<1> item) {
      int gid = item.get_global_id(0);
      for (int i = 0; i < 4; i++)
	for (int j = 0; j < 2; j++)
          temp[i][j] = sycl::cos((float)(i + j + gid));

      for (int i = 0; i < 4; i++)
	for (int j = 0; j < 2; j++)
          in[i*2+j] = temp[i][j];
    });
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_in.get_access<sycl_read>(cgh);
    cgh.copy(acc, in);
  }).wait();

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4; j++)
      printf("%f ", in[i*4+j]);
  return 0; 
}
