#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

void reverse (int* d, const int len, sycl::nd_item<3> item_ct1, int *s)
{

  int t = item_ct1.get_local_id(2);
  int tr = len-t-1;
  s[t] = d[t];
  item_ct1.barrier();
  d[t] = s[tr];
}

int main() {
  const int len = 256;
  const int iteration = 1 << 16;
  int d[len];
  for (int i = 0; i < len; i++) d[i] = i;

  int *dd;
  dpct::dpct_malloc((void **)&dd, sizeof(int) * len);
  dpct::dpct_memcpy(dd, d, sizeof(int) * len, dpct::host_to_device);
  for (int i = 0; i <= iteration; i++)
  {
    dpct::buffer_t dd_buf_ct0 = dpct::get_buffer(dd);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          s_acc_ct1(sycl::range<1>(256), cgh);
      auto dd_acc_ct0 =
          dd_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         reverse((int *)(&dd_acc_ct0[0]), len, item_ct1,
                                 s_acc_ct1.get_pointer());
                       });
    });
  }
  dpct::dpct_memcpy(d, dd, sizeof(int) * len, dpct::device_to_host);
  dpct::dpct_free(dd);
  for (int i = 0; i < len; i++) assert(d[i] == len-i-1);

  return 0;
}
