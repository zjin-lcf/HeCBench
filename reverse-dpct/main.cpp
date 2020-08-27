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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int len = 256;
  const int iteration = 1 << 20;
  int d[len];
  for (int i = 0; i < len; i++) d[i] = i;

  int *dd;
  dd = sycl::malloc_device<int>(len, q_ct1);
  q_ct1.memcpy(dd, d, sizeof(int) * len).wait();
  for (int i = 0; i <= iteration; i++)
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          s_acc_ct1(sycl::range<1>(256), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         reverse(dd, len, item_ct1, s_acc_ct1.get_pointer());
                       });
    });
  q_ct1.memcpy(d, dd, sizeof(int) * len).wait();
  sycl::free(dd, q_ct1);
  for (int i = 0; i < len; i++) assert(d[i] == len-i-1);

  return 0;
}
