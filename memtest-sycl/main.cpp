#include <stdio.h>
#include "common.h"
#include "kernels.cpp"

void check (queue &q, buffer<unsigned, 1> &err_cnt) {
  unsigned err = 0;
  // read error
  q.submit([&] (handler &cgh) {
    auto acc = err_cnt.get_access<sycl_read>(cgh);
    cgh.copy(acc, &err);
  }).wait();

  printf("%s\n", (err != 0) ? "FAIL" : "PASS");

  // reset
  q.submit([&] (handler &cgh) {
    auto acc = err_cnt.get_access<sycl_write>(cgh);
    cgh.fill(acc, 0u);
  });
}

// moving inversion tests with complementary patterns
void moving_inversion (
  queue &q,
  buffer<unsigned, 1> &err_cnt,
  buffer<unsigned long, 1> &err_addr,
  buffer<unsigned long, 1> &err_expect,
  buffer<unsigned long, 1> &err_current,
  buffer<unsigned long, 1> &err_second_read,
  buffer<char, 1> &dev_mem,
  unsigned long mem_size,
  unsigned long p1)
{
  
  unsigned long p2 = ~p1;
  range<1> gws (64*1024);
  range<1> lws (64);

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_write>(cgh);
    cgh.parallel_for<class test_write_pattern>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      kernel_write(item, mem.get_pointer(), mem_size, p1);
    });
  });

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_read_write>(cgh);
    auto cnt = err_cnt.get_access<sycl_read_write>(cgh);
    auto addr = err_addr.get_access<sycl_write>(cgh);
    auto exp = err_expect.get_access<sycl_write>(cgh);
    auto curr = err_current.get_access<sycl_write>(cgh);
    auto read = err_second_read.get_access<sycl_write>(cgh);
    cgh.parallel_for<class test_pattern_readwrite>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      kernel_read_write(item, mem.get_pointer(), mem_size,
                 p1, p2,
                 cnt.get_pointer(),
                 addr.get_pointer(),
                 exp.get_pointer(),
                 curr.get_pointer(),
                 read.get_pointer());
    });
  });

  p1 = p2;

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_read>(cgh);
    auto cnt = err_cnt.get_access<sycl_read_write>(cgh);
    auto addr = err_addr.get_access<sycl_write>(cgh);
    auto exp = err_expect.get_access<sycl_write>(cgh);
    auto curr = err_current.get_access<sycl_write>(cgh);
    auto read = err_second_read.get_access<sycl_write>(cgh);
    cgh.parallel_for<class test_pattern_read>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      kernel_read(item, mem.get_pointer(), mem_size,
                 p1, 
                 cnt.get_pointer(),
                 addr.get_pointer(),
                 exp.get_pointer(),
                 curr.get_pointer(),
                 read.get_pointer());
    });
  });

  check(q, err_cnt);
}

int main() {

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  unsigned err_count = 0;

  buffer<unsigned, 1> err_cnt (&err_count, 1);
  buffer<unsigned long, 1> err_addr (MAX_ERR_RECORD_COUNT);
  buffer<unsigned long, 1> err_expect (MAX_ERR_RECORD_COUNT);
  buffer<unsigned long, 1> err_current (MAX_ERR_RECORD_COUNT);
  buffer<unsigned long, 1> err_second_read (MAX_ERR_RECORD_COUNT);

  unsigned long mem_size = 1024*1024*1024;
  unsigned long msize_in_mb = mem_size >>20;
  buffer<char, 1> dev_mem (mem_size);

  range<1> gws0 (64*1024);
  range<1> lws0 (64);

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class test_k0_write>(nd_range<1>(gws0, lws0), [=] (nd_item<1> item) {
      kernel0_write(item, mem.get_pointer(), mem_size);
    });
  });

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_read>(cgh);
    auto cnt = err_cnt.get_access<sycl_read_write>(cgh);
    auto addr = err_addr.get_access<sycl_write>(cgh);
    auto exp = err_expect.get_access<sycl_write>(cgh);
    auto curr = err_current.get_access<sycl_write>(cgh);
    auto read = err_second_read.get_access<sycl_write>(cgh);
    cgh.parallel_for<class test_k0_read>(nd_range<1>(gws0, lws0), [=] (nd_item<1> item) {
      kernel0_read(item, mem.get_pointer(), mem_size,
                 cnt.get_pointer(),
                 addr.get_pointer(),
                 exp.get_pointer(),
                 curr.get_pointer(),
                 read.get_pointer());
    });
  });

  check(q, err_cnt);

  range<1> gws1 (64*1024);
  range<1> lws1 (64);

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class test_k1_write>(nd_range<1>(gws1, lws1), [=] (nd_item<1> item) {
      kernel1_write(item, mem.get_pointer(), mem_size);
    });
  });

  q.submit([&] (handler &cgh) {
    auto mem = dev_mem.get_access<sycl_read>(cgh);
    auto cnt = err_cnt.get_access<sycl_read_write>(cgh);
    auto addr = err_addr.get_access<sycl_write>(cgh);
    auto exp = err_expect.get_access<sycl_write>(cgh);
    auto curr = err_current.get_access<sycl_write>(cgh);
    auto read = err_second_read.get_access<sycl_write>(cgh);
    cgh.parallel_for<class test_k1_read>(nd_range<1>(gws1, lws1), [=] (nd_item<1> item) {
      kernel1_read(item, mem.get_pointer(), mem_size,
                 cnt.get_pointer(),
                 addr.get_pointer(),
                 exp.get_pointer(),
                 curr.get_pointer(),
                 read.get_pointer());
    });
  });
  check(q, err_cnt);

  unsigned long p1 = 0;
  unsigned long p2 = ~p1;
  moving_inversion (q, err_cnt, err_addr, err_expect, err_current,
                    err_second_read, dev_mem, mem_size, p1);
}
