#include <iostream>
#include <cstdio>
#include <chrono>
#include "common.h"

#define NOW std::chrono::high_resolution_clock::now()

#include "fastdiv.h"
#include "kernels.h"

// Functional test returns 1 when it fails; otherwise it returns 0
int test(queue &q)
{
  const int blocks = 256;
  const int divisor_count = 100000;
  const int divident_count = 1000000;

  int grids = (divident_count + blocks - 1) / blocks;

  range<1> gws (grids * blocks);
  range<1> lws (blocks);

  int buf[4];
  buffer<int, 1> d_buf (4);

  std::cout << "Running functional test on " << divisor_count << " divisors, with " 
            << grids * blocks << " dividents for each divisor" << std::endl;

  for(int d = 1; d < divisor_count; ++d)
  {
    for(int sign = 1; sign >= -1; sign -= 2)
    {
      int divisor = d * sign;
      q.submit([&] (handler &cgh) {
        auto acc = d_buf.get_access<sycl_discard_write>(cgh);
        cgh.fill(acc, 0);
      });
      q.submit([&] (handler &cgh) {
        auto acc = d_buf.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class test>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          check(item, divisor, acc.get_pointer());
        });
      });
      q.submit([&] (handler &cgh) {
        auto acc = d_buf.get_access<sycl_read>(cgh);
        cgh.copy(acc, buf);
      }).wait();

      if (buf[0] > 0)
      {
        std::cout << buf[0] << " wrong results, one of them is for divident " 
                  << buf[1] << ", correct quotient = " << buf[2] 
                  << ", fast computed quotient = " << buf[3] << std::endl;
        return 1;
      }
    }
  }
  return 0;
}

int main(int argc, char* argv[])
{
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  const int grids = 32 * 1024;
  const int blocks = 256;

  // performance evaluation after functional test is done
  if (test(q)) return 1;

  range<1> gws (grids * blocks);
  range<1> lws (blocks);
  // warmup may be needed for accurate performance measurement with chrono
  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class warm_thru>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        throughput_test<int>(item, 3, 5, 7, 0, 0);
      });
    });
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class warm_thru_fast>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        throughput_test<int_fastdiv>(item, 3, 5, 7, 0, 0);
      });
    });
  }
  q.wait();

  std::cout << "THROUGHPUT TEST" << std::endl;

  std::cout << "Benchmarking plain division by constant... ";
  auto start = NOW;

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class bench_thru>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        throughput_test<int>(item, 3, 5, 7, 0, 0);
      });
    });
  q.wait();

  auto end = NOW;
  std::chrono::duration<double> elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class bench_thru_fast>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        throughput_test<int_fastdiv>(item, 3, 5, 7, 0, 0);
      });
    });
  q.wait();

  end = NOW;
  std::chrono::duration<double> elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;

  // warmup
  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class warm_lat>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        latency_test<int>(item, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
      });
    });
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class warm_lat_fast>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        latency_test<int_fastdiv>(item, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
      });
    });
  }
  q.wait();

  std::cout << "LATENCY TEST" << std::endl;
  std::cout << "Benchmarking plain division by constant... ";
  start = NOW;

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class bench_lat>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        latency_test<int>(item, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
      });
    });
  q.wait();

  end = NOW;
  elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class bench_lat_fast>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        latency_test<int_fastdiv>(item, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
      });
    });
  q.wait();

  end = NOW;
  elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;
  return 0;
}
