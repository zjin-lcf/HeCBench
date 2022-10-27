#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include "common.h"
#include "benchmark.h"
#include "kernels.h"

void run_benchmark(queue &q)
{
  int i, j, cnt, val_ref, val_eff;
  uint64_t time_vals[SIZES_CNT_MAX][BASES_CNT_MAX][2];

  buffer<uint32_t, 1> d_bases32 (bases32, sizeof(bases32) / sizeof(bases32[0]));

  buffer<uint32_t, 1> d_n32 (BENCHMARK_ITERATIONS);

  int val_dev;
  buffer<int, 1> d_val(1);

  range<1> gws ((BENCHMARK_ITERATIONS + 255) / 256 * 256);
  range<1> lws (256);

  printf("Starting benchmark...\n");

  bool ok = true;
  double mr32_sf_time = 0.0, mr32_eff_time = 0.0;

  for (i = 0; i < SIZES_CNT32; i++) {
    val_ref = val_eff = 0;

    q.submit([&] (handler &cgh) {
      auto acc = d_n32.get_access<sycl_discard_write>(cgh);
      cgh.copy(n32[i], acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_val.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0);
    });

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_eff += efficient_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][0] = elapsed_time(start);
    }

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_ref += straightforward_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][1] = elapsed_time(start);
    }

    // verify the results of simple and efficient versions on a host
    if (val_ref != val_eff) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_ref = %d, val_eff = %d\n", val_ref, val_eff);
      break;
    }

    q.wait();
    auto start = std::chrono::steady_clock::now();

    // the efficient version is faster than the simple version on a device
    q.submit([&] (handler &cgh) {
      auto b = d_bases32.get_access<sycl_read>(cgh);
      auto n = d_n32.get_access<sycl_read>(cgh);
      auto v = d_val.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class simple>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
       mr32_sf(item, b.get_pointer(), n.get_pointer(), 
               v.get_pointer(), BENCHMARK_ITERATIONS);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    mr32_sf_time += time;

    q.submit([&] (handler &cgh) {
      auto acc = d_val.get_access<sycl_read>(cgh);
      cgh.copy(acc, &val_dev);
    }).wait();

    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }

    q.submit([&] (handler &cgh) {
      auto acc = d_val.get_access<sycl_discard_write>(cgh);
      cgh.fill(acc, 0);
    });

    q.wait();
    start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      auto b = d_bases32.get_access<sycl_read>(cgh);
      auto n = d_n32.get_access<sycl_read>(cgh);
      auto v = d_val.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class efficient>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
       mr32_eff(item, b.get_pointer(), n.get_pointer(), 
                v.get_pointer(), BENCHMARK_ITERATIONS);
      });
    });

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    mr32_eff_time += time;

    q.submit([&] (handler &cgh) {
      auto acc = d_val.get_access<sycl_read>(cgh);
      cgh.copy(acc, &val_dev);
    }).wait();

    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }
  }

  printf("Total kernel execution time (mr32_simple  ): %f (ms)\n", mr32_sf_time * 1e-6);
  printf("Total kernel execution time (mr32_efficent): %f (ms)\n", mr32_eff_time * 1e-6);
  printf("%s\n", ok ? "PASS" : "FAIL");

  // device results are not included
  print_results(bits32, SIZES_CNT32, BASES_CNT32, time_vals);
}

int main()
{
#ifdef _WIN32
  system("mode CON: COLS=98");
#endif

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  printf("Setting random primes...\n");
  set_nprimes();
  run_benchmark(q);

  printf("Setting random odd integers...\n");
  set_nintegers();
  run_benchmark(q);

  return 0;
}
