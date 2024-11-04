/*
 * GPU-based parallel implementation of the IID test of NIST SP 800-90B.
 *
 * Copyright(C) < 2020 > <Yewon Kim>
 *
 * This program is free software : you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see < https://www.gnu.org/licenses/>.
 */

#include <sycl/sycl.hpp>
#include "header.h"
#include "kernel_functions.hpp"

/**
 * @brief Perform 10,000 iterations in parallel on the GPU. (exclude the compression test)
 *  - That is, perform {$N} iterations in the GPU and repeat ceil(10,000 / $N) times.
 *  - In each iteration, the original data are shuffled, 18 statistical tests are performed on the shuffled data,
 *    and the results are compared with the original test statistics.
 * @param double $gpu_runtime: Runtime of 10,000 iterations measured by the C++ chrono 
 * @param uint32_t $counts[]: The counters, that is original test statistics's rankings
 * @param double $results[]: The results of 19 statistical tests on the original data
 * @param double $mean: Mean value of the original data(input)
 * @param double $median: Median value of the original data(input)
 * @param uint8_t $data[]: The original data(input), which consists of (noise) samples
 * @param uint32_t $size: The size of sample in bits (1~8)
 * @param uint32_t $len: The number of samples in the original data
 * @param uint32_t $N: The number of iterations processing in parallel on the GPU
 * @param uint32_t $num_block: The number of thread blocks
 * @param uint32_t $num_thread: The number of threads per block
 * @return bool $iid_check_result
 */
bool gpu_permutation_testing(double *gpu_runtime, uint32_t *counts, double *results,
                             double mean, double median, uint8_t *data, uint32_t size,
                             uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread)
{
  uint32_t i;
  uint8_t num_runtest = 0;
  uint32_t loop = 10000 / N;
  if ((10000 % N) != 0)  loop++;
  uint32_t blen = 0;
  if (size == 1) {
    blen = len / 8;
    if ((len % 8) != 0)  blen++;
  }
  size_t Nlen = (size_t)N * len;
  size_t Nblen = (size_t)N * blen;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint8_t *dev_bNdata;

  /* allocate memory on the GPU. */
  uint8_t *dev_data = sycl::malloc_device<uint8_t>(len, q);
  uint8_t *dev_Ndata = (uint8_t *)sycl::malloc_device(Nlen, q);

  double *dev_results = sycl::malloc_device<double>(18, q);
  uint32_t *dev_cnt = sycl::malloc_device<uint32_t>(54, q);
  if (size == 1)
    dev_bNdata = sycl::malloc_device<uint8_t>(Nblen, q);

  /* copy data from the CPU to the GPU. */
  q.memcpy(dev_data, data, len * sizeof(uint8_t));
  q.memcpy(dev_results, results, 18 * sizeof(double));
  q.memcpy(dev_cnt, counts, 54 * sizeof(uint32_t));
  q.wait();

  /* start the timer. */
  auto start = std::chrono::steady_clock::now();

  sycl::range<1> gws (num_block * num_thread);
  sycl::range<1> gws2 (4 * num_block * num_thread);
  sycl::range<1> gws3 (2 * num_block * num_thread);
  sycl::range<1> lws (num_thread);

  /* generate {$N} shuffled data by permuting the original data {$N} times in parallel.
   * perform 18 statistical tests on each of {$N} shuffled data and compares the shuffled and original test statistics in parallel.
   */
  for (i = 0; i < loop; i++) {
    if (size == 1) {
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        binary_shuffling_kernel(dev_Ndata, dev_bNdata,
                                dev_data, len,
                                blen, N, item);
      });

      q.parallel_for(sycl::nd_range<1>(gws2, lws), [=](sycl::nd_item<1> item) {
        binary_statistical_tests_kernel(
          dev_cnt, dev_results, mean, median, dev_Ndata,
          dev_bNdata, size, len, blen, N, num_block,
          item);
      });

      /* copy data from the GPU to the CPU. */
      q.memcpy(counts, dev_cnt, 54 * sizeof(uint32_t)).wait();
      num_runtest = 0;
      for (int t = 0; t < 18; t++) {
        if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
          num_runtest++;
      }
      if (num_runtest == 18)
        break;
    }
    else {
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        shuffling_kernel(dev_Ndata, dev_data, len, N, item);
      });

      q.parallel_for(sycl::nd_range<1>(gws3, lws), [=](sycl::nd_item<1> item) {
        statistical_tests_kernel(dev_cnt, dev_results, mean, median, dev_Ndata,
                                 size, len, N, num_block, item);
      });

      /* copy data from the GPU to the CPU. */
      q.memcpy(counts, dev_cnt, 54 * sizeof(uint32_t)).wait();
      num_runtest = 0;
      for (int t = 0; t < 18; t++) {
        if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
          num_runtest++;
      }
      if (num_runtest == 18)
        break;
    }
  }

  /* stop the timer. */
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  /* calculate the run-time of the permutation testing */
  *gpu_runtime = (double)time * 1e-9;

  sycl::free(dev_data, q);
  sycl::free(dev_Ndata, q);
  sycl::free(dev_results, q);
  sycl::free(dev_cnt, q);
  if (size == 1)
    sycl::free(dev_bNdata, q);

  if (num_runtest == 18) // IID
    return true;
  else // Non-IID
    return false;
}
