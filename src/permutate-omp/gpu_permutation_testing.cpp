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

#include <omp.h>
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

  uint8_t *Ndata = (uint8_t *) malloc (Nlen);

  // memory allocation anyway 
  uint8_t *bNdata = (uint8_t*) malloc (Nblen);

  /* copy data from the CPU to the GPU. */
  #pragma omp target data map (to: data[0:len], \
                                   results[0:18], \
                                   counts[0:54]) \
                          map (alloc: Ndata[0:Nlen], \
                                      bNdata[0:Nblen])
  {
    /* start the timer. */
    auto start = std::chrono::steady_clock::now();

    /* generate {$N} shuffled data by permuting the original data {$N} times in parallel.
     * perform 18 statistical tests on each of {$N} shuffled data and compares the shuffled and original test statistics in parallel.
     */
    for (i = 0; i < loop; i++) {
      if (size == 1) {
        binary_shuffling_kernel(Ndata, bNdata, data, len, blen, N, num_block, num_thread);

        binary_statistical_tests_kernel(counts, results, mean, median, Ndata,
                                        bNdata, size, len, blen, N, num_block, num_thread);

        /* copy data from the GPU to the CPU. */
        #pragma omp target update from (counts[0:54])

        num_runtest = 0;
        for (int t = 0; t < 18; t++) {
          if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
            num_runtest++;
        }
        if (num_runtest == 18)
          break;
      }
      else {
        shuffling_kernel(Ndata, data, len, N, num_block, num_thread);

        statistical_tests_kernel(counts, results, mean, median, Ndata,
                                 size, len, N, num_block, num_thread);

        /* copy data from the GPU to the CPU. */
        #pragma omp target update from (counts[0:54])

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

  } // OpenMP

  free(Ndata);
  free(bNdata);

  if (num_runtest == 18) // IID
    return true;
  else // Non-IID
    return false;
}
