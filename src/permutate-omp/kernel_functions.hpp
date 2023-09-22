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

#ifndef _KERNEL_FUNCTIONS_H_
#define _KERNEL_FUNCTIONS_H_
#include "header.h"
#include "device_functions.hpp"

uint32_t LCG_random (uint64_t * seed)
{
  // LCG parameters
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

/**
 * @brief The kernel function: Generate {$N} shuffled data by permuting the original data {$N} times in parallel.
 * @param uint8_t $Ndata[] {$N} shuffled data
 * @param const uint8_t $data[] Original data(input)
 * @param const uint32_t $len: Number of samples in the original data
 * @param const uint32_t $N: Number of iterations processing in parallel on the GPU
 * @return void
 */

void shuffling_kernel(
  uint8_t *Ndata, const uint8_t *data,
  const uint32_t len, const uint32_t N,
  const uint32_t num_block, const uint32_t num_thread)
{
  uint32_t size = num_block * num_thread; 

  #pragma omp target teams distribute parallel for thread_limit(num_thread)
  for (uint64_t tid = 0; tid < size; tid++) {

    uint64_t i = 0, j = len - 1, random = 0;
    uint8_t tmp = 0;
    uint64_t idx = 0, idx2 = 0;

    uint64_t seed = 0;

    for (i = 0; i < len; i++) {
      idx= i * N + tid;
      Ndata[idx] = data[i];
      seed += Ndata[idx];
    }
    seed = seed ^ tid;

    while (j > 0) {
      random = LCG_random(&seed) % j;
      idx = random * N + tid;
      idx2 = j * N + tid;
      tmp = Ndata[idx];
      Ndata[idx] = Ndata[idx2];
      Ndata[idx2] = tmp;
      j--;
    }
  }
}

/**
 * @brief The kernel function: Perform 18 statistical tests on each of {$N} shuffled data,
 *                 and compares the shuffled and original test statistics in parallel.
 * @param uint32_t $counts[]: The counters, that is original test statistics's rankings
 * @param const double $results[]: Results of 19 Statisitcal tests on the original data(input)
 * @param const double $mean: Mean value of the original data(input)
 * @param const double $median: Median value of the original data(input)
 * @param const uint8_t $Ndata: {$N} shuffled data
 * @param const uint32_t $size: The size of sample in bits (1~8)
 * @param const uint32_t $len: The number of samples in the original data
 * @param const uint32_t $N: The number of iterations processing in parallel on the GPU
 * @param constuint32_t $num_block: The number of thread blocks
 * @return void
 */

void statistical_tests_kernel(
  uint32_t *counts, const double *results, const double mean, const double median,
  const uint8_t *Ndata, const uint32_t size, const uint32_t len, const uint32_t N,
  const uint32_t num_block,
  const uint32_t num_thread)
{
  #pragma omp target teams num_teams(2*num_block) thread_limit(num_thread)
  {
    #pragma omp parallel 
    {
      uint32_t gid = omp_get_team_num();
      uint32_t tid = omp_get_thread_num() + (gid % num_block) * num_thread;

      if ((gid / num_block) == 0) {
        double result1 = 0, result2 = 0;
        dev_test7_8(&result1, &result2, Ndata, size, len, N, tid);
        if (result1 > results[6]) {
          #pragma omp atomic update
          counts[18]++;
        }
        else if (result1 == results[6]) {
          #pragma omp atomic update
          counts[19]++;
        }
        else {
          #pragma omp atomic update
          counts[20]++;
        }

        if (result2 > results[7]) {
          #pragma omp atomic update
          counts[21]++;
        }
        else if (result2 == results[7]) {
          #pragma omp atomic update
          counts[22]++;
        }
        else {
          #pragma omp atomic update
          counts[23]++;
        }
      } else if ((gid / num_block) == 1) {
        double result1 = 0, result2 = 0, result3 = 0, result4 = 0, result5 = 0;

        dev_test1(&result1, mean, Ndata, len, N, tid);
        if ((float)result1 > (float)results[0]) {
          #pragma omp atomic update
          counts[0]++;
        }
        else if ((float)result1 == (float)results[0]) {
          #pragma omp atomic update
          counts[1]++;
        }
        else {
          #pragma omp atomic update
          counts[2]++;
        }

        dev_test2_6(&result1, &result2, &result3, &result4, &result5, median, Ndata, len, N, tid);
        if (result1 > results[1]) {
          #pragma omp atomic update
          counts[3]++;
        }
        else if (result1 == results[1]) {
          #pragma omp atomic update
          counts[4]++;
        }
        else {
          #pragma omp atomic update
          counts[5]++;
        }

        if (result2 > results[2]) {
          #pragma omp atomic update
          counts[6]++;
        }
        else if (result2 == results[2]) {
          #pragma omp atomic update
          counts[7]++;
        }
        else {
          #pragma omp atomic update
          counts[8]++;
        }
        if (result3 > results[3]) {
          #pragma omp atomic update
          counts[9]++;
        }
        else if (result3 == results[3]) {
          #pragma omp atomic update
          counts[10]++;
        }
        else {
          #pragma omp atomic update
          counts[11]++;
        }

        if (result4 > results[4]) {
          #pragma omp atomic update
          counts[12]++;
        }
        else if (result4 == results[4]) {
          #pragma omp atomic update
          counts[13]++;
        }
        else {
          #pragma omp atomic update
          counts[14]++;
        }

        if (result5 > results[5]) {
          #pragma omp atomic update
          counts[15]++;
        }
        else if (result5 == results[5]) {
          #pragma omp atomic update
          counts[16]++;
        }
        else {
          #pragma omp atomic update
          counts[17]++;
        }

        dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 1);
        if (result1 > results[8]) {
          #pragma omp atomic update
          counts[24]++;
        }
        else if (result1 == results[8]) {
          #pragma omp atomic update
          counts[25]++;
        }
        else {
          #pragma omp atomic update
          counts[26]++;
        }

        if (result2 > results[13]) {
          #pragma omp atomic update
          counts[39]++;
        }
        else if (result2 == results[13]) {
          #pragma omp atomic update
          counts[40]++;
        }
        else {
          #pragma omp atomic update
          counts[41]++;
        }

        dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 2);
        if (result1 > results[9]) {
          #pragma omp atomic update
          counts[27]++;
        }
        else if (result1 == results[9]) {
          #pragma omp atomic update
          counts[28]++;
        }
        else {
          #pragma omp atomic update
          counts[29]++;
        }

        if (result2 > results[14]) {
          #pragma omp atomic update
          counts[42]++;
        }
        else if (result2 == results[14]) {
          #pragma omp atomic update
          counts[43]++;
        }
        else {
          #pragma omp atomic update
          counts[44]++;
        }

        dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 8);
        if (result1 > results[10]) {
          #pragma omp atomic update
          counts[30]++;
        }
        else if (result1 == results[10]) {
          #pragma omp atomic update
          counts[31]++;
        }
        else {
          #pragma omp atomic update
          counts[32]++;
        }

        if (result2 > results[15]) {
          #pragma omp atomic update
          counts[45]++;
        }
        else if (result2 == results[15]) {
          #pragma omp atomic update
          counts[46]++;
        }
        else {
          #pragma omp atomic update
          counts[47]++;
        }

        dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 16);
        if (result1 > results[11]) {
          #pragma omp atomic update
          counts[33]++;
        }
        else if (result1 == results[11]) {
          #pragma omp atomic update
          counts[34]++;
        }
        else {
          #pragma omp atomic update
          counts[35]++;
        }

        if (result2 > results[16]) {
          #pragma omp atomic update
          counts[48]++;
        }
        else if (result2 == results[16]) {
          #pragma omp atomic update
          counts[49]++;
        }
        else {
          #pragma omp atomic update
          counts[50]++;
        }

        dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 32);
        if (result1 > results[12]) {
          #pragma omp atomic update
          counts[36]++;
        }
        else if (result1 == results[12]) {
          #pragma omp atomic update
          counts[37]++;
        }
        else {
          #pragma omp atomic update
          counts[38]++;
        }

        if (result2 > results[17]) {
          #pragma omp atomic update
          counts[51]++;
        }
        else if (result2 == results[17]) {
          #pragma omp atomic update
          counts[52]++;
        }
        else {
          #pragma omp atomic update
          counts[53]++;
        }
      }
    }
  }
}

/**
 * @brief The kernel function for binary data
 *  - Generate {$N} shuffled data by permuting the original data {$N} times in parallel and perform conversion II.
 * @param uint8_t $Ndata[] {$N} shuffled data
 * @param uint8_t $bNdata[] {$N} shuffled data after conversion II
 * @param const uint8_t $data[] Original data(input)
 * @param const uint32_t $len: Number of samples in the original data
 * @param const uint32_t $blen: Number of samples in the original data after converion II
 * @param const uint32_t $N: Number of iterations processing in parallel on the GPU
 * @return void
 */

void binary_shuffling_kernel(
  uint8_t *Ndata,
  uint8_t *bNdata, const uint8_t *data,
  const uint32_t len, const uint32_t blen, const uint32_t N,
  int num_block,
  int num_thread)
{
  uint32_t size = num_block * num_thread; 

  #pragma omp target teams distribute parallel for thread_limit(num_thread)
  for (uint32_t tid = 0; tid < size; tid++) {
    uint32_t i = 0, j = len - 1, random = 0;
    uint8_t tmp = 0;

    uint64_t seed = 0;

    for (i = 0; i < len; i++) {
      Ndata[i * N + tid] = data[i];
      seed += data[i]; 
    }
    seed = seed ^ tid;

    while (j > 0) {
      random = LCG_random(&seed) % j;
      tmp = Ndata[random * N + tid];
      Ndata[random * N + tid] = Ndata[j * N + tid];
      Ndata[j * N + tid] = tmp;
      j--;
    }

    for (i = 0; i < blen; i++) {
      tmp = (Ndata[8 * i * N + tid] & 0x1) << 7;
      tmp ^= (Ndata[(8 * i + 1) * N + tid] & 0x1) << 6;
      tmp ^= (Ndata[(8 * i + 2) * N + tid] & 0x1) << 5;
      tmp ^= (Ndata[(8 * i + 3) * N + tid] & 0x1) << 4;
      tmp ^= (Ndata[(8 * i + 4) * N + tid] & 0x1) << 3;
      tmp ^= (Ndata[(8 * i + 5) * N + tid] & 0x1) << 2;
      tmp ^= (Ndata[(8 * i + 6) * N + tid] & 0x1) << 1;
      tmp ^= (Ndata[(8 * i + 7) * N + tid] & 0x1);
      bNdata[i * N + tid] = tmp;
    }
  }
}


/**
 * @brief The kernel function for binary data
 *  - Perform 18 statistical tests on each of {$N} shuffled data and compares the shuffled and original test statistics in parallel.
 * @param uint32_t $counts[]: The counters, that is original test statistics's rankings
 * @param const double $results[]: Results of 19 Statisitcal tests on the original data(input)
 * @param const double $mean: Mean value of the original data(input)
 * @param const double $median: Median value of the original data(input)
 * @param const uint8_t $Ndata: {$N} shuffled data
 * @param const uint8_t $bNdata: {$N} shuffled data after conversion II
 * @param const uint32_t $size: The size of sample in bits (1~8)
 * @param const uint32_t $len: The number of samples in the original data
 * @param const uint32_t $blen: Number of samples in the original data after converion II
 * @param const uint32_t $N: The number of iterations processing in parallel on the GPU
 * @param constuint32_t $num_block: The number of thread blocks
 * @return void
 */

void binary_statistical_tests_kernel(
  uint32_t *counts, const double *results, const double mean, const double median,
  const uint8_t *Ndata, const uint8_t *bNdata, 
  const uint32_t size, const uint32_t len, const uint32_t blen,
  const uint32_t N,
  const uint32_t num_block,
  const uint32_t num_thread)
{
  #pragma omp target teams num_teams(4*num_block) thread_limit(num_thread)
  {
    #pragma omp parallel 
    {
      double result1 = 0, result2 = 0, result3 = 0;
      uint32_t gid = omp_get_team_num();
      uint32_t tid = omp_get_thread_num() + (gid % num_block) * num_thread;

      if ((gid / num_block) == 0) {
        dev_test1(&result1, mean, Ndata, len, N, tid);
        if ((float)result1 > (float)results[0]) {
          #pragma omp atomic update
          counts[0]++;
        }
        else if ((float)result1 == (float)results[0]) {
          #pragma omp atomic update
          counts[1]++;
        }
        else {
          #pragma omp atomic update
          counts[2]++;
        }
      }
      else if ((gid / num_block) == 1) {
        dev_test5_6(&result1, &result2, median, Ndata, len, N, tid);
        if (result1 > results[4]) {
          #pragma omp atomic update
          counts[12]++;
        }
        else if (result1 == results[4]) {
          #pragma omp atomic update
          counts[13]++;
        }
        else {
          #pragma omp atomic update
          counts[14]++;
        }
        if (result2 > results[5]) {
          #pragma omp atomic update
          counts[15]++;
        }
        else if (result2 == results[5]) {
          #pragma omp atomic update
          counts[16]++;
        }
        else {
          #pragma omp atomic update
          counts[17]++;
        }

        dev_binary_test2_4(&result1, &result2, &result3, bNdata, blen, N, tid);
        if (result1 > results[1]) {
          #pragma omp atomic update
          counts[3]++;
        }
        else if (result1 == results[1]) {
          #pragma omp atomic update
          counts[4]++;
        }
        else {
          #pragma omp atomic update
          counts[5]++;
        }
        if (result2 > results[2]) {
          #pragma omp atomic update
          counts[6]++;
        }
        else if (result2 == results[2]) {
          #pragma omp atomic update
          counts[7]++;
        }
        else {
          #pragma omp atomic update
          counts[8]++;
        }
        if (result3 > results[3]) {
          #pragma omp atomic update
          counts[9]++;
        }
        else if (result3 == results[3]) {
          #pragma omp atomic update
          counts[10]++;
        }
        else {
          #pragma omp atomic update
          counts[11]++;
        }
      }
      else if ((gid / num_block) == 2) {
        dev_test7_8(&result1, &result2, bNdata, 8, blen, N, tid);
        if (result1 > results[6]) {
          #pragma omp atomic update
          counts[18]++;
        }
        else if (result1 == results[6]) {
          #pragma omp atomic update
          counts[19]++;
        }
        else {
          #pragma omp atomic update
          counts[20]++;
        }
        if (result2 > results[7]) {
          #pragma omp atomic update
          counts[21]++;
        }
        else if (result2 == results[7]) {
          #pragma omp atomic update
             counts[22]++;
        }
        else {
          #pragma omp atomic update
          counts[23]++;
        }
      }
      else if ((gid / num_block) == 3) {
        dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 1);
        if (result1 > results[8]) {
          #pragma omp atomic update
          counts[24]++;
        }
        else if (result1 == results[8]) {
          #pragma omp atomic update
             counts[25]++;
        }
        else {
          #pragma omp atomic update
          counts[26]++;
        }
        if (result2 > results[13]) {
          #pragma omp atomic update
          counts[39]++;
        }
        else if (result2 == results[13]) {
          #pragma omp atomic update
          counts[40]++;
        }
        else {
          #pragma omp atomic update
          counts[41]++;
        }

        dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 2);
        if (result1 > results[9]) {
          #pragma omp atomic update
          counts[27]++;
        }
        else if (result1 == results[9]) {
          #pragma omp atomic update
          counts[28]++;
        }
        else {
          #pragma omp atomic update
          counts[29]++;
        }
        if (result2 > results[14]) {
          #pragma omp atomic update
          counts[42]++;
        }
        else if (result2 == results[14]) {
          #pragma omp atomic update
          counts[43]++;
        }
        else {
          #pragma omp atomic update
          counts[44]++;
        }

        dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 8);
        if (result1 > results[10]) {
          #pragma omp atomic update
          counts[30]++;
        }
        else if (result1 == results[10]) {
          #pragma omp atomic update
          counts[31]++;
        }
        else {
          #pragma omp atomic update
          counts[32]++;
        }
        if (result2 > results[15]) {
          #pragma omp atomic update
          counts[45]++;
        }
        else if (result2 == results[15]) {
          #pragma omp atomic update
          counts[46]++;
        }
        else {
          #pragma omp atomic update
          counts[47]++;
        }

        dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 16);
        if (result1 > results[11]) {
          #pragma omp atomic update
          counts[33]++;
        }
        else if (result1 == results[11]) {
          #pragma omp atomic update
          counts[34]++;
        }
        else {
          #pragma omp atomic update
          counts[35]++;
        }
        if (result2 > results[16]) {
          #pragma omp atomic update
          counts[48]++;
        }
        else if (result2 == results[16]) {
          #pragma omp atomic update
          counts[49]++;
        }
        else {
          #pragma omp atomic update
          counts[50]++;
        }

        dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 32);
        if (result1 > results[12]) {
          #pragma omp atomic update
          counts[36]++;
        }
        else if (result1 == results[12]) {
          #pragma omp atomic update
          counts[37]++;
        }
        else {
          #pragma omp atomic update
          counts[38]++;
        }
        if (result2 > results[17]) {
          #pragma omp atomic update
          counts[51]++;
        }
        else if (result2 == results[17]) {
          #pragma omp atomic update
          counts[52]++;
        }
        else {
          #pragma omp atomic update
          counts[53]++;
        }
      }
    }
  }
}

#endif
