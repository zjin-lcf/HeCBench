#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "rand_helpers.h"
#include "constants.h"

void Hadamard4x4a(float &p, float &q, float &r, float &s)
{
  float t = (p + q + r + s) / 2;
  p = p - t;
  q = q - t;
  r = t - r;
  s = t - s;
}

void Hadamard4x4b(float &p, float &q, float &r, float &s)
{
  float t = (p + q + r + s) / 2;
  p = t - p;
  q = t - q;
  r = r - t;
  s = s - t;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // host buffers
  float *hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
  {
  	float x = RandN();
  	hostPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
  	rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> devPool(hostPool, WALLACE_TOTAL_POOL_SIZE);
  buffer<float, 1> devicerngChi2Corrections(rngChi2Corrections, WALLACE_CHI2_COUNT);
  buffer<float, 1> device_randomNumbers(WALLACE_OUTPUT_SIZE);
  devPool.set_final_data(nullptr);
  
  range<1> rng_wallace_grid(WALLACE_NUM_BLOCKS * WALLACE_NUM_THREADS);
  range<1> rng_wallace_threads(WALLACE_NUM_THREADS);
  const unsigned m_seed = 1;
  
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &h) {
      auto globalPool = devPool.get_access<sycl_read_write>(h);
      auto generatedRandomNumberPool = device_randomNumbers.get_access<sycl_write>(h);
      auto chi2Corrections = devicerngChi2Corrections.get_access<sycl_read>(h);
      accessor<float, 1, sycl_read_write, access::target::local> pool (WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE, h);
      h.parallel_for<class wallace>(nd_range<1>(rng_wallace_grid, rng_wallace_threads), [=] (nd_item<1> item) { 
        const unsigned lcg_a = 241;
        const unsigned lcg_c = 59;
        const unsigned lcg_m = 256;
        const unsigned mod_mask = lcg_m - 1;

        const unsigned lid = item.get_local_id(0);
        const unsigned gid = item.get_group(0);
        const unsigned offset = mul24(WALLACE_POOL_SIZE, gid);

        #pragma unroll
        for (unsigned i = 0; i < 8; i++)
          pool[lid + WALLACE_NUM_THREADS * i] = globalPool[offset + lid + WALLACE_NUM_THREADS * i];

        item.barrier(access::fence_space::local_space);

        unsigned t_seed = m_seed;

        // Loop generating generatedRandomNumberPools repeatedly
        for (unsigned loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++)
        {
          t_seed = (1664525U * t_seed + 1013904223U) & 0xFFFFFFFF;

          unsigned intermediate_address = mul24(loop, 8 * WALLACE_TOTAL_NUM_THREADS) + 
            mul24(8 * WALLACE_NUM_THREADS, gid) + lid;

          if (lid == 0)
            pool[WALLACE_CHI2_OFFSET] = chi2Corrections[mul24(gid, WALLACE_NUM_OUTPUTS_PER_RUN) + loop];
          item.barrier(access::fence_space::local_space);
          float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
          for (unsigned i = 0; i < 8; i++)
          {
            generatedRandomNumberPool[intermediate_address + i * WALLACE_NUM_THREADS] = 
              pool[mul24(i, WALLACE_NUM_THREADS) + lid] * chi2CorrAndScale;
          }

          float rin0_0, rin1_0, rin2_0, rin3_0, rin0_1, rin1_1, rin2_1, rin3_1;
          for (unsigned i = 0; i < WALLACE_NUM_POOL_PASSES; i++)
          {
            unsigned seed = (t_seed + lid) & mod_mask;
            item.barrier(access::fence_space::local_space);
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin0_0 = pool[((seed << 3))];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin1_0 = pool[((seed << 3) + 1)];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin2_0 = pool[((seed << 3) + 2)];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin3_0 = pool[((seed << 3) + 3)];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin0_1 = pool[((seed << 3) + 4)];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin1_1 = pool[((seed << 3) + 5)];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin2_1 = pool[((seed << 3) + 6)];
            seed = (mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin3_1 = pool[((seed << 3) + 7)];

            item.barrier(access::fence_space::local_space);

            Hadamard4x4a(rin0_0, rin1_0, rin2_0, rin3_0);
            pool[0 * WALLACE_NUM_THREADS + lid] = rin0_0;
            pool[1 * WALLACE_NUM_THREADS + lid] = rin1_0;
            pool[2 * WALLACE_NUM_THREADS + lid] = rin2_0;
            pool[3 * WALLACE_NUM_THREADS + lid] = rin3_0;

            Hadamard4x4b(rin0_1, rin1_1, rin2_1, rin3_1);
            pool[4 * WALLACE_NUM_THREADS + lid] = rin0_1;
            pool[5 * WALLACE_NUM_THREADS + lid] = rin1_1;
            pool[6 * WALLACE_NUM_THREADS + lid] = rin2_1;
            pool[7 * WALLACE_NUM_THREADS + lid] = rin3_1;

            item.barrier(access::fence_space::local_space);
          }
        }

        item.barrier(access::fence_space::local_space);

        #pragma unroll
        for (unsigned i = 0; i < 8; i++)
          globalPool[offset + lid + WALLACE_NUM_THREADS * i] = pool[lid + WALLACE_NUM_THREADS * i];
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  q.submit([&] (handler &h) {
    auto d_rng_acc = device_randomNumbers.get_access<sycl_read>(h);
    h.copy(d_rng_acc, randomNumbers);
  }).wait();

#ifdef DEBUG
  // random numbers are different for each i iteration 
  for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) 
    printf("%.3f\n", randomNumbers[n]);
#endif
  
  free(rngChi2Corrections);
  free(randomNumbers);
  free(hostPool);
  return 0;
}
