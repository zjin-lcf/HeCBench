#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
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
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_Pool= sycl::malloc_device<float>(WALLACE_TOTAL_POOL_SIZE, q);
  float *d_rngChi2Corrections= sycl::malloc_device<float>(WALLACE_CHI2_COUNT, q);
  float *d_randomNumbers= sycl::malloc_device<float>(WALLACE_OUTPUT_SIZE, q);

  q.memcpy(d_Pool, hostPool, sizeof(float) * WALLACE_TOTAL_POOL_SIZE);
  q.memcpy(d_rngChi2Corrections, rngChi2Corrections, sizeof(float) * WALLACE_CHI2_COUNT);

  sycl::range<1> gws (WALLACE_NUM_BLOCKS * WALLACE_NUM_THREADS);
  sycl::range<1> lws (WALLACE_NUM_THREADS);
  const unsigned m_seed = 1;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor<float, 1> pool (
        sycl::range<1>(WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE), h);
      h.parallel_for<class wallace>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        const unsigned lcg_a = 241;
        const unsigned lcg_c = 59;
        const unsigned lcg_m = 256;
        const unsigned mod_mask = lcg_m - 1;

        const unsigned lid = item.get_local_id(0);
        const unsigned gid = item.get_group(0);
        const unsigned offset = sycl::mul24(WALLACE_POOL_SIZE, gid);

        #pragma unroll
        for (unsigned i = 0; i < 8; i++)
          pool[lid + WALLACE_NUM_THREADS * i] = d_Pool[offset + lid + WALLACE_NUM_THREADS * i];

        item.barrier(sycl::access::fence_space::local_space);

        unsigned t_seed = m_seed;

        // Loop generating d_randomNumberss repeatedly
        for (unsigned loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++)
        {
          t_seed = (1664525U * t_seed + 1013904223U) & 0xFFFFFFFF;

          unsigned intermediate_address = sycl::mul24(loop, 8 * WALLACE_TOTAL_NUM_THREADS) +
            sycl::mul24(8 * WALLACE_NUM_THREADS, gid) + lid;

          if (lid == 0)
            pool[WALLACE_CHI2_OFFSET] = d_rngChi2Corrections[sycl::mul24(gid, WALLACE_NUM_OUTPUTS_PER_RUN) + loop];
          item.barrier(sycl::access::fence_space::local_space);
          float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
          for (unsigned i = 0; i < 8; i++)
          {
            d_randomNumbers[intermediate_address + i * WALLACE_NUM_THREADS] =
              pool[sycl::mul24(i, WALLACE_NUM_THREADS) + lid] * chi2CorrAndScale;
          }

          float rin0_0, rin1_0, rin2_0, rin3_0, rin0_1, rin1_1, rin2_1, rin3_1;
          for (unsigned i = 0; i < WALLACE_NUM_POOL_PASSES; i++)
          {
            unsigned seed = (t_seed + lid) & mod_mask;
            item.barrier(sycl::access::fence_space::local_space);
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin0_0 = pool[((seed << 3))];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin1_0 = pool[((seed << 3) + 1)];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin2_0 = pool[((seed << 3) + 2)];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin3_0 = pool[((seed << 3) + 3)];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin0_1 = pool[((seed << 3) + 4)];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin1_1 = pool[((seed << 3) + 5)];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin2_1 = pool[((seed << 3) + 6)];
            seed = (sycl::mul24(seed, lcg_a) + lcg_c) & mod_mask;
            rin3_1 = pool[((seed << 3) + 7)];

            item.barrier(sycl::access::fence_space::local_space);

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

            item.barrier(sycl::access::fence_space::local_space);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        #pragma unroll
        for (unsigned i = 0; i < 8; i++)
          d_Pool[offset + lid + WALLACE_NUM_THREADS * i] = pool[lid + WALLACE_NUM_THREADS * i];
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(randomNumbers, d_randomNumbers, sizeof(float) * WALLACE_OUTPUT_SIZE).wait();

#ifdef DEBUG
  // random numbers are different for each i iteration
  for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++)
    printf("%.3f\n", randomNumbers[n]);
#endif

  free(rngChi2Corrections);
  free(randomNumbers);
  free(hostPool);
  sycl::free(d_rngChi2Corrections, q);
  sycl::free(d_randomNumbers, q);
  sycl::free(d_Pool, q);
  return 0;
}
