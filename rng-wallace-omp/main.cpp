#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "rand_helpers.h"
#include "constants.h"

#define mul24(a,b) (a)*(b)

#pragma omp declare target
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
#pragma omp end declare target

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // host buffers
  float *globalPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
  {
    float x = RandN();
    globalPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
    rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

  float *chi2Corrections = rngChi2Corrections;

  const unsigned m_seed = 1;

  #pragma omp target data map(to: globalPool[0: WALLACE_TOTAL_POOL_SIZE], \
                                  chi2Corrections[0:WALLACE_CHI2_COUNT]) \
                          map(alloc: randomNumbers[0: WALLACE_OUTPUT_SIZE])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams num_teams(WALLACE_NUM_BLOCKS) thread_limit(WALLACE_NUM_THREADS)
      {
        float pool[WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE];
        #pragma omp parallel
        {
          const unsigned lcg_a = 241;
          const unsigned lcg_c = 59;
          const unsigned lcg_m = 256;
          const unsigned mod_mask = lcg_m - 1;

          const unsigned lid = omp_get_thread_num();
          const unsigned gid = omp_get_team_num();
          const unsigned offset = mul24(WALLACE_POOL_SIZE, gid);

          #pragma unroll
          for (unsigned i = 0; i < 8; i++)
            pool[lid + WALLACE_NUM_THREADS * i] = globalPool[offset + lid + WALLACE_NUM_THREADS * i];

          #pragma omp barrier

          unsigned t_seed = m_seed;

          // Loop generating generatedRandomNumberPools repeatedly
          for (unsigned loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++)
          {
            t_seed = (1664525U * t_seed + 1013904223U) & 0xFFFFFFFF;

            unsigned intermediate_address = mul24(loop, 8 * WALLACE_TOTAL_NUM_THREADS) + 
              mul24(8 * WALLACE_NUM_THREADS, gid) + lid;

            if (lid == 0)
              pool[WALLACE_CHI2_OFFSET] = chi2Corrections[mul24(gid, WALLACE_NUM_OUTPUTS_PER_RUN) + loop];

            #pragma omp barrier
            float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
            for (unsigned i = 0; i < 8; i++)
            {
              randomNumbers[intermediate_address + i * WALLACE_NUM_THREADS] = 
                pool[mul24(i, WALLACE_NUM_THREADS) + lid] * chi2CorrAndScale;
            }

            float rin0_0, rin1_0, rin2_0, rin3_0, rin0_1, rin1_1, rin2_1, rin3_1;
            for (unsigned i = 0; i < WALLACE_NUM_POOL_PASSES; i++)
            {
              unsigned seed = (t_seed + lid) & mod_mask;
              #pragma omp barrier
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

              #pragma omp barrier

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

              #pragma omp barrier
            }
          }
          #pragma omp barrier

          #pragma unroll
          for (unsigned i = 0; i < 8; i++)
            globalPool[offset + lid + WALLACE_NUM_THREADS * i] = pool[lid + WALLACE_NUM_THREADS * i];
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

    #pragma omp target update from(randomNumbers[0:WALLACE_OUTPUT_SIZE])

    #ifdef DEBUG
      // random numbers are different for each i iteration 
      for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) 
        printf("%.3f\n", randomNumbers[n]);
    #endif
  }

  free(rngChi2Corrections);
  free(randomNumbers);
  free(globalPool);
  return 0;
}
