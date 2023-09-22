/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#include "binomialOptions.h"
#include "realtype.h"

#define max(a, b) ((a) < (b) ? (b) : (a))

//Preprocessed input option data
typedef struct
{
  real S;
  real X;
  real vDt;
  real puByDf;
  real pdByDf;
} __TOptionData;


// Overloaded shortcut functions for different precision modes
#ifndef DOUBLE_PRECISION
inline float expiryCallValue(float S, float X, float vDt, int i)
{
  float d = S * expf(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}
#else
inline double expiryCallValue(double S, double X, double vDt, int i)
{
  double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
  return (d > 0.0) ? d : 0.0;
}
#endif


// GPU kernel
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

// Host-side interface to GPU binomialOptions
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN,
    int numIterations
    )
{
  __TOptionData d_OptionData[MAX_OPTIONS];

  for (int i = 0; i < optN; i++)
  {
    const real      T = optionData[i].T;
    const real      R = optionData[i].R;
    const real      V = optionData[i].V;

    const real     dt = T / (real)NUM_STEPS;
    const real    vDt = V * sqrt(dt);
    const real    rDt = R * dt;
    //Per-step interest and discount factors
    const real     If = exp(rDt);
    const real     Df = exp(-rDt);
    //Values and pseudoprobabilities of upward and downward moves
    const real      u = exp(vDt);
    const real      d = exp(-vDt);
    const real     pu = (If - d) / (u - d);
    const real     pd = (real)1.0 - pu;
    const real puByDf = pu * Df;
    const real pdByDf = pd * Df;

    d_OptionData[i].S      = (real)optionData[i].S;
    d_OptionData[i].X      = (real)optionData[i].X;
    d_OptionData[i].vDt    = (real)vDt;
    d_OptionData[i].puByDf = (real)puByDf;
    d_OptionData[i].pdByDf = (real)pdByDf;
  }

  #pragma omp target data map(to: d_OptionData[0:MAX_OPTIONS]) \
                          map(from: callValue[0:MAX_OPTIONS])
  {
    auto start = std::chrono::steady_clock::now();
  
    for (int i = 0; i < numIterations; i++) {
      #pragma omp target teams num_teams(optN) thread_limit(THREADBLOCK_SIZE)
      {
        real call_exchange[THREADBLOCK_SIZE + 1];
        #pragma omp parallel 
        {
          const int     tid = omp_get_thread_num();
          const int     bid = omp_get_team_num();
          const real      S = d_OptionData[bid].S;
          const real      X = d_OptionData[bid].X;
          const real    vDt = d_OptionData[bid].vDt;
          const real puByDf = d_OptionData[bid].puByDf;
          const real pdByDf = d_OptionData[bid].pdByDf;
  
          real call[ELEMS_PER_THREAD + 1];
          #pragma unroll
          for(int i = 0; i < ELEMS_PER_THREAD; ++i)
            call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);
  
          if (tid == 0)
            call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);
  
          int final_it = max(0, tid * ELEMS_PER_THREAD - 1);
  
          #pragma unroll 16
          for(int i = NUM_STEPS; i > 0; --i)
          {
            call_exchange[tid] = call[0];
            #pragma omp barrier
            call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
            #pragma omp barrier
  
            if (i > final_it)
            {
              #pragma unroll
              for(int j = 0; j < ELEMS_PER_THREAD; ++j)
                call[j] = puByDf * call[j + 1] + pdByDf * call[j];
            }
          }
  
          if (tid == 0)
          {
            callValue[bid] = call[0];
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (us)\n", time * 1e-3f / numIterations);
  }
}
