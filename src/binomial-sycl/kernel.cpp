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
#include <sycl/sycl.hpp>

#include "binomialOptions.h"
#include "realtype.h"


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
  float d = S * sycl::exp(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}
#else
inline double expiryCallValue(double S, double X, double vDt, int i)
{
  double d = S * sycl::exp(vDt * (2.0 * i - NUM_STEPS)) - X;
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
  __TOptionData h_OptionData[MAX_OPTIONS];

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

    h_OptionData[i].S      = (real)optionData[i].S;
    h_OptionData[i].X      = (real)optionData[i].X;
    h_OptionData[i].vDt    = (real)vDt;
    h_OptionData[i].puByDf = (real)puByDf;
    h_OptionData[i].pdByDf = (real)pdByDf;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  __TOptionData *d_optionData = sycl::malloc_device<__TOptionData>(MAX_OPTIONS, q);
  q.memcpy(d_optionData, h_OptionData, sizeof(__TOptionData) * MAX_OPTIONS);

  real *d_callValue = sycl::malloc_device<real>(MAX_OPTIONS, q);

  sycl::range<1> gws (optN * THREADBLOCK_SIZE);
  sycl::range<1> lws (THREADBLOCK_SIZE);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<real, 1> call_exchange(sycl::range<1>(THREADBLOCK_SIZE + 1), cgh);
      cgh.parallel_for<class kernel>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        const int     tid = item.get_local_id(0);
        const int     bid = item.get_group(0);
        const real      S = d_optionData[bid].S;
        const real      X = d_optionData[bid].X;
        const real    vDt = d_optionData[bid].vDt;
        const real puByDf = d_optionData[bid].puByDf;
        const real pdByDf = d_optionData[bid].pdByDf;

        real call[ELEMS_PER_THREAD + 1];
        #pragma unroll
        for(int i = 0; i < ELEMS_PER_THREAD; ++i)
          call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

        if (tid == 0)
          call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

        int final_it = sycl::max(0, tid * ELEMS_PER_THREAD - 1);

        #pragma unroll 16
        for(int i = NUM_STEPS; i > 0; --i)
        {
          call_exchange[tid] = call[0];
          item.barrier(sycl::access::fence_space::local_space);
          call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
          item.barrier(sycl::access::fence_space::local_space);

          if (i > final_it)
          {
            #pragma unroll
            for(int j = 0; j < ELEMS_PER_THREAD; ++j)
              call[j] = puByDf * call[j + 1] + pdByDf * call[j];
          }
        }

        if (tid == 0)
        {
          d_callValue[bid] = call[0];
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (us)\n", time * 1e-3f / numIterations);

  q.memcpy(callValue, d_callValue, optN *sizeof(real)).wait();
  sycl::free(d_optionData, q);
  sycl::free(d_callValue, q);
}
