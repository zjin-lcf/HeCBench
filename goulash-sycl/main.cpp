/*
  Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sycl/sycl.hpp>
#include "utils.h"

void gate(sycl::nd_item<1> &item, double* __restrict m_gate, 
          const long nCells, const double* __restrict Vm) 
{
  long i = item.get_global_id(0);
  if (i >= nCells) return;

  double sum1,sum2;
  const double x = Vm[i];
  const int Mhu_l = 10;
  const int Mhu_m = 5;
  const double Mhu_a[] = { 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18};

  sum1 = 0;
  for (int j = Mhu_m-1; j >= 0; j--)
    sum1 = Mhu_a[j] + x*sum1;
  sum2 = 0;
  int k = Mhu_m + Mhu_l - 1;
  for (int j = k; j >= Mhu_m; j--)
    sum2 = Mhu_a[j] + x * sum2;
  double mhu = sum1/sum2;

  const int Tau_m = 18;
  const double Tau_a[] = {1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,  1.0000000000000000e+00};

  sum1 = 0;
  for (int j = Tau_m-1; j >= 0; j--)
    sum1 = Tau_a[j] + x*sum1;
  double tauR = sum1;
  m_gate[i] += (mhu - m_gate[i])*(1-sycl::exp(-tauR));
}

int main(int argc, char* argv[]) 
{
  if (argc != 3)
  {
    printf ("Usage: %s <Iterations> <Kernel_GBs_used>\n\n", argv[0]);
    exit (1);
  }

  /* Get iteration count and target kernel memory used arguments */
  long iterations = atol(argv[1]);
  double kernel_mem_used = atof(argv[2]);

  /* Calculate nCells from target memory target */
  long nCells = (long) ((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) / (sizeof(double) * 2));
  printf("Number of cells: %ld\n", nCells);

  double* m_gate = (double*)calloc(nCells,sizeof(double));
  if (m_gate == NULL) printf ("failed calloc m_gate\n");

  // reference results
  double* m_gate_h = (double*)calloc(nCells,sizeof(double));
  if (m_gate_h == NULL) printf ("failed calloc m_gate_h\n");

  double* Vm = (double*)calloc(nCells,sizeof(double));
  if (Vm == NULL) printf ("failed calloc Vm\n");

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_m_gate = sycl::malloc_device<double>(nCells, q);
  q.memcpy(d_m_gate, m_gate, sizeof(double) * nCells);

  double *d_Vm = sycl::malloc_device<double>(nCells, q);
  q.memcpy(d_Vm, Vm, sizeof(double) * nCells);

  sycl::range<1> gws ((nCells + 255)/256*256);
  sycl::range<1> lws (256);

  double kernel_starttime, kernel_endtime, kernel_runtime;

  for (long itime=0; itime<=iterations; itime++) {
    /* Start timer after warm-up iteration 0 */
    if (itime == 1) {
      q.memcpy(m_gate, d_m_gate, sizeof(double) * nCells).wait();
      kernel_starttime = secs_elapsed();
    }

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class mgate>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        gate(item, d_m_gate, nCells, d_Vm);
      });
    });
  }

  q.wait();
  kernel_endtime = secs_elapsed();
  kernel_runtime = kernel_endtime-kernel_starttime;
  printf("total kernel time %lf(s) for %ld iterations\n", kernel_runtime, iterations-1);

  sycl::free(d_Vm, q);
  sycl::free(d_m_gate, q);

  // verify
  reference(m_gate_h, nCells, Vm);

  bool ok = true;
  for (long i = 0; i < nCells; i++) {
    if (fabs(m_gate[i] - m_gate_h[i]) > 1e-6) {
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  if (m_gate != NULL) free(m_gate);
  if (m_gate_h != NULL) free(m_gate_h);
  if (Vm != NULL) free(Vm);
  return 0;
}
