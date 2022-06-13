#include <stdio.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "../alp-cuda/reference.h"
#include "kernels.h"

int main()
{
  double costheta = 0.3;
  int lmax = LMAX;
  #ifdef CHECK
  double h_plm[NDLM];
  #endif

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  auto start = std::chrono::steady_clock::now();

  double *d_plm = sycl::malloc_shared<double>(NDLM, q);

  range<1> gws (LMAX);
  range<1> lws (LMAX);
  for (int l = 0; l <= lmax; l++) { 
    q.submit([&] (handler &cgh) {
      accessor<double, 1, sycl_read_write, access::target::local> sm(NDLM, cgh);
      cgh.parallel_for<class al>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        associatedLegendre(item,costheta,l,sm.get_pointer(),d_plm);
      });
    });

    q.wait();
    
    #ifdef CHECK
    // compute on host
    associatedLegendreFunctionNormalized<double>(costheta,l,h_plm);

    for(int i = 0; i <= l; i++)
    {
      if(fabs(h_plm[i] - d_plm[i]) > 1e-6)
      {
        fprintf(stderr, "%d: %lf != %lf\n", i, h_plm[i], d_plm[i]);
        break;
      }
    }
    #endif
  }
  sycl::free(d_plm, q);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total execution time %f (s)\n", time * 1e-9f);

  return 0;
}
