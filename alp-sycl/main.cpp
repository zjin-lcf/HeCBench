#include <stdio.h>
#include <math.h>
#include <chrono>
#include "common.h"
#include "reference.h"
#include "kernels.h"

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  double costheta = 0.3;
  int lmax = LMAX;
  double h_plm[NDLM];
  double r_plm[NDLM]; // reference

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  double *d_plm = malloc_device<double>(NDLM, q);

  range<1> gws (LMAX);
  range<1> lws (LMAX);

  // warmup up and check results
  bool ok = true;
  for (int l = 0; l <= lmax; l++) { 
    q.submit([&] (handler &cgh) {
      accessor<double, 1, sycl_read_write, access::target::local> sm(NDLM, cgh);
      cgh.parallel_for<class test>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        associatedLegendre(item,costheta,l,sm.get_pointer(),d_plm);
      });
    });

    // compute on host
    associatedLegendreFunctionNormalized<double>(costheta,l,r_plm);

    q.memcpy(h_plm, d_plm, NDLM * sizeof(double)).wait();

    for(int i = 0; i <= l; i++)
    {
      if(fabs(h_plm[i] - r_plm[i]) > 1e-6)
      {
        fprintf(stderr, "%d: %lf != %lf\n", i, h_plm[i], r_plm[i]);
        ok = false;
        break;
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  // performance measurement
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    for (int l = 0; l <= lmax; l++) {
      q.submit([&] (handler &cgh) {
        accessor<double, 1, sycl_read_write, access::target::local> sm(NDLM, cgh);
        cgh.parallel_for<class eval>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          associatedLegendre(item,costheta,l,sm.get_pointer(),d_plm);
        });
      });
    }
  }

  q.wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of associatedLegendre kernels %f (us)\n",
         time * 1e-3f / repeat);

  free(d_plm, q);

  return 0;
}
