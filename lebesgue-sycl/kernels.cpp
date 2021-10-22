#include <math.h>
#include "common.h"

void lebesgue_kernel (
  nd_item<1> &item,
  double *__restrict__ lmax,
  double *__restrict__ linterp,
  const double *__restrict__ xfun, 
  const double *__restrict__ x,
  const int n, const int nfun) 
{
  int j = item.get_global_id(0);
  if (j >= nfun) return;
  for (int i = 0; i < n; i++ )
    linterp[i*nfun+j] = 1.0;

  for (int i1 = 0; i1 < n; i1++ )
    for (int i2 = 0; i2 < n; i2++ )
      if ( i1 != i2 )
        linterp[i1*nfun+j] = linterp[i1*nfun+j] * ( xfun[j] - x[i2] ) / ( x[i1] - x[i2] );

  double t = 0.0;
  for (int i = 0; i < n; i++ )
    t += fabs ( linterp[i*nfun+j] );

  // atomicMax(lmax, t);
  auto lmax_ref = ext::oneapi::atomic_ref<double, 
                  ext::oneapi::memory_order::relaxed,
                  ext::oneapi::memory_scope::device,
                  access::address_space::global_space> (lmax[0]);
  lmax_ref.fetch_max(t);
}

double lebesgue_function ( queue &q, int n, double x[], int nfun, double xfun[] )
{
  double lmax = 0.0;

  buffer<double, 1> d_max ( &lmax, 1 );

  buffer<double, 1> d_interp ( n * nfun );

  buffer<double, 1> d_x ( x, n );

  buffer<double, 1> d_xfun ( xfun, nfun );

  range<1> gws ((nfun + 255)/256*256);
  range<1> lws (256);

  q.submit([&] (handler &cgh) {
    auto lmax = d_max.get_access<sycl_read_write>(cgh);
    auto interp = d_interp.get_access<sycl_read_write>(cgh);
    auto xfun = d_xfun.get_access<sycl_read>(cgh);
    auto x = d_x.get_access<sycl_read>(cgh);
    cgh.parallel_for<class k>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      lebesgue_kernel (item,
              lmax.get_pointer(), 
              interp.get_pointer(),
              xfun.get_pointer(),
              x.get_pointer(),
              n, nfun);
    });
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_max.get_access<sycl_read>(cgh);
    cgh.copy(acc, &lmax);
  }).wait();

  return lmax;
}
