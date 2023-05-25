#include <math.h>
#include <sycl/sycl.hpp>

void lebesgue_kernel (
  sycl::nd_item<1> &item,
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
    t += sycl::fabs ( linterp[i*nfun+j] );

  // atomicMax(lmax, t);
  auto lmax_ref = sycl::atomic_ref<double,
                  sycl::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space>(lmax[0]);
  lmax_ref.fetch_max(t);
}

double lebesgue_function ( sycl::queue &q, int n, double x[], int nfun, double xfun[] )
{
  double lmax = 0.0;

  double *d_max = sycl::malloc_device<double>( 1, q );
  q.memcpy(d_max, &lmax, sizeof ( double ));

  double *d_interp = sycl::malloc_device<double>( n * nfun, q );

  double *d_x = sycl::malloc_device<double>( n, q );
  q.memcpy(d_x, x, n * sizeof ( double ));

  double *d_xfun = sycl::malloc_device<double>( nfun, q );
  q.memcpy(d_xfun, xfun, nfun * sizeof ( double ));

  sycl::range<1> gws ((nfun + 255)/256*256);
  sycl::range<1> lws (256);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class k>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      lebesgue_kernel (
        item, d_max, d_interp, d_xfun, d_x, n, nfun);
    });
  });

  q.memcpy(&lmax, d_max, sizeof ( double )).wait();

  sycl::free(d_max, q);
  sycl::free(d_interp, q);
  sycl::free(d_xfun, q);
  sycl::free(d_x, q);

  return lmax;
}
