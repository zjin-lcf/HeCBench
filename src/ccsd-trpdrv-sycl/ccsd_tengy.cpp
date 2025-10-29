#include <chrono>
#include <sycl/sycl.hpp>

#define BLOCK_SIZE 16

inline void atomicAdd(double *val, const double operand)
{
  sycl::atomic_ref<double,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> ref (*val);
  ref.fetch_add(operand);
}

void ccsd_kernel(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                 const double * __restrict__ f2n,    const double * __restrict__ f2t,
                 const double * __restrict__ f3n,    const double * __restrict__ f3t,
                 const double * __restrict__ f4n,    const double * __restrict__ f4t,
                 const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                 const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                 const double * __restrict__ eorb,   const double eaijk,
                 double * __restrict__ emp4i, double * __restrict__ emp5i,
                 double * __restrict__ emp4k, double * __restrict__ emp5k,
                 const int ncor, const int nocc, const int nvir,
                 const sycl::nd_item<2> &item)
{
  const int b = item.get_global_id(1);
  const int c = item.get_global_id(0);

  if (b < nvir && c < nvir) {

    const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

    // nvir < 10000 so this should never overflow
    const int bc = b+c*nvir;
    const int cb = c+b*nvir;

    const double f1nbc = f1n[bc];
    const double f1tbc = f1t[bc];
    const double f1ncb = f1n[cb];
    const double f1tcb = f1t[cb];

    const double f2nbc = f2n[bc];
    const double f2tbc = f2t[bc];
    const double f2ncb = f2n[cb];
    const double f2tcb = f2t[cb];

    const double f3nbc = f3n[bc];
    const double f3tbc = f3t[bc];
    const double f3ncb = f3n[cb];
    const double f3tcb = f3t[cb];

    const double f4nbc = f4n[bc];
    const double f4tbc = f4t[bc];
    const double f4ncb = f4n[cb];
    const double f4tcb = f4t[cb];

    atomicAdd(emp4i, denom * (f1tbc + f1ncb + f2tcb + f3nbc + f4ncb) *
                       (f1tbc - f2tbc * 2 - f3tbc * 2 + f4tbc) -
                   denom * (f1nbc + f1tcb + f2ncb + f3ncb) *
                       (f1tbc * 2 - f2tbc - f3tbc + f4tbc * 2) +
                   denom * 3 *
                       (f1nbc * (f1nbc + f3ncb + f4tcb * 2) + f2nbc * f2tcb +
                        f3nbc * f4tbc));

    atomicAdd(emp4k, denom * (f1nbc + f1tcb + f2ncb + f3tbc + f4tcb) *
                       (f1nbc - f2nbc * 2 - f3nbc * 2 + f4nbc) -
                   denom * (f1tbc + f1ncb + f2tcb + f3tcb) *
                       (f1nbc * 2 - f2nbc - f3nbc + f4nbc * 2) +
                   denom * 3 *
                       (f1tbc * (f1tbc + f3tcb + f4ncb * 2) + f2tbc * f2ncb +
                        f3tbc * f4nbc));

    const double t1v1b = t1v1[b];
    const double t1v2b = t1v2[b];

    const double dintx1c = dintx1[c];
    const double dintx2c = dintx2[c];
    const double dintc1c = dintc1[c];
    const double dintc2c = dintc2[c];

    atomicAdd(emp5i, denom * t1v1b * dintx1c *
                       (f1tbc + f2nbc + f4ncb -
                        (f3tbc + f4nbc + f2ncb + f1nbc + f2tbc + f3ncb) * 2 +
                        (f3nbc + f4tbc + f1ncb) * 4) +
                   denom * t1v1b * dintc1c *
                       (f1nbc + f4nbc + f1tcb - (f2nbc + f3nbc + f2tcb) * 2));
    atomicAdd(emp5k, denom * t1v2b * dintx2c *
                       (f1nbc + f2tbc + f4tcb -
                        (f3nbc + f4tbc + f2tcb + f1tbc + f2nbc + f3tcb) * 2 +
                        (f3tbc + f4nbc + f1tcb) * 4) +
                   denom * t1v2b * dintc2c *
                       (f1tbc + f4tbc + f1ncb - (f2tbc + f3tbc + f2ncb) * 2));
  }
}

long ccsd_tengy_gpu(sycl::queue &q,
    const double * __restrict f1n,    const double * __restrict f1t,
    const double * __restrict f2n,    const double * __restrict f2t,
    const double * __restrict f3n,    const double * __restrict f3t,
    const double * __restrict f4n,    const double * __restrict f4t,
    const double * __restrict dintc1, const double * __restrict dintx1, const double * __restrict t1v1,
    const double * __restrict dintc2, const double * __restrict dintx2, const double * __restrict t1v2,
    const double * __restrict eorb,   const double eaijk,
    double * __restrict emp4i_, double * __restrict emp5i_,
    double * __restrict emp4k_, double * __restrict emp5k_,
    const int ncor, const int nocc, const int nvir)
{
  double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

  double *d_f1n, *d_f2n, *d_f3n, *d_f4n;
  double *d_f1t, *d_f2t, *d_f3t, *d_f4t;
  double *d_dintc1, *d_dintc2, *d_dintx1, *d_dintx2;
  double *d_t1v1, *d_t1v2, *d_eorb;
  double *d_emp5i, *d_emp4i, *d_emp5k, *d_emp4k;
  d_f1n = sycl::malloc_device<double>(nvir * nvir, q);
  d_f2n = sycl::malloc_device<double>(nvir * nvir, q);
  d_f3n = sycl::malloc_device<double>(nvir * nvir, q);
  d_f4n = sycl::malloc_device<double>(nvir * nvir, q);
  d_f1t = sycl::malloc_device<double>(nvir * nvir, q);
  d_f2t = sycl::malloc_device<double>(nvir * nvir, q);
  d_f3t = sycl::malloc_device<double>(nvir * nvir, q);
  d_f4t = sycl::malloc_device<double>(nvir * nvir, q);
  d_dintc1 = sycl::malloc_device<double>(nvir, q);
  d_dintc2 = sycl::malloc_device<double>(nvir, q);
  d_dintx1 = sycl::malloc_device<double>(nvir, q);
  d_dintx2 = sycl::malloc_device<double>(nvir, q);
  d_t1v1 = sycl::malloc_device<double>(nvir, q);
  d_t1v2 = sycl::malloc_device<double>(nvir, q);
  d_eorb = sycl::malloc_device<double>((ncor + nocc + nvir), q);
  d_emp5i = sycl::malloc_device<double>(1, q);
  d_emp4i = sycl::malloc_device<double>(1, q);
  d_emp5k = sycl::malloc_device<double>(1, q);
  d_emp4k = sycl::malloc_device<double>(1, q);

  q.memcpy(d_f1n, f1n, nvir * nvir * sizeof(double));
  q.memcpy(d_f2n, f2n, nvir * nvir * sizeof(double));
  q.memcpy(d_f3n, f3n, nvir * nvir * sizeof(double));
  q.memcpy(d_f4n, f4n, nvir * nvir * sizeof(double));
  q.memcpy(d_f1t, f1t, nvir * nvir * sizeof(double));
  q.memcpy(d_f2t, f2t, nvir * nvir * sizeof(double));
  q.memcpy(d_f3t, f3t, nvir * nvir * sizeof(double));
  q.memcpy(d_f4t, f4t, nvir * nvir * sizeof(double));
  q.memcpy(d_dintc1, dintc1, nvir * sizeof(double));
  q.memcpy(d_dintc2, dintc2, nvir * sizeof(double));
  q.memcpy(d_dintx1, dintx1, nvir * sizeof(double));
  q.memcpy(d_dintx2, dintx2, nvir * sizeof(double));
  q.memcpy(d_t1v1, t1v1, nvir * sizeof(double));
  q.memcpy(d_t1v2, t1v2, nvir * sizeof(double));
  q.memcpy(d_eorb, eorb, (ncor + nocc + nvir) * sizeof(double));
  q.memcpy(d_emp5i, &emp5i, sizeof(double));
  q.memcpy(d_emp4i, &emp4i, sizeof(double));
  q.memcpy(d_emp5k, &emp5k, sizeof(double));
  q.memcpy(d_emp4k, &emp4k, sizeof(double));

  q.wait();
  auto t0 = std::chrono::steady_clock::now();

  sycl::range<2> gws((nvir+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE,
                     (nvir+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  sycl::range<2> lws(BLOCK_SIZE, BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class tengy>(
      sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        ccsd_kernel(d_f1n, d_f1t, d_f2n, d_f2t, d_f3n, d_f3t, d_f4n, d_f4t,
                    d_dintc1, d_dintx1, d_t1v1, d_dintc2, d_dintx2, d_t1v2,
                    d_eorb, eaijk, d_emp4i, d_emp5i, d_emp4k, d_emp5k, ncor,
                    nocc, nvir, item);
    });
  }).wait();

  auto t1 = std::chrono::steady_clock::now();
  long time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  q.memcpy(&emp5i, d_emp5i, sizeof(double));
  q.memcpy(&emp4i, d_emp4i, sizeof(double));
  q.memcpy(&emp5k, d_emp5k, sizeof(double));
  q.memcpy(&emp4k, d_emp4k, sizeof(double));
  q.wait();

  sycl::free(d_f1n, q);
  sycl::free(d_f2n, q);
  sycl::free(d_f3n, q);
  sycl::free(d_f4n, q);
  sycl::free(d_f1t, q);
  sycl::free(d_f2t, q);
  sycl::free(d_f3t, q);
  sycl::free(d_f4t, q);
  sycl::free(d_dintc1, q);
  sycl::free(d_dintc2, q);
  sycl::free(d_dintx1, q);
  sycl::free(d_dintx2, q);
  sycl::free(d_t1v1, q);
  sycl::free(d_t1v2, q);
  sycl::free(d_eorb, q);
  sycl::free(d_emp5i, q);
  sycl::free(d_emp4i, q);
  sycl::free(d_emp5k, q);
  sycl::free(d_emp4k, q);

  *emp4i_ = emp4i;
  *emp4k_ = emp4k;
  *emp5i_ = emp5i;
  *emp5k_ = emp5k;
  return time;
}
