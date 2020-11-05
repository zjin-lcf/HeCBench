#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define BLOCK_SIZE 16

  
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
                 sycl::nd_item<3> item_ct1)
{

    const int b = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
    const int c = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                  item_ct1.get_local_id(1);

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

        dpct::atomic_fetch_add(emp4i,
                               denom * (f1tbc + f1ncb + f2tcb + f3nbc + f4ncb) *
                                       (f1tbc - f2tbc * 2 - f3tbc * 2 + f4tbc) -
                                   denom * (f1nbc + f1tcb + f2ncb + f3ncb) *
                                       (f1tbc * 2 - f2tbc - f3tbc + f4tbc * 2) +
                                   denom * 3 *
                                       (f1nbc * (f1nbc + f3ncb + f4tcb * 2) +
                                        f2nbc * f2tcb + f3nbc * f4tbc));

        dpct::atomic_fetch_add(emp4k,
                               denom * (f1nbc + f1tcb + f2ncb + f3tbc + f4tcb) *
                                       (f1nbc - f2nbc * 2 - f3nbc * 2 + f4nbc) -
                                   denom * (f1tbc + f1ncb + f2tcb + f3tcb) *
                                       (f1nbc * 2 - f2nbc - f3nbc + f4nbc * 2) +
                                   denom * 3 *
                                       (f1tbc * (f1tbc + f3tcb + f4ncb * 2) +
                                        f2tbc * f2ncb + f3tbc * f4nbc));

    const double t1v1b = t1v1[b];
    const double t1v2b = t1v2[b];

    const double dintx1c = dintx1[c];
    const double dintx2c = dintx2[c];
    const double dintc1c = dintc1[c];
    const double dintc2c = dintc2[c];

        dpct::atomic_fetch_add(
            emp5i,
            denom * t1v1b * dintx1c *
                    (f1tbc + f2nbc + f4ncb -
                     (f3tbc + f4nbc + f2ncb + f1nbc + f2tbc + f3ncb) * 2 +
                     (f3nbc + f4tbc + f1ncb) * 4) +
                denom * t1v1b * dintc1c *
                    (f1nbc + f4nbc + f1tcb - (f2nbc + f3nbc + f2tcb) * 2));
        dpct::atomic_fetch_add(
            emp5k,
            denom * t1v2b * dintx2c *
                    (f1nbc + f2tbc + f4tcb -
                     (f3nbc + f4tbc + f2tcb + f1tbc + f2nbc + f3tcb) * 2 +
                     (f3tbc + f4nbc + f1tcb) * 4) +
                denom * t1v2b * dintc2c *
                    (f1tbc + f4tbc + f1ncb - (f2tbc + f3tbc + f2ncb) * 2));
  }

}

void ccsd_tengy_gpu(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                    const double * __restrict__  f2n,    const double * __restrict__ f2t,
                    const double * __restrict__  f3n,    const double * __restrict__ f3t,
                    const double * __restrict__  f4n,    const double * __restrict__ f4t,
                    const double * __restrict__  dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                    const double * __restrict__  dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                    const double * __restrict__  eorb,   const double eaijk,
                    double * __restrict__ emp4i_, double * __restrict__ emp5i_,
                    double * __restrict__ emp4k_, double * __restrict__ emp5k_,
                    const int ncor, const int nocc, const int nvir)
{
  double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

  double *d_f1n, *d_f2n, *d_f3n, *d_f4n;
  double *d_f1t, *d_f2t, *d_f3t, *d_f4t;
  double *d_dintc1, *d_dintc2, *d_dintx1, *d_dintx2;
  double *d_t1v1, *d_t1v2, *d_eorb;
  double *d_emp5i, *d_emp4i, *d_emp5k, *d_emp4k;
    dpct::dpct_malloc((void **)&d_f1n, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f2n, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f3n, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f4n, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f1t, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f2t, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f3t, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_f4t, nvir * nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_dintc1, nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_dintc2, nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_dintx1, nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_dintx2, nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_t1v1, nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_t1v2, nvir * sizeof(double));
    dpct::dpct_malloc((void **)&d_eorb, (ncor + nocc + nvir) * sizeof(double));
    dpct::dpct_malloc((void **)&d_emp5i, sizeof(double));
    dpct::dpct_malloc((void **)&d_emp4i, sizeof(double));
    dpct::dpct_malloc((void **)&d_emp5k, sizeof(double));
    dpct::dpct_malloc((void **)&d_emp4k, sizeof(double));

    dpct::dpct_memcpy(d_f1n, f1n, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f2n, f2n, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f3n, f3n, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f4n, f4n, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f1t, f1t, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f2t, f2t, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f3t, f3t, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_f4t, f4t, nvir * nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_dintc1, dintc1, nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_dintc2, dintc2, nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_dintx1, dintx1, nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_dintx2, dintx2, nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_t1v1, t1v1, nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_t1v2, t1v2, nvir * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_eorb, eorb, (ncor + nocc + nvir) * sizeof(double),
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_emp5i, &emp5i, sizeof(double), dpct::host_to_device);
    dpct::dpct_memcpy(d_emp4i, &emp4i, sizeof(double), dpct::host_to_device);
    dpct::dpct_memcpy(d_emp5k, &emp5k, sizeof(double), dpct::host_to_device);
    dpct::dpct_memcpy(d_emp5k, &emp4k, sizeof(double), dpct::host_to_device);

    {
        dpct::buffer_t d_f1n_buf_ct0 = dpct::get_buffer(d_f1n);
        dpct::buffer_t d_f1t_buf_ct1 = dpct::get_buffer(d_f1t);
        dpct::buffer_t d_f2n_buf_ct2 = dpct::get_buffer(d_f2n);
        dpct::buffer_t d_f2t_buf_ct3 = dpct::get_buffer(d_f2t);
        dpct::buffer_t d_f3n_buf_ct4 = dpct::get_buffer(d_f3n);
        dpct::buffer_t d_f3t_buf_ct5 = dpct::get_buffer(d_f3t);
        dpct::buffer_t d_f4n_buf_ct6 = dpct::get_buffer(d_f4n);
        dpct::buffer_t d_f4t_buf_ct7 = dpct::get_buffer(d_f4t);
        dpct::buffer_t d_dintc1_buf_ct8 = dpct::get_buffer(d_dintc1);
        dpct::buffer_t d_dintx1_buf_ct9 = dpct::get_buffer(d_dintx1);
        dpct::buffer_t d_t1v1_buf_ct10 = dpct::get_buffer(d_t1v1);
        dpct::buffer_t d_dintc2_buf_ct11 = dpct::get_buffer(d_dintc2);
        dpct::buffer_t d_dintx2_buf_ct12 = dpct::get_buffer(d_dintx2);
        dpct::buffer_t d_t1v2_buf_ct13 = dpct::get_buffer(d_t1v2);
        dpct::buffer_t d_eorb_buf_ct14 = dpct::get_buffer(d_eorb);
        dpct::buffer_t d_emp4i_buf_ct16 = dpct::get_buffer(d_emp4i);
        dpct::buffer_t d_emp5i_buf_ct17 = dpct::get_buffer(d_emp5i);
        dpct::buffer_t d_emp4k_buf_ct18 = dpct::get_buffer(d_emp4k);
        dpct::buffer_t d_emp5k_buf_ct19 = dpct::get_buffer(d_emp5k);
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_f1n_acc_ct0 =
                d_f1n_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f1t_acc_ct1 =
                d_f1t_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f2n_acc_ct2 =
                d_f2n_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f2t_acc_ct3 =
                d_f2t_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f3n_acc_ct4 =
                d_f3n_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f3t_acc_ct5 =
                d_f3t_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f4n_acc_ct6 =
                d_f4n_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
            auto d_f4t_acc_ct7 =
                d_f4t_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
            auto d_dintc1_acc_ct8 =
                d_dintc1_buf_ct8.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_dintx1_acc_ct9 =
                d_dintx1_buf_ct9.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_t1v1_acc_ct10 =
                d_t1v1_buf_ct10.get_access<sycl::access::mode::read_write>(cgh);
            auto d_dintc2_acc_ct11 =
                d_dintc2_buf_ct11.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_dintx2_acc_ct12 =
                d_dintx2_buf_ct12.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_t1v2_acc_ct13 =
                d_t1v2_buf_ct13.get_access<sycl::access::mode::read_write>(cgh);
            auto d_eorb_acc_ct14 =
                d_eorb_buf_ct14.get_access<sycl::access::mode::read_write>(cgh);
            auto d_emp4i_acc_ct16 =
                d_emp4i_buf_ct16.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_emp5i_acc_ct17 =
                d_emp5i_buf_ct17.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_emp4k_acc_ct18 =
                d_emp4k_buf_ct18.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_emp5k_acc_ct19 =
                d_emp5k_buf_ct19.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, (nvir + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                   (nvir + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                        sycl::range<3>(1, BLOCK_SIZE, BLOCK_SIZE),
                    sycl::range<3>(1, BLOCK_SIZE, BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1) {
                    ccsd_kernel((const double *)(&d_f1n_acc_ct0[0]),
                                (const double *)(&d_f1t_acc_ct1[0]),
                                (const double *)(&d_f2n_acc_ct2[0]),
                                (const double *)(&d_f2t_acc_ct3[0]),
                                (const double *)(&d_f3n_acc_ct4[0]),
                                (const double *)(&d_f3t_acc_ct5[0]),
                                (const double *)(&d_f4n_acc_ct6[0]),
                                (const double *)(&d_f4t_acc_ct7[0]),
                                (const double *)(&d_dintc1_acc_ct8[0]),
                                (const double *)(&d_dintx1_acc_ct9[0]),
                                (const double *)(&d_t1v1_acc_ct10[0]),
                                (const double *)(&d_dintc2_acc_ct11[0]),
                                (const double *)(&d_dintx2_acc_ct12[0]),
                                (const double *)(&d_t1v2_acc_ct13[0]),
                                (const double *)(&d_eorb_acc_ct14[0]), eaijk,
                                (double *)(&d_emp4i_acc_ct16[0]),
                                (double *)(&d_emp5i_acc_ct17[0]),
                                (double *)(&d_emp4k_acc_ct18[0]),
                                (double *)(&d_emp5k_acc_ct19[0]), ncor, nocc,
                                nvir, item_ct1);
                });
        });
    }

#ifdef DEBUG
  // make the host block until the device is finished
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
#endif

    dpct::dpct_memcpy(&emp5i, d_emp5i, sizeof(double), dpct::device_to_host);
    dpct::dpct_memcpy(&emp4i, d_emp4i, sizeof(double), dpct::device_to_host);
    dpct::dpct_memcpy(&emp5k, d_emp5k, sizeof(double), dpct::device_to_host);
    dpct::dpct_memcpy(&emp4k, d_emp5k, sizeof(double), dpct::device_to_host);

    dpct::dpct_free(d_f1n);
    dpct::dpct_free(d_f2n);
    dpct::dpct_free(d_f3n);
    dpct::dpct_free(d_f4n);
    dpct::dpct_free(d_f1t);
    dpct::dpct_free(d_f2t);
    dpct::dpct_free(d_f3t);
    dpct::dpct_free(d_f4t);
    dpct::dpct_free(d_dintc1);
    dpct::dpct_free(d_dintc2);
    dpct::dpct_free(d_dintx1);
    dpct::dpct_free(d_dintx2);
    dpct::dpct_free(d_t1v1);
    dpct::dpct_free(d_t1v2);
    dpct::dpct_free(d_eorb);
    dpct::dpct_free(d_emp5i);
    dpct::dpct_free(d_emp4i);
    dpct::dpct_free(d_emp5k);
    dpct::dpct_free(d_emp4k);

  *emp4i_ = emp4i;
  *emp4k_ = emp4k;
  *emp5i_ = emp5i;
  *emp5k_ = emp5k;
}

