
#ifndef _DIVERGENCE_HPP_
#define _DIVERGENCE_HPP_

#if defined(__INTEL_COMPILER)
#define NOVECDEP _Pragma("ivdep")
#define ALWAYSVECTORIZE _Pragma("vector always")
#define ALIGN(vardec) __declspec(align) vardec
#define ALIGNTO(vardec, boundary) \
  __declspec(align(boundary)) vardec
#elif defined(__GNUG__)
#if(__GNUG__ == 4 && __GNUC_MINOR__ >= 9) || __GNUG__ > 4
#define NOVECDEP _Pragma("GCC ivdep")
#define ALWAYSVECTORIZE _Pragma("GCC vector always")
#else
#pragma message( \
    "G++ <4.9 Does not support vectorization pragmas")
#define NOVECDEP
#define ALWAYSVECTORIZE
#endif

#define ALIGN(vardec) __attribute__((aligned)) vardec
#define ALIGNTO(vardec, boundary) \
  __attribute__((aligned(boundary))) vardec
#endif

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#define BLOCK_SIZE 16

constexpr const int dim = 2;

template <int np, typename real>
using real_vector = real[np][np][dim];

template <int np, typename real>
using real_scalar = real[np][np];

template <int np, typename real>
struct element {
  real_scalar<np, real> metdet;
  real Dinv[np][np][2][2];
  real_scalar<np, real> rmetdet;
};

template <int np, typename real>
struct derivative {
  real_scalar<np, real> Dvv;
};

using real = double;

void 
div_kernel (real* gv, 
            real* Dvv, 
            real* div, 
            real* vvtemp, 
            real* rmetdet, 
            int np ,
            sycl::nd_item<3> item_ct1)
{

  constexpr const real rrearth = 1.5683814303638645E-7;
  int l = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);
  int j = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) +
          item_ct1.get_local_id(1);
  if (l < np && j < np) {

    real dudx00 = 0.0;
    real dvdy00 = 0.0;
    for(int i = 0; i < np; i++) {
      dudx00 += Dvv[l*np+i] * gv[j*np*dim+i*dim];
      dvdy00 += Dvv[l*np+i] * gv[i*np*dim+j*dim+1];
    }
    div[j*np+l] = dudx00;
    vvtemp[l*np+j] = dvdy00;
  }
  item_ct1.barrier();

  if (l < np && j < np) 
    div[l*np+j] = (div[l*np+j] + vvtemp[l*np+j]) * 
                  (rmetdet[l*np+j] * rrearth);
}

template <int np, typename real>
__attribute__((noinline)) void divergence_sphere_gpu(
    const  real_vector<np, real> v,
    const derivative<np, real> & deriv,
    const element<np, real> & elem,
     real_scalar<np, real> div) {

  using rv = real_vector<np, real>;
  ALIGNTO(rv gv, 16);

  /* Convert to contra variant form and multiply by g */
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int k = 0; k < dim; k++) {
        gv[j][i][k] = elem.metdet[j][i] *
                      (elem.Dinv[j][i][k][0] * v[j][i][0] +
                       elem.Dinv[j][i][k][1] * v[j][i][1]);
      }
    }
  }

  real* d_gv;
  real* d_Dvv;
  real* d_div;
  real* d_vvtemp;
  real* d_rmetdet;

  dpct::dpct_malloc((void **)&d_gv, sizeof(real) * np * np * dim);
  dpct::dpct_malloc((void **)&d_Dvv, sizeof(real) * np * np);
  dpct::dpct_malloc((void **)&d_div, sizeof(real) * np * np);
  dpct::dpct_malloc((void **)&d_vvtemp, sizeof(real) * np * np);
  dpct::dpct_malloc((void **)&d_rmetdet, sizeof(real) * np * np);

  dpct::dpct_memcpy(d_Dvv, deriv.Dvv, sizeof(real) * np * np,
                    dpct::host_to_device);
  dpct::dpct_memcpy(d_gv, gv, sizeof(real) * np * np * dim,
                    dpct::host_to_device);
  dpct::dpct_memcpy(d_rmetdet, elem.rmetdet, sizeof(real) * np * np,
                    dpct::host_to_device);

  {
    std::pair<dpct::buffer_t, size_t> d_gv_buf_ct0 =
        dpct::get_buffer_and_offset(d_gv);
    size_t d_gv_offset_ct0 = d_gv_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_Dvv_buf_ct1 =
        dpct::get_buffer_and_offset(d_Dvv);
    size_t d_Dvv_offset_ct1 = d_Dvv_buf_ct1.second;
    dpct::buffer_t d_div_buf_ct2 = dpct::get_buffer(d_div);
    dpct::buffer_t d_vvtemp_buf_ct3 = dpct::get_buffer(d_vvtemp);
    std::pair<dpct::buffer_t, size_t> d_rmetdet_buf_ct4 =
        dpct::get_buffer_and_offset(d_rmetdet);
    size_t d_rmetdet_offset_ct4 = d_rmetdet_buf_ct4.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto d_gv_acc_ct0 =
          d_gv_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Dvv_acc_ct1 =
          d_Dvv_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_div_acc_ct2 =
          d_div_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
      auto d_vvtemp_acc_ct3 =
          d_vvtemp_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto d_rmetdet_acc_ct4 =
          d_rmetdet_buf_ct4.first.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1,
                                           (np + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                           (np + BLOCK_SIZE - 1) / BLOCK_SIZE) *
                                sycl::range<3>(1, BLOCK_SIZE, BLOCK_SIZE),
                            sycl::range<3>(1, BLOCK_SIZE, BLOCK_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            real *d_gv_ct0 = (real *)(&d_gv_acc_ct0[0] + d_gv_offset_ct0);
            real *d_Dvv_ct1 = (real *)(&d_Dvv_acc_ct1[0] + d_Dvv_offset_ct1);
            real *d_rmetdet_ct4 =
                (real *)(&d_rmetdet_acc_ct4[0] + d_rmetdet_offset_ct4);
            div_kernel(d_gv_ct0, d_Dvv_ct1, (real *)(&d_div_acc_ct2[0]),
                       (real *)(&d_vvtemp_acc_ct3[0]), d_rmetdet_ct4, np,
                       item_ct1);
          });
    });
  }

  dpct::dpct_memcpy(div, d_div, sizeof(real) * np * np, dpct::device_to_host);

  dpct::dpct_free(d_gv);
  dpct::dpct_free(d_Dvv);
  dpct::dpct_free(d_div);
  dpct::dpct_free(d_vvtemp);
  dpct::dpct_free(d_rmetdet);
}

template <int np, typename real>
__attribute__((noinline)) void divergence_sphere_cpu(
    const  real_vector<np, real> v,
    const derivative<np, real> & deriv,
    const element<np, real> & elem,
     real_scalar<np, real> div) {
  /* Computes the spherical divergence of v based on the
   * provided metric terms in elem and deriv
   * Returns the divergence in div
   */
  using rs = real_scalar<np, real>;
  using rv = real_vector<np, real>;
  /* Convert to contra variant form and multiply by g */
  ALIGNTO( rv gv, 16);
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int k = 0; k < dim; k++) {
        gv[j][i][k] = elem.metdet[j][i] *
                      (elem.Dinv[j][i][k][0] * v[j][i][0] +
                       elem.Dinv[j][i][k][1] * v[j][i][1]);
      }
    }
  }
  /* Compute d/dx and d/dy */
  ALIGNTO( rs vvtemp, 16);
  for(int l = 0; l < np; l++) {
    for(int j = 0; j < np; j++) {
      ALIGNTO(real dudx00, 16) = 0.0;
      ALIGNTO(real dvdy00, 16) = 0.0;
      for(int i = 0; i < np; i++) {
        dudx00 = dudx00 + deriv.Dvv[l][i] * gv[j][i][0];
        dvdy00 += deriv.Dvv[l][i] * gv[i][j][1];
      }
      div[j][l] = dudx00;
      vvtemp[l][j] = dvdy00;
    }
  }
  constexpr const real rrearth = 1.5683814303638645E-7;

  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      div[i][j] = (div[i][j] + vvtemp[i][j]) *
                  (elem.rmetdet[i][j] * rrearth);
    }
  }
}

#endif
