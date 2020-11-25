
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

#include "common.h"
#define BLOCK_SIZE 16

constexpr const int dim = 2;

template <int np, typename real>
struct element {
  real metdet[np*np];
  real Dinv[np*np*2*2];
  real rmetdet[np*np];
};

template <int np, typename real>
struct derivative {
  real Dvv[np*np];
};

using real = double;


template <int np, typename real>
__attribute__((noinline)) void divergence_sphere_gpu(
    queue &q,
    const  real *v,
    const derivative<np, real> &deriv,
    const element<np, real> &elem,
     real *div) {

  real gv[np*np*dim]; 

  /* Convert to contra variant form and multiply by g */
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int k = 0; k < dim; k++) {
        gv[j*np*dim+i*dim+k] = elem.metdet[j*np+i] *
                      (elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim] * v[j*dim*np+dim*i] +
                       elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim+1] * v[j*dim*np+dim*i+1]);
      }
    }
  }

  buffer<real,1> d_gv(gv, np*np*dim); 
  buffer<real,1> d_Dvv(deriv.Dvv, np*np);
  buffer<real,1> d_div(div, np*np);
  buffer<real,1> d_vvtemp(np*np);
  buffer<real,1> d_rmetdet(elem.rmetdet, np*np);

  range<2> global_work_size((np+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE, 
      (np+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  range<2> local_work_size(BLOCK_SIZE, BLOCK_SIZE);

  q.submit([&] (handler &cgh) {
    auto gv = d_gv.template get_access<sycl_read>(cgh);
    auto Dvv = d_Dvv.template get_access<sycl_read>(cgh);
    auto div = d_div.template get_access<sycl_discard_read_write>(cgh);
    auto vvtemp = d_vvtemp.template get_access<sycl_discard_read_write>(cgh);
    auto rmetdet = d_rmetdet.template get_access<sycl_read>(cgh);
    cgh.parallel_for<class divergence_test>(nd_range<2>(global_work_size, local_work_size), [=] (nd_item<2> item) {
      constexpr const real rrearth = 1.5683814303638645E-7;
      const int l = item.get_global_id(1);
      const int j = item.get_global_id(0); 
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
      item.barrier(access::fence_space::local_space);

      if (l < np && j < np) 
        div[l*np+j] = (div[l*np+j] + vvtemp[l*np+j]) * 
                      (rmetdet[l*np+j] * rrearth);
    });
  });
}

template <int np, typename real>
__attribute__((noinline)) void divergence_sphere_cpu(
    const  real *v,
    const derivative<np, real> & deriv,
    const element<np, real> & elem,
     real *div) {
  /* Computes the spherical divergence of v based on the
   * provided metric terms in elem and deriv
   * Returns the divergence in div
   */
  /* Convert to contra variant form and multiply by g */
  real gv[np*np*dim];

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int k = 0; k < dim; k++) {
        gv[j*np*dim+i*dim+k] = elem.metdet[j*np+i] *
                      (elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim] * v[j*dim*np+dim*i] +
                       elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim+1] * v[j*dim*np+dim*i+1]);
      }
    }
  }
  /* Compute d/dx and d/dy */
  real vvtemp[np*np];
  for(int l = 0; l < np; l++) {
    for(int j = 0; j < np; j++) {
      real dudx00 = 0;
      real dvdy00 = 0;
      for(int i = 0; i < np; i++) {
        dudx00 += deriv.Dvv[l*np+i] * gv[j*np*dim+i*dim];
        dvdy00 += deriv.Dvv[l*np+i] * gv[i*np*dim+j*dim+1];
      }
      div[j*np+l] = dudx00;
      vvtemp[l*np+j] = dvdy00;
    }
  }
  constexpr const real rrearth = 1.5683814303638645E-7;

  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      div[i*np+j] = (div[i*np+j] + vvtemp[i*np+j]) *
                  (elem.rmetdet[i*np+j] * rrearth);
    }
  }
}

#endif
