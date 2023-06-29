#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "constants.h"
#include <sycl/sycl.hpp>

#define R 4
#define NDIM 8

__attribute__ ((always_inline))
void target_inner_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    const float hdx_2, const float hdy_2, const float hdz_2,
    const float coef0,
    const float coefx_1, const float coefx_2, const float coefx_3, const float coefx_4,
    const float coefy_1, const float coefy_2, const float coefy_3, const float coefy_4,
    const float coefz_1, const float coefz_2, const float coefz_3, const float coefz_4,
    sycl::nd_item<3> &item,
    const sycl::local_accessor<float, 3> &s_u,
    const float *u,
    float *v,
    const float *vp,
    const float *eta
    ) {
  const llint i0 = x3 + item.get_local_range(0) * item.get_group(0);
  const llint j0 = y3 + item.get_local_range(1) * item.get_group(1);
  const llint k0 = z3 + item.get_local_range(2) * item.get_group(2);

  const int ti = item.get_local_id(0);
  const int tj = item.get_local_id(1);
  const int tk = item.get_local_id(2);

  const llint i = i0 + ti;
  const llint j = j0 + tj;
  const llint k = k0 + tk;

  const llint sui = ti + R;
  const llint suj = tj + R;
  const llint suk = tk + R;

  s_u[ti][tj][tk] = 0.f;

  if (ti < 2*R && tj < 2*R && tk < 2*R)
    s_u[NDIM+ti][NDIM+tj][NDIM+tk] = 0.f;

  item.barrier(sycl::access::fence_space::local_space);

  const int z_side = ti / R;
  s_u[ti+z_side*NDIM][suj][suk] = u[IDX3_l(i+(z_side*2-1)*R,j,k)];
  const int y_side = tj / R;
  s_u[sui][tj+y_side*NDIM][suk] = u[IDX3_l(i,j+(y_side*2-1)*R,k)];
  s_u[sui][suj][tk] = u[IDX3_l(i,j,k-R)];
  s_u[sui][suj][tk+NDIM] = u[IDX3_l(i,j,k+R)];

  item.barrier(sycl::access::fence_space::local_space);

  if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

  float lap = coef0 * s_u[sui][suj][suk] +
    coefx_1 * (s_u[sui+1][suj][suk] + s_u[sui-1][suj][suk]) +
    coefy_1 * (s_u[sui][suj+1][suk] + s_u[sui][suj-1][suk]) +
    coefz_1 * (s_u[sui][suj][suk+1] + s_u[sui][suj][suk-1]) +
    coefx_2 * (s_u[sui+2][suj][suk] + s_u[sui-2][suj][suk]) +
    coefy_2 * (s_u[sui][suj+2][suk] + s_u[sui][suj-2][suk]) +
    coefz_2 * (s_u[sui][suj][suk+2] + s_u[sui][suj][suk-2]) +
    coefx_3 * (s_u[sui+3][suj][suk] + s_u[sui-3][suj][suk]) +
    coefy_3 * (s_u[sui][suj+3][suk] + s_u[sui][suj-3][suk]) +
    coefz_3 * (s_u[sui][suj][suk+3] + s_u[sui][suj][suk-3]) +
    coefx_4 * (s_u[sui+4][suj][suk] + s_u[sui-4][suj][suk]) +
    coefy_4 * (s_u[sui][suj+4][suk] + s_u[sui][suj-4][suk]) +
    coefz_4 * (s_u[sui][suj][suk+4] + s_u[sui][suj][suk-4]);
  v[IDX3_l(i,j,k)] = 2.f * s_u[sui][suj][suk] + vp[IDX3(i,j,k)] * lap - v[IDX3_l(i,j,k)];
}

__attribute__ ((always_inline))
void target_pml_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    const float hdx_2, const float hdy_2, const float hdz_2,
    const float coef0,
    const float coefx_1, const float coefx_2, const float coefx_3, const float coefx_4,
    const float coefy_1, const float coefy_2, const float coefy_3, const float coefy_4,
    const float coefz_1, const float coefz_2, const float coefz_3, const float coefz_4,
    sycl::nd_item<3> &item,
    const sycl::local_accessor<float, 3> &s_u,
    const float *u,
    float *v,
    const float *vp,
    float *phi,
    const float *eta
    ) {
  const llint i0 = x3 + item.get_local_range(0) * item.get_group(0);
  const llint j0 = y3 + item.get_local_range(1) * item.get_group(1);
  const llint k0 = z3 + item.get_local_range(2) * item.get_group(2);

  const int ti = item.get_local_id(0);
  const int tj = item.get_local_id(1);
  const int tk = item.get_local_id(2);

  const llint i = i0 + ti;
  const llint j = j0 + tj;
  const llint k = k0 + tk;

  const llint sui = ti + R;
  const llint suj = tj + R;
  const llint suk = tk + R;

  s_u[ti][tj][tk] = 0.f;

  if (ti < 2*R && tj < 2*R && tk < 2*R)
    s_u[NDIM+ti][NDIM+tj][NDIM+tk] = 0.f;

  item.barrier(sycl::access::fence_space::local_space);

  const int z_side = ti / R;
  s_u[ti+z_side*NDIM][suj][suk] = u[IDX3_l(i+(z_side*2-1)*R,j,k)];
  const int y_side = tj / R;
  s_u[sui][tj+y_side*NDIM][suk] = u[IDX3_l(i,j+(y_side*2-1)*R,k)];
  s_u[sui][suj][tk] = u[IDX3_l(i,j,k-R)];
  s_u[sui][suj][tk+NDIM] = u[IDX3_l(i,j,k+R)];

  item.barrier(sycl::access::fence_space::local_space);

  if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

  float lap = coef0 * s_u[sui][suj][suk] +
    coefx_1 * (s_u[sui+1][suj][suk] + s_u[sui-1][suj][suk]) +
    coefy_1 * (s_u[sui][suj+1][suk] + s_u[sui][suj-1][suk]) +
    coefz_1 * (s_u[sui][suj][suk+1] + s_u[sui][suj][suk-1]) +
    coefx_2 * (s_u[sui+2][suj][suk] + s_u[sui-2][suj][suk]) +
    coefy_2 * (s_u[sui][suj+2][suk] + s_u[sui][suj-2][suk]) +
    coefz_2 * (s_u[sui][suj][suk+2] + s_u[sui][suj][suk-2]) +
    coefx_3 * (s_u[sui+3][suj][suk] + s_u[sui-3][suj][suk]) +
    coefy_3 * (s_u[sui][suj+3][suk] + s_u[sui][suj-3][suk]) +
    coefz_3 * (s_u[sui][suj][suk+3] + s_u[sui][suj][suk-3]) +
    coefx_4 * (s_u[sui+4][suj][suk] + s_u[sui-4][suj][suk]) +
    coefy_4 * (s_u[sui][suj+4][suk] + s_u[sui][suj-4][suk]) +
    coefz_4 * (s_u[sui][suj][suk+4] + s_u[sui][suj][suk-4]);

  const float s_eta_c = eta[IDX3_eta1(i,j,k)];

  v[IDX3_l(i,j,k)] = ((2.f*s_eta_c + 2.f - s_eta_c*s_eta_c)*s_u[sui][suj][suk] +
      (vp[IDX3(i,j,k)] * (lap + phi[IDX3(i,j,k)]) - v[IDX3_l(i,j,k)])) /
    (2.f*s_eta_c+1.f);

  phi[IDX3(i,j,k)] =
    (phi[IDX3(i,j,k)] -
     ((eta[IDX3_eta1(i+1,j,k)]-eta[IDX3_eta1(i-1,j,k)]) *
      (s_u[sui+1][suj][suk]-s_u[sui-1][suj][suk]) * hdx_2 +
      (eta[IDX3_eta1(i,j+1,k)]-eta[IDX3_eta1(i,j-1,k)]) *
      (s_u[sui][suj+1][suk]-s_u[sui][suj-1][suk]) * hdy_2 +
      (eta[IDX3_eta1(i,j,k+1)]-eta[IDX3_eta1(i,j,k-1)]) *
      (s_u[sui][suj][suk+1]-s_u[sui][suj][suk-1]) * hdz_2)) / (1.f + s_eta_c);
}

void minimod(sycl::queue &q,
    uint nsteps, double *time_kernel,
    llint nx, llint ny, llint nz,
    llint x1, llint x2, llint x3, llint x4, llint x5, llint x6,
    llint y1, llint y2, llint y3, llint y4, llint y5, llint y6,
    llint z1, llint z2, llint z3, llint z4, llint z5, llint z6,
    llint lx, llint ly, llint lz,
    llint sx, llint sy, llint sz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict coefx, const float *__restrict coefy, const float *__restrict coefz,
    float *__restrict u, const float *__restrict v, const float *__restrict vp,
    const float *__restrict phi, const float *__restrict eta, const float *__restrict source
     ) {
  struct timespec start, end;

  const llint size_u = (nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
  const llint size_v = size_u;
  const llint size_phi = nx*ny*nz;
  const llint size_vp = size_phi;
  const llint size_eta = (nx+2)*(ny+2)*(nz+2);

  const llint size_u_ext = ((((nx+NDIM-1) / NDIM + 1) * NDIM) + 2 * lx)
    * ((((ny+NDIM-1) / NDIM + 1) * NDIM) + 2 * ly)
    * ((((nz+NDIM-1) / NDIM + 1) * NDIM) + 2 * lz);

  float *d_u = sycl::malloc_device<float>(size_u, q);
  float *d_v = sycl::malloc_device<float>(size_u, q);
  float *d_vp = sycl::malloc_device<float>(size_vp, q);
  float *d_phi = sycl::malloc_device<float>(size_phi, q);
  float *d_eta = sycl::malloc_device<float>(size_eta, q);

  q.memcpy(d_u, u, sizeof(float) * size_u);
  q.memcpy(d_v, v, sizeof(float) * size_v);
  q.memcpy(d_vp, vp, sizeof(float) * size_vp);
  q.memcpy(d_phi, phi, sizeof(float) * size_phi);
  q.memcpy(d_eta, eta, sizeof(float) * size_eta);

  const llint xmin = 0; const llint xmax = nx;
  const llint ymin = 0; const llint ymax = ny;

  sycl::range<3> threadsPerBlock(NDIM, NDIM, NDIM);
  const float coef0 = coefx[0]+coefy[0]+coefz[0];
  const float coefx_1 = coefx[1];
  const float coefx_2 = coefx[2];
  const float coefx_3 = coefx[3];
  const float coefx_4 = coefx[4];
  const float coefy_1 = coefy[1];
  const float coefy_2 = coefy[2];
  const float coefy_3 = coefy[3];
  const float coefy_4 = coefy[4];
  const float coefz_1 = coefz[1];
  const float coefz_2 = coefz[2];
  const float coefz_3 = coefz[3];
  const float coefz_4 = coefz[4];

  #ifdef DEBUG
  const uint npo = 100;
  #endif

  q.wait();
  clock_gettime(CLOCK_REALTIME, &start);

  for (uint istep = 1; istep <= nsteps; ++istep) {

    sycl::range<3> n_block_front((nx+NDIM-1) / NDIM * NDIM,
                                 (ny+NDIM-1) / NDIM * NDIM,
                                 (z2-z1+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class front>(
        sycl::nd_range<3>(n_block_front,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_pml_3d_kernel(nx,ny,nz,
            xmin,xmax,ymin,ymax,z1,z2,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
            item, s_u, d_u, d_v, d_vp, d_phi, d_eta);
      });
    });

    sycl::range<3> n_block_top((nx+NDIM-1) / NDIM * NDIM,
                               (y2-y1+NDIM-1) / NDIM * NDIM,
                               (z4-z3+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class top>(
        sycl::nd_range<3>(n_block_top,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_pml_3d_kernel(nx,ny,nz,
            xmin,xmax,y1,y2,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
	    item, s_u, d_u, d_v, d_vp, d_phi, d_eta);
      });
    });

    sycl::range<3> n_block_left((x2-x1+NDIM-1) / NDIM * NDIM,
                                (y4-y3+NDIM-1) / NDIM * NDIM,
                                (z4-z3+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class left>(
        sycl::nd_range<3>(n_block_left,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_pml_3d_kernel(nx,ny,nz,
            x1,x2,y3,y4,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
	    item, s_u, d_u, d_v, d_vp, d_phi, d_eta);
      });
    });

    sycl::range<3> n_block_center ((x4-x3+NDIM-1) / NDIM * NDIM ,
                                   (y4-y3+NDIM-1) / NDIM * NDIM ,
                                   (z4-z3+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class center>(
        sycl::nd_range<3>(n_block_center,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_inner_3d_kernel(nx,ny,nz,
            x3,x4,y3,y4,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
	    item, s_u, d_u, d_v, d_vp, d_eta);
      });
    });

    sycl::range<3> n_block_right((x6-x5+NDIM-1) / NDIM * NDIM,
                                 (y4-y3+NDIM-1) / NDIM * NDIM,
                                 (z4-z3+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class right>(
        sycl::nd_range<3>(n_block_right,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_pml_3d_kernel(nx,ny,nz,
            x5,x6,y3,y4,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
	    item, s_u, d_u, d_v, d_vp, d_phi, d_eta);
      });
    });

    sycl::range<3> n_block_bottom((nx+NDIM-1) / NDIM * NDIM,
                                  (y6-y5+NDIM-1) / NDIM * NDIM,
                                  (z4-z3+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class bottom>(sycl::nd_range<3>(n_block_bottom,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_pml_3d_kernel(nx,ny,nz,
            xmin,xmax,y5,y6,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
	    item, s_u, d_u, d_v, d_vp, d_phi, d_eta);
      });
    });

    sycl::range<3> n_block_back((nx+NDIM-1) / NDIM * NDIM,
                                (ny+NDIM-1) / NDIM * NDIM,
                                (z6-z5+NDIM-1) / NDIM * NDIM);

    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor <float, 3> s_u (sycl::range<3>{
          NDIM+2*R,
          NDIM+2*R,
          NDIM+2*R}, h);
      h.parallel_for<class back>(
        sycl::nd_range<3>(n_block_back,threadsPerBlock), [=] (sycl::nd_item<3> item) {
        target_pml_3d_kernel(nx,ny,nz,
            xmin,xmax,ymin,ymax,z5,z6,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coef0,
            coefx_1, coefx_2, coefx_3, coefx_4,
            coefy_1, coefy_2, coefy_3, coefy_4,
            coefz_1, coefz_2, coefz_3, coefz_4,
	    item, s_u, d_u, d_v, d_vp, d_phi, d_eta);
      });
    });

    llint idx = IDX3_l(sx,sy,sz);
    float s = source[istep];
    q.submit([&] (sycl::handler &h) {
      h.single_task<class add_source>([=]() {
        d_v[idx] += s;
      });
    });

    float *t = d_u;
    d_u = d_v;
    d_v = t;

    // Print out
    #ifdef DEBUG
    if (istep % npo == 0) {
      printf("time step %u / %u\n", istep, nsteps);
    }
    #endif
  }

  q.wait();
  clock_gettime(CLOCK_REALTIME, &end);
  *time_kernel = (end.tv_sec  - start.tv_sec) +
                 (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

  q.memcpy(u, d_u, sizeof(float) * size_u).wait();
  sycl::free(d_u, q);
  sycl::free(d_v, q);
  sycl::free(d_vp, q);
  sycl::free(d_phi, q);
  sycl::free(d_eta, q);
}
