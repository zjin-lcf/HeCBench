#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "constants.h"
#include "common.h"

#define N_RADIUS 4
#define N_THREADS_PER_BLOCK_DIM 8

void target_inner_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    const float hdx_2, const float hdy_2, const float hdz_2,
    const float coef0,
    const float coefx_1, const float coefx_2, const float coefx_3, const float coefx_4,
    const float coefy_1, const float coefy_2, const float coefy_3, const float coefy_4,
    const float coefz_1, const float coefz_2, const float coefz_3, const float coefz_4,
    nd_item<3> &item, 
    const accessor <float, 3, sycl_read_write, access::target::local> &s_u,
    const accessor<float, 1, sycl_read, access::target::global_buffer> &u,
    const accessor<float, 1, sycl_read_write, access::target::global_buffer> &v,
    const accessor<float, 1, sycl_read, access::target::global_buffer> &vp,
    const accessor<float, 1, sycl_read, access::target::global_buffer> &eta
    ) {
  const llint i0 = x3 + item.get_local_range(0) * item.get_group(0);
  const llint j0 = y3 + item.get_local_range(1) * item.get_group(1);
  const llint k0 = z3 + item.get_local_range(2) * item.get_group(2); 

  const llint i = i0 + item.get_local_id(0);
  const llint j = j0 + item.get_local_id(1);
  const llint k = k0 + item.get_local_id(2);

  const llint sui = item.get_local_id(0) + N_RADIUS;
  const llint suj = item.get_local_id(1) + N_RADIUS;
  const llint suk = item.get_local_id(2) + N_RADIUS;

  const int z_side = item.get_local_id(0) / N_RADIUS;
  s_u[item.get_local_id(0)+z_side*N_THREADS_PER_BLOCK_DIM][suj][suk] = u[IDX3_l(i0+item.get_local_id(0)+(z_side*2-1)*N_RADIUS,j,k)];
  const int y_side = item.get_local_id(1) / N_RADIUS;
  s_u[sui][item.get_local_id(1)+y_side*N_THREADS_PER_BLOCK_DIM][suk] = u[IDX3_l(i,j0+item.get_local_id(1)+(y_side*2-1)*N_RADIUS,k)];
  s_u[sui][suj][item.get_local_id(2)] = u[IDX3_l(i,j,k0+item.get_local_id(2)-N_RADIUS)];
  s_u[sui][suj][item.get_local_id(2)+N_THREADS_PER_BLOCK_DIM] = u[IDX3_l(i,j,k0+item.get_local_id(2)+N_RADIUS)];

  item.barrier(access::fence_space::local_space);

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

void target_pml_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    const float hdx_2, const float hdy_2, const float hdz_2,
    const float coef0,
    const float coefx_1, const float coefx_2, const float coefx_3, const float coefx_4,
    const float coefy_1, const float coefy_2, const float coefy_3, const float coefy_4,
    const float coefz_1, const float coefz_2, const float coefz_3, const float coefz_4,
    nd_item<3> &item, 
    const accessor <float, 3, sycl_read_write, access::target::local> &s_u,
    const accessor<float, 1, sycl_read, access::target::global_buffer> &u,
    const accessor<float, 1, sycl_read_write, access::target::global_buffer> &v,
    const accessor<float, 1, sycl_read, access::target::global_buffer> &vp,
    const accessor<float, 1, sycl_read_write, access::target::global_buffer> &phi,
    const accessor<float, 1, sycl_read, access::target::global_buffer> &eta
    ) {
  const llint i0 = x3 + item.get_local_range(0) * item.get_group(0);
  const llint j0 = y3 + item.get_local_range(1) * item.get_group(1);
  const llint k0 = z3 + item.get_local_range(2) * item.get_group(2); 

  const llint i = i0 + item.get_local_id(0);
  const llint j = j0 + item.get_local_id(1);
  const llint k = k0 + item.get_local_id(2);

  const llint sui = item.get_local_id(0) + N_RADIUS;
  const llint suj = item.get_local_id(1) + N_RADIUS;
  const llint suk = item.get_local_id(2) + N_RADIUS;

  const int z_side = item.get_local_id(0) / N_RADIUS;
  s_u[item.get_local_id(0)+z_side*N_THREADS_PER_BLOCK_DIM][suj][suk] = u[IDX3_l(i0+item.get_local_id(0)+(z_side*2-1)*N_RADIUS,j,k)];
  const int y_side = item.get_local_id(1) / N_RADIUS;
  s_u[sui][item.get_local_id(1)+y_side*N_THREADS_PER_BLOCK_DIM][suk] = u[IDX3_l(i,j0+item.get_local_id(1)+(y_side*2-1)*N_RADIUS,k)];
  s_u[sui][suj][item.get_local_id(2)] = u[IDX3_l(i,j,k0+item.get_local_id(2)-N_RADIUS)];
  s_u[sui][suj][item.get_local_id(2)+N_THREADS_PER_BLOCK_DIM] = u[IDX3_l(i,j,k0+item.get_local_id(2)+N_RADIUS)];

  item.barrier(access::fence_space::local_space);

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


void minimod( queue &q,
    uint nsteps, double *time_kernel,
    llint nx, llint ny, llint nz,
    llint x1, llint x2, llint x3, llint x4, llint x5, llint x6,
    llint y1, llint y2, llint y3, llint y4, llint y5, llint y6,
    llint z1, llint z2, llint z3, llint z4, llint z5, llint z6,
    llint lx, llint ly, llint lz,
    llint sx, llint sy, llint sz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ coefx, const float *__restrict__ coefy, const float *__restrict__ coefz,
    float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta, const float *__restrict__ source
     ) {
  struct timespec start, end;

  const llint size_u = (nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
  const llint size_v = size_u;
  const llint size_phi = nx*ny*nz;
  const llint size_vp = size_phi;
  const llint size_eta = (nx+2)*(ny+2)*(nz+2);

  const llint size_u_ext = ((((nx+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM + 1) * N_THREADS_PER_BLOCK_DIM) + 2 * lx)
    * ((((ny+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM + 1) * N_THREADS_PER_BLOCK_DIM) + 2 * ly)
    * ((((nz+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM + 1) * N_THREADS_PER_BLOCK_DIM) + 2 * lz);


  buffer<float, 1> d_u (size_u_ext);
  buffer<float, 1> d_v (size_u_ext);
  buffer<float, 1> d_vp (vp, size_vp);
  buffer<float, 1> d_phi (phi, size_phi);
  buffer<float, 1> d_eta (eta, size_eta);

  q.submit([&] (handler &h) {
      auto du = d_u.get_access<sycl_write>(h, range<1>(size_u));
      h.copy(u, du);
      });
  q.submit([&] (handler &h) {
      auto dv = d_v.get_access<sycl_write>(h, range<1>(size_v));
      h.copy(v, dv);
      });


  const llint xmin = 0; const llint xmax = nx;
  const llint ymin = 0; const llint ymax = ny;

  range<3> threadsPerBlock(N_THREADS_PER_BLOCK_DIM, N_THREADS_PER_BLOCK_DIM, N_THREADS_PER_BLOCK_DIM);
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

  const uint npo = 100;
  for (uint istep = 1; istep <= nsteps; ++istep) {
    clock_gettime(CLOCK_REALTIME, &start);

    range<3> n_block_front(
        (nx+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (ny+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (z2-z1+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM
	);

    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        auto phi = d_phi.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class front>(nd_range<3>(n_block_front,threadsPerBlock), [=] (nd_item<3> item) {

            target_pml_3d_kernel(nx,ny,nz,
                xmin,xmax,ymin,ymax,z1,z2,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, phi, eta);
            });
    });

    range<3> n_block_top(
        (nx+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (y2-y1+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (z4-z3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM
	);

    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        auto phi = d_phi.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class top>(nd_range<3>(n_block_top,threadsPerBlock), [=] (nd_item<3> item) {

            target_pml_3d_kernel(nx,ny,nz,
                xmin,xmax,y1,y2,z3,z4,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, phi, eta);
            });
    });

    range<3> n_block_left(
        (x2-x1+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (y4-y3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (z4-z3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM
	);
    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        auto phi = d_phi.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class left>(nd_range<3>(n_block_left,threadsPerBlock), [=] (nd_item<3> item) {
            target_pml_3d_kernel(nx,ny,nz,
                x1,x2,y3,y4,z3,z4,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, phi, eta);

            });
    });


    range<3> n_block_center (
        (x4-x3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM ,
        (y4-y3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM ,
        (z4-z3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM 
	);
    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class center>(nd_range<3>(n_block_center,threadsPerBlock), [=] (nd_item<3> item) {
            target_inner_3d_kernel(nx,ny,nz,
                x3,x4,y3,y4,z3,z4,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, eta);
            });
    });

    range<3> n_block_right(
        (x6-x5+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (y4-y3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (z4-z3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM
	);
    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        auto phi = d_phi.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class right>(nd_range<3>(n_block_right,threadsPerBlock), [=] (nd_item<3> item) {
            target_pml_3d_kernel(nx,ny,nz,
                x5,x6,y3,y4,z3,z4,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, phi, eta);
            });
    });

    range<3> n_block_bottom(
        (nx+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (y6-y5+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (z4-z3+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM
	);
    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        auto phi = d_phi.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class bottom>(nd_range<3>(n_block_bottom,threadsPerBlock), [=] (nd_item<3> item) {
            target_pml_3d_kernel(nx,ny,nz,
                xmin,xmax,y5,y6,z3,z4,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, phi, eta);
            });
    });

    range<3> n_block_back(
        (nx+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (ny+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM,
        (z6-z5+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM * N_THREADS_PER_BLOCK_DIM
	);
    q.submit([&] (handler &h) {
        auto u = d_u.get_access<sycl_read>(h);
        auto vp = d_vp.get_access<sycl_read>(h);
        auto eta = d_eta.get_access<sycl_read>(h);
        auto v = d_v.get_access<sycl_read_write>(h);
        auto phi = d_phi.get_access<sycl_read_write>(h);
        accessor <float, 3, sycl_read_write, access::target::local> s_u ({
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS,
            N_THREADS_PER_BLOCK_DIM+2*N_RADIUS}, h);
        h.parallel_for<class back>(nd_range<3>(n_block_back,threadsPerBlock), [=] (nd_item<3> item) {
            target_pml_3d_kernel(nx,ny,nz,
                xmin,xmax,ymin,ymax,z5,z6,
                lx,ly,lz,
                hdx_2, hdy_2, hdz_2,
                coef0,
                coefx_1, coefx_2, coefx_3, coefx_4,
                coefy_1, coefy_2, coefy_3, coefy_4,
                coefz_1, coefz_2, coefz_3, coefz_4, 
                item, s_u, u, v, vp, phi, eta);
        });
    });

    llint idx = IDX3_l(sx,sy,sz);
    float s = source[istep];
    q.submit([&] (handler &h) {
        auto g_u = d_v.get_access<sycl_read_write>(h);
        h.single_task<class add_source>([=]() {
            g_u[idx] += s;
        });
    });

    clock_gettime(CLOCK_REALTIME, &end);
    *time_kernel += (end.tv_sec  - start.tv_sec) +
      (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

    auto t = std::move(d_u);
    d_u = std::move(d_v);
    d_v = std::move(t);

    // Print out
    if (istep % npo == 0) {
      printf("time step %u / %u\n", istep, nsteps);
    }
  }

  q.submit([&] (handler &h) {
      auto du = d_u.get_access<sycl_read>(h, range<1>(size_u));
      h.copy(du, u);
  });
  q.wait();
}
