#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "constants.h"

#define N_RADIUS 4
#define N_THREADS_PER_BLOCK_DIM 8

void target_inner_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta
,
    sycl::nd_item<3> item_ct1, dpct::accessor<float, dpct::local, 3> s_u) {

    const llint i0 =
        x3 + item_ct1.get_group(0) * item_ct1.get_local_range().get(0);
    const llint j0 =
        y3 + item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
    const llint k0 =
        z3 + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);

    const llint i = i0 + item_ct1.get_local_id(0);
    const llint j = j0 + item_ct1.get_local_id(1);
    const llint k = k0 + item_ct1.get_local_id(2);

    const llint sui = item_ct1.get_local_id(0) + N_RADIUS;
    const llint suj = item_ct1.get_local_id(1) + N_RADIUS;
    const llint suk = item_ct1.get_local_id(2) + N_RADIUS;

    const int z_side = item_ct1.get_local_id(0) / N_RADIUS;
    s_u[item_ct1.get_local_id(0) + z_side * N_THREADS_PER_BLOCK_DIM][suj][suk] =
        u[IDX3_l(i0 + item_ct1.get_local_id(0) + (z_side * 2 - 1) * N_RADIUS, j,
                 k)];
    const int y_side = item_ct1.get_local_id(1) / N_RADIUS;
    s_u[sui][item_ct1.get_local_id(1) + y_side * N_THREADS_PER_BLOCK_DIM][suk] =
        u[IDX3_l(i, j0 + item_ct1.get_local_id(1) + (y_side * 2 - 1) * N_RADIUS,
                 k)];
    s_u[sui][suj][item_ct1.get_local_id(2)] =
        u[IDX3_l(i, j, k0 + item_ct1.get_local_id(2) - N_RADIUS)];
    s_u[sui][suj][item_ct1.get_local_id(2) + N_THREADS_PER_BLOCK_DIM] =
        u[IDX3_l(i, j, k0 + item_ct1.get_local_id(2) + N_RADIUS)];

    item_ct1.barrier();

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
/*
    float lap = __fmaf_rn(coef0, s_u[sui][suj][suk]
              , __fmaf_rn(coefx_1, __fadd_rn(s_u[sui+1][suj][suk],s_u[sui-1][suj][suk])
              , __fmaf_rn(coefy_1, __fadd_rn(s_u[sui][suj+1][suk],s_u[sui][suj-1][suk])
              , __fmaf_rn(coefz_1, __fadd_rn(s_u[sui][suj][suk+1],s_u[sui][suj][suk-1])
              , __fmaf_rn(coefx_2, __fadd_rn(s_u[sui+2][suj][suk],s_u[sui-2][suj][suk])
              , __fmaf_rn(coefy_2, __fadd_rn(s_u[sui][suj+2][suk],s_u[sui][suj-2][suk])
              , __fmaf_rn(coefz_2, __fadd_rn(s_u[sui][suj][suk+2],s_u[sui][suj][suk-2])
              , __fmaf_rn(coefx_3, __fadd_rn(s_u[sui+3][suj][suk],s_u[sui-3][suj][suk])
              , __fmaf_rn(coefy_3, __fadd_rn(s_u[sui][suj+3][suk],s_u[sui][suj-3][suk])
              , __fmaf_rn(coefz_3, __fadd_rn(s_u[sui][suj][suk+3],s_u[sui][suj][suk-3])
              , __fmaf_rn(coefx_4, __fadd_rn(s_u[sui+4][suj][suk],s_u[sui-4][suj][suk])
              , __fmaf_rn(coefy_4, __fadd_rn(s_u[sui][suj+4][suk],s_u[sui][suj-4][suk])
              , __fmul_rn(coefz_4, __fadd_rn(s_u[sui][suj][suk+4],s_u[sui][suj][suk-4])
    )))))))))))));
*/

/*
    v[IDX3_l(i,j,k)] = __fmaf_rn(2.f, s_u[sui][suj][suk],
        __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3_l(i,j,k)])
    );
*/
    v[IDX3_l(i,j,k)] = 2.f * s_u[sui][suj][suk] + vp[IDX3(i,j,k)] * lap - v[IDX3_l(i,j,k)];
}

void target_pml_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta
,
    sycl::nd_item<3> item_ct1, dpct::accessor<float, dpct::local, 3> s_u) {

    const llint i0 =
        x3 + item_ct1.get_group(0) * item_ct1.get_local_range().get(0);
    const llint j0 =
        y3 + item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
    const llint k0 =
        z3 + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);

    const llint i = i0 + item_ct1.get_local_id(0);
    const llint j = j0 + item_ct1.get_local_id(1);
    const llint k = k0 + item_ct1.get_local_id(2);

    const llint sui = item_ct1.get_local_id(0) + N_RADIUS;
    const llint suj = item_ct1.get_local_id(1) + N_RADIUS;
    const llint suk = item_ct1.get_local_id(2) + N_RADIUS;

    const int z_side = item_ct1.get_local_id(0) / N_RADIUS;
    s_u[item_ct1.get_local_id(0) + z_side * N_THREADS_PER_BLOCK_DIM][suj][suk] =
        u[IDX3_l(i0 + item_ct1.get_local_id(0) + (z_side * 2 - 1) * N_RADIUS, j,
                 k)];
    const int y_side = item_ct1.get_local_id(1) / N_RADIUS;
    s_u[sui][item_ct1.get_local_id(1) + y_side * N_THREADS_PER_BLOCK_DIM][suk] =
        u[IDX3_l(i, j0 + item_ct1.get_local_id(1) + (y_side * 2 - 1) * N_RADIUS,
                 k)];
    s_u[sui][suj][item_ct1.get_local_id(2)] =
        u[IDX3_l(i, j, k0 + item_ct1.get_local_id(2) - N_RADIUS)];
    s_u[sui][suj][item_ct1.get_local_id(2) + N_THREADS_PER_BLOCK_DIM] =
        u[IDX3_l(i, j, k0 + item_ct1.get_local_id(2) + N_RADIUS)];

    item_ct1.barrier();

    if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

/*
    float lap = __fmaf_rn(coef0, s_u[sui][suj][suk]
        , __fmaf_rn(coefx_1, __fadd_rn(s_u[sui+1][suj][suk],s_u[sui-1][suj][suk])
        , __fmaf_rn(coefy_1, __fadd_rn(s_u[sui][suj+1][suk],s_u[sui][suj-1][suk])
        , __fmaf_rn(coefz_1, __fadd_rn(s_u[sui][suj][suk+1],s_u[sui][suj][suk-1])
        , __fmaf_rn(coefx_2, __fadd_rn(s_u[sui+2][suj][suk],s_u[sui-2][suj][suk])
        , __fmaf_rn(coefy_2, __fadd_rn(s_u[sui][suj+2][suk],s_u[sui][suj-2][suk])
        , __fmaf_rn(coefz_2, __fadd_rn(s_u[sui][suj][suk+2],s_u[sui][suj][suk-2])
        , __fmaf_rn(coefx_3, __fadd_rn(s_u[sui+3][suj][suk],s_u[sui-3][suj][suk])
        , __fmaf_rn(coefy_3, __fadd_rn(s_u[sui][suj+3][suk],s_u[sui][suj-3][suk])
        , __fmaf_rn(coefz_3, __fadd_rn(s_u[sui][suj][suk+3],s_u[sui][suj][suk-3])
        , __fmaf_rn(coefx_4, __fadd_rn(s_u[sui+4][suj][suk],s_u[sui-4][suj][suk])
        , __fmaf_rn(coefy_4, __fadd_rn(s_u[sui][suj+4][suk],s_u[sui][suj-4][suk])
        , __fmul_rn(coefz_4, __fadd_rn(s_u[sui][suj][suk+4],s_u[sui][suj][suk-4])
    )))))))))))));
*/
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

/*
    v[IDX3_l(i,j,k)] = __fdiv_rn(
        __fmaf_rn(
            __fmaf_rn(2.f, s_eta_c,
                __fsub_rn(2.f,
                    __fmul_rn(s_eta_c, s_eta_c)
                )
            ),
            s_u[sui][suj][suk],
            __fmaf_rn(
                vp[IDX3(i,j,k)],
                __fadd_rn(lap, phi[IDX3(i,j,k)]),
                -v[IDX3_l(i,j,k)]
            )
        ),
        __fmaf_rn(2.f, s_eta_c, 1.f)
    );
*/
    v[IDX3_l(i,j,k)] = ((2.f*s_eta_c + 2.f - s_eta_c*s_eta_c)*s_u[sui][suj][suk] + 
		    (vp[IDX3(i,j,k)] * (lap + phi[IDX3(i,j,k)]) - v[IDX3_l(i,j,k)])) / 
	    (2.f*s_eta_c+1.f);

/*
    phi[IDX3(i,j,k)] = __fdiv_rn(
            __fsub_rn(
                phi[IDX3(i,j,k)],
                __fmaf_rn(
                __fmul_rn(
                    __fsub_rn(eta[IDX3_eta1(i+1,j,k)], eta[IDX3_eta1(i-1,j,k)]),
                    __fsub_rn(s_u[sui+1][suj][suk], s_u[sui-1][suj][suk])
                ), hdx_2,
                __fmaf_rn(
                __fmul_rn(
                    __fsub_rn(eta[IDX3_eta1(i,j+1,k)], eta[IDX3_eta1(i,j-1,k)]),
                    __fsub_rn(s_u[sui][suj+1][suk], s_u[sui][suj-1][suk])
                ), hdy_2,
                __fmul_rn(
                    __fmul_rn(
                        __fsub_rn(eta[IDX3_eta1(i,j,k+1)], eta[IDX3_eta1(i,j,k-1)]),
                        __fsub_rn(s_u[sui][suj][suk+1], s_u[sui][suj][suk-1])
                    ),
                hdz_2)
                ))
            )
        ,
        __fadd_rn(1.f, s_eta_c)
    );
*/
    phi[IDX3(i,j,k)] = 
     (phi[IDX3(i,j,k)] - 
     ((eta[IDX3_eta1(i+1,j,k)]-eta[IDX3_eta1(i-1,j,k)]) * 
     (s_u[sui+1][suj][suk]-s_u[sui-1][suj][suk]) * hdx_2 + 
     (eta[IDX3_eta1(i,j+1,k)]-eta[IDX3_eta1(i,j-1,k)]) *
     (s_u[sui][suj+1][suk]-s_u[sui][suj-1][suk]) * hdy_2 +
     (eta[IDX3_eta1(i,j,k+1)]-eta[IDX3_eta1(i,j,k-1)]) *
     (s_u[sui][suj][suk+1]-s_u[sui][suj][suk-1]) * hdz_2)) / (1.f + s_eta_c);
}

void kernel_add_source_kernel(float *g_u, llint idx, float source) {
    g_u[idx] += source;
}

void target(uint nsteps, double *time_kernel, llint nx, llint ny, llint nz,
            llint x1, llint x2, llint x3, llint x4, llint x5, llint x6,
            llint y1, llint y2, llint y3, llint y4, llint y5, llint y6,
            llint z1, llint z2, llint z3, llint z4, llint z5, llint z6,
            llint lx, llint ly, llint lz, llint sx, llint sy, llint sz,
            float hdx_2, float hdy_2, float hdz_2,
            const float *__restrict__ coefx, const float *__restrict__ coefy,
            const float *__restrict__ coefz, float *__restrict__ u,
            const float *__restrict__ v, const float *__restrict__ vp,
            const float *__restrict__ phi, const float *__restrict__ eta,
            const float *__restrict__ source) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    struct timespec start, end;

    const llint size_u = (nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
    const llint size_v = size_u;
    const llint size_phi = nx*ny*nz;
    const llint size_vp = size_phi;
    const llint size_eta = (nx+2)*(ny+2)*(nz+2);

    const llint size_u_ext = ((((nx+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM + 1) * N_THREADS_PER_BLOCK_DIM) + 2 * lx)
                           * ((((ny+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM + 1) * N_THREADS_PER_BLOCK_DIM) + 2 * ly)
                           * ((((nz+N_THREADS_PER_BLOCK_DIM-1) / N_THREADS_PER_BLOCK_DIM + 1) * N_THREADS_PER_BLOCK_DIM) + 2 * lz);

    float *d_u, *d_v, *d_vp, *d_phi, *d_eta;
    dpct::dpct_malloc(&d_u, sizeof(float) * size_u_ext);
    dpct::dpct_malloc(&d_v, sizeof(float) * size_u_ext);
    dpct::dpct_malloc(&d_vp, sizeof(float) * size_vp);
    dpct::dpct_malloc(&d_phi, sizeof(float) * size_phi);
    dpct::dpct_malloc(&d_eta, sizeof(float) * size_eta);

    dpct::dpct_memcpy(d_u, u, sizeof(float) * size_u, dpct::host_to_device);
    dpct::dpct_memcpy(d_v, v, sizeof(float) * size_v, dpct::host_to_device);
    dpct::dpct_memcpy(d_vp, vp, sizeof(float) * size_vp, dpct::host_to_device);
    dpct::dpct_memcpy(d_phi, phi, sizeof(float) * size_phi,
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_eta, eta, sizeof(float) * size_eta,
                      dpct::host_to_device);

    const llint xmin = 0; const llint xmax = nx;
    const llint ymin = 0; const llint ymax = ny;

    sycl::range<3> threadsPerBlock(N_THREADS_PER_BLOCK_DIM,
                                   N_THREADS_PER_BLOCK_DIM,
                                   N_THREADS_PER_BLOCK_DIM);

    const uint npo = 100;
    for (uint istep = 1; istep <= nsteps; ++istep) {
        clock_gettime(CLOCK_REALTIME, &start);

        sycl::range<3> n_block_front(
            (z2 - z1 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (ny + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (nx + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_front * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_pml_3d_kernel(
                            nx, ny, nz, xmin, xmax, ymin, ymax, z1, z2, lx, ly,
                            lz, hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        sycl::range<3> n_block_top(
            (z4 - z3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (y2 - y1 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (nx + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_top * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_pml_3d_kernel(
                            nx, ny, nz, xmin, xmax, y1, y2, z3, z4, lx, ly, lz,
                            hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        sycl::range<3> n_block_left(
            (z4 - z3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (y4 - y3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (x2 - x1 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_left * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_pml_3d_kernel(
                            nx, ny, nz, x1, x2, y3, y4, z3, z4, lx, ly, lz,
                            hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        sycl::range<3> n_block_center(
            (z4 - z3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (y4 - y3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (x4 - x3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_center * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_inner_3d_kernel(
                            nx, ny, nz, x3, x4, y3, y4, z3, z4, lx, ly, lz,
                            hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (const float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        sycl::range<3> n_block_right(
            (z4 - z3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (y4 - y3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (x6 - x5 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_right * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_pml_3d_kernel(
                            nx, ny, nz, x5, x6, y3, y4, z3, z4, lx, ly, lz,
                            hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        sycl::range<3> n_block_bottom(
            (z4 - z3 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (y6 - y5 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (nx + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_bottom * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_pml_3d_kernel(
                            nx, ny, nz, xmin, xmax, y5, y6, z3, z4, lx, ly, lz,
                            hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        sycl::range<3> n_block_back(
            (z6 - z5 + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (ny + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM,
            (nx + N_THREADS_PER_BLOCK_DIM - 1) / N_THREADS_PER_BLOCK_DIM);
        {
            dpct::buffer_t d_u_buf_ct28 = dpct::get_buffer(d_u);
            dpct::buffer_t d_v_buf_ct29 = dpct::get_buffer(d_v);
            dpct::buffer_t d_vp_buf_ct30 = dpct::get_buffer(d_vp);
            dpct::buffer_t d_phi_buf_ct31 = dpct::get_buffer(d_phi);
            dpct::buffer_t d_eta_buf_ct32 = dpct::get_buffer(d_eta);
            q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<3> s_u_range_ct1(
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/,
                    16 /*N_THREADS_PER_BLOCK_DIM+2*N_RADIUS*/);

                sycl::accessor<float, 3, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    s_u_acc_ct1(s_u_range_ct1, cgh);
                auto d_u_acc_ct28 =
                    d_u_buf_ct28.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_v_acc_ct29 =
                    d_v_buf_ct29.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_vp_acc_ct30 =
                    d_vp_buf_ct30.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_phi_acc_ct31 =
                    d_phi_buf_ct31.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_eta_acc_ct32 =
                    d_eta_buf_ct32.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = n_block_back * threadsPerBlock;

                auto coefx_coefy_coefz_ct15 = coefx[0] + coefy[0] + coefz[0];
                auto coefx_ct16 = coefx[1];
                auto coefx_ct17 = coefx[2];
                auto coefx_ct18 = coefx[3];
                auto coefx_ct19 = coefx[4];
                auto coefy_ct20 = coefy[1];
                auto coefy_ct21 = coefy[2];
                auto coefy_ct22 = coefy[3];
                auto coefy_ct23 = coefy[4];
                auto coefz_ct24 = coefz[1];
                auto coefz_ct25 = coefz[2];
                auto coefz_ct26 = coefz[3];
                auto coefz_ct27 = coefz[4];

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(threadsPerBlock.get(2),
                                                     threadsPerBlock.get(1),
                                                     threadsPerBlock.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        target_pml_3d_kernel(
                            nx, ny, nz, xmin, xmax, ymin, ymax, z5, z6, lx, ly,
                            lz, hdx_2, hdy_2, hdz_2, coefx_coefy_coefz_ct15,
                            coefx_ct16, coefx_ct17, coefx_ct18, coefx_ct19,
                            coefy_ct20, coefy_ct21, coefy_ct22, coefy_ct23,
                            coefz_ct24, coefz_ct25, coefz_ct26, coefz_ct27,
                            (const float *)(&d_u_acc_ct28[0]),
                            (float *)(&d_v_acc_ct29[0]),
                            (const float *)(&d_vp_acc_ct30[0]),
                            (float *)(&d_phi_acc_ct31[0]),
                            (const float *)(&d_eta_acc_ct32[0]), item_ct1,
                            dpct::accessor<float, dpct::local, 3>(
                                s_u_acc_ct1, s_u_range_ct1));
                    });
            });
        }

        {
            dpct::buffer_t d_v_buf_ct0 = dpct::get_buffer(d_v);
            q_ct1.submit([&](sycl::handler &cgh) {
                auto d_v_acc_ct0 =
                    d_v_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

                auto source_istep_ct2 = source[istep];

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
                                                   sycl::range<3>(1, 1, 1)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     kernel_add_source_kernel(
                                         (float *)(&d_v_acc_ct0[0]),
                                         IDX3_l(sx, sy, sz), source_istep_ct2);
                                 });
            });
        }
        clock_gettime(CLOCK_REALTIME, &end);
        *time_kernel += (end.tv_sec  - start.tv_sec) +
                        (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

        float *t = d_u;
        d_u = d_v;
        d_v = t;

        // Print out
        if (istep % npo == 0) {
            printf("time step %u / %u\n", istep, nsteps);
        }
    }

    dpct::dpct_memcpy(u, d_u, sizeof(float) * size_u, dpct::device_to_host);
    dpct::dpct_free(d_u);
    dpct::dpct_free(d_v);
    dpct::dpct_free(d_vp);
    dpct::dpct_free(d_phi);
    dpct::dpct_free(d_eta);
}
