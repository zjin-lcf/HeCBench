#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "constants.h"

#define N_THREADS_PER_BLOCK 256

void find_min_max_u_kernel(
    const float *__restrict__ g_u, float *__restrict__ g_max, float *__restrict__ g_min
,
    sycl::nd_item<3> item_ct1, uint8_t *dpct_local) {
    auto sdata = (float *)dpct_local;

    unsigned int tid = item_ct1.get_local_id(2);
    unsigned int tidFromBack = item_ct1.get_local_range().get(2) - 1 - tid;
    unsigned int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
    sdata[tid] = g_u[i];
    item_ct1.barrier();

    for (unsigned int s = item_ct1.get_local_range().get(2) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sdata[tid + s] > sdata[tid])
            {
                sdata[tid] = sdata[tid + s];
            }
        }
        if (tidFromBack < s) {
            if (sdata[tid - s] < sdata[tid]) {
                sdata[tid] = sdata[tid - s];
            }
        }
        item_ct1.barrier();
    }

    if (tid == 0)
    {
        g_max[item_ct1.get_group(2)] = sdata[0];
    }
    if (tidFromBack == 0)
    {
        g_min[item_ct1.get_group(2)] = sdata[tid];
    }
}

void find_min_max_u_cuda(
    const float *__restrict__ u, llint u_size, float *__restrict__ min_u, float *__restrict__ max_u
) {
    llint u_block = u_size / N_THREADS_PER_BLOCK;
    llint u_remainder = u_size % N_THREADS_PER_BLOCK;

    llint d_block = u_block;
    if (u_remainder != 0) { d_block += 1; }
    llint d_size = d_block * N_THREADS_PER_BLOCK;

    llint reminder_size = N_THREADS_PER_BLOCK - u_remainder;
    float *reminder = (float *)malloc(reminder_size * sizeof(float));
    memcpy(reminder, u, reminder_size * sizeof(float));

    float* max = (float*)malloc(d_block * sizeof(float));
    float *min = (float*)malloc(d_block * sizeof(float));

    float* d_u, * d_max, * d_min;
    dpct::dpct_malloc(&d_u, d_size * sizeof(float));
    dpct::dpct_malloc(&d_max, d_block * sizeof(float));
    dpct::dpct_malloc(&d_min, d_block * sizeof(float));

    dpct::dpct_memcpy(d_u, u, u_size * sizeof(float), dpct::host_to_device);
    dpct::dpct_memcpy(d_u + u_size, reminder, reminder_size * sizeof(float),
                      dpct::host_to_device);
    {
        dpct::buffer_t d_u_buf_ct0 = dpct::get_buffer(d_u);
        dpct::buffer_t d_max_buf_ct1 = dpct::get_buffer(d_max);
        dpct::buffer_t d_min_buf_ct2 = dpct::get_buffer(d_min);
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dpct_local_acc_ct1(
                    sycl::range<1>(sizeof(float) * N_THREADS_PER_BLOCK), cgh);
            auto d_u_acc_ct0 =
                d_u_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_max_acc_ct1 =
                d_max_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
            auto d_min_acc_ct2 =
                d_min_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, d_block) *
                                      sycl::range<3>(1, 1, N_THREADS_PER_BLOCK),
                                  sycl::range<3>(1, 1, N_THREADS_PER_BLOCK)),
                [=](sycl::nd_item<3> item_ct1) {
                    find_min_max_u_kernel((const float *)(&d_u_acc_ct0[0]),
                                          (float *)(&d_max_acc_ct1[0]),
                                          (float *)(&d_min_acc_ct2[0]),
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                });
        });
    }
    dpct::dpct_memcpy(max, d_max, d_block * sizeof(float),
                      dpct::device_to_host);
    dpct::dpct_memcpy(min, d_min, d_block * sizeof(float),
                      dpct::device_to_host);

    *min_u = FLT_MAX, *max_u = FLT_MIN;
    for (size_t i = 0; i < d_block; i++) {
        *min_u = fminf(*min_u, min[i]);
        *max_u = fmaxf(*max_u, max[i]);
    }

    dpct::dpct_free(d_max);
    dpct::dpct_free(d_min);
    dpct::dpct_free(d_u);
    free(reminder);
    free(max);
    free(min);
}
