#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <string.h>
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

void find_min_max_u_cuda(const float *__restrict__ u, llint u_size,
                         float *__restrict__ min_u, float *__restrict__ max_u) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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
    d_u = (float *)sycl::malloc_device(d_size * sizeof(float), q_ct1);
    d_max = (float *)sycl::malloc_device(d_block * sizeof(float), q_ct1);
    d_min = (float *)sycl::malloc_device(d_block * sizeof(float), q_ct1);

    q_ct1.memcpy(d_u, u, u_size * sizeof(float)).wait();
    q_ct1.memcpy(d_u + u_size, reminder, reminder_size * sizeof(float)).wait();
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(
                sycl::range<1>(sizeof(float) * N_THREADS_PER_BLOCK), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, d_block) *
                                  sycl::range<3>(1, 1, N_THREADS_PER_BLOCK),
                              sycl::range<3>(1, 1, N_THREADS_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                find_min_max_u_kernel(d_u, d_max, d_min, item_ct1,
                                      dpct_local_acc_ct1.get_pointer());
            });
    });
    q_ct1.memcpy(max, d_max, d_block * sizeof(float)).wait();
    q_ct1.memcpy(min, d_min, d_block * sizeof(float)).wait();

    *min_u = FLT_MAX, *max_u = FLT_MIN;
    for (size_t i = 0; i < d_block; i++) {
        *min_u = fminf(*min_u, min[i]);
        *max_u = fmaxf(*max_u, max[i]);
    }

    sycl::free(d_max, q_ct1);
    sycl::free(d_min, q_ct1);
    sycl::free(d_u, q_ct1);
    free(reminder);
    free(max);
    free(min);
}
