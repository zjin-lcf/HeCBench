/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This example shows how to use the clock function to measure the performance of
// a kernel accurately.
//
// Blocks are executed in parallel and out of order. Since there's no synchronization
// mechanism between blocks, we measure the clock once for each block. The clock
// samples are written to device memory.

// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in device memory.
static void timedReduction(const float *input, float *output, uint64_t *timer,
                           float *shared, sycl::nd_item<1> &item)
{
    const int tid = item.get_local_id(0);
    const int bid = item.get_group(0);

    if (tid == 0) timer[bid] = syclex::clock<syclex::clock_scope::device>();

    // Copy input.
    shared[tid] = input[tid];
    shared[tid + item.get_local_range(0)] = input[tid + item.get_local_range(0)];

    // Perform reduction to find minimum.
    for (int d = item.get_local_range(0); d > 0; d /= 2)
    {
        item.barrier(sycl::access::fence_space::local_space);

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    item.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) timer[bid + item.get_group_range(0)] = syclex::clock<syclex::clock_scope::device>();
}


// This example shows how to use the clock function to measure the performance of
// a kernel accurately.
//
// Blocks are executed in parallel and out of order. Since there's no synchronization
// mechanism between blocks, we measure the clock once for each block. The clock
// samples are written to device memory.

#ifndef NUM_BLOCKS
#define NUM_BLOCKS    32
#endif

#define NUM_THREADS   256

// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.
//
// Here are some numbers I get on my G80:
//    blocks - clocks
//    1 - 3096
//    8 - 3232
//    16 - 3364
//    32 - 4615
//    64 - 9981
//
// With less than 16 blocks some of the multiprocessors of the device are idle. With
// more than 16 you are using all the multiprocessors, but there's only one block per
// multiprocessor and that doesn't allow you to hide the latency of the memory. With
// more than 32 the speed scales linearly.

int main(int argc, char **argv)
{
    printf("SYCL Clock sample\n");

    float *dinput = NULL;
    float *doutput = NULL;
    uint64_t *dtimer = NULL;

    uint64_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++)
    {
        input[i] = (float)i;
    }

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    dinput = sycl::malloc_device<float>(NUM_THREADS * 2, q);
    doutput = sycl::malloc_device<float>(NUM_BLOCKS, q);
    dtimer = sycl::malloc_device<uint64_t>(NUM_BLOCKS * 2, q);

    q.memcpy(dinput, input, sizeof(float) * NUM_THREADS * 2);

    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(2 * NUM_THREADS), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(NUM_BLOCKS * NUM_THREADS),
                              sycl::range<1>(NUM_THREADS)),
            [=](sycl::nd_item<1> item) {
                timedReduction(dinput, doutput, dtimer,
                               smem.get_multi_ptr<sycl::access::decorated::no>().get(), item);
            });
    });

    q.memcpy(timer, dtimer, sizeof(uint64_t) * NUM_BLOCKS * 2).wait();

    sycl::free(dinput, q);
    sycl::free(doutput, q);
    sycl::free(dtimer, q);

    long double totalBlockTime = 0;
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // printf("Block %d: clocks: %lu\n", i, timer[NUM_BLOCKS+i] - timer[i]);
        totalBlockTime += timer[NUM_BLOCKS+i] - timer[i];
    }

    // Compute the difference between the last block end and the first block start.
    uint64_t minStart = timer[0];
    uint64_t maxEnd = timer[NUM_BLOCKS];

    for (int i = 1; i < NUM_BLOCKS; i++)
    {
        minStart = timer[i] < minStart ? timer[i] : minStart;
        maxEnd = timer[NUM_BLOCKS+i] > maxEnd ? timer[NUM_BLOCKS+i] : maxEnd;
    }

    printf("Total clocks = %lu\n", (maxEnd - minStart));
    printf("Execution efficiency = %Lf\n", 100 * totalBlockTime / (long double)(maxEnd - minStart));

    return EXIT_SUCCESS;
}
