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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA

// Includes, kernels
#include "simpleAtomicIntrinsics_kernel.hpp"

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
int
computeGold(int *gpuData, const int len)
{
    int val = 0;

    for (int i = 0; i < len; ++i)
    {
        val += 10;
    }

    if (val != gpuData[0])
    {
        printf("Add failed %d %d\n", val, gpuData[0]);
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val -= 10;
    }

    if (val != gpuData[1])
    {
        printf("Sub failed: %d %d\n", val, gpuData[1]);
    }

    val = -(1<<8);

    for (int i = 0; i < len; ++i)
    {
        val = max(val, i);
    }

    if (val != gpuData[2])
    {
        printf("Max failed: %d %d\n", val, gpuData[2]);
    }

    val = 1 << 8;

    for (int i = 0; i < len; ++i)
    {
        val = min(val, i);
    }

    if (val != gpuData[3])
    {
        printf("Min failed: %d %d\n", val, gpuData[3]);
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        val &= (2 * i + 7);
    }

    if (val != gpuData[4])
    {
        printf("And failed: %d %d\n", val, gpuData[4]);
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val |= (1 << i);
    }

    if (val != gpuData[5])
    {
        printf("Or failed: %d %d\n", val, gpuData[5]);
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        val ^= i;
    }

    if (val != gpuData[6])
    {
        printf("Xor failed %d %d\n", val, gpuData[6]);
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    unsigned int len = 1 << 27;
    unsigned int numThreads = 256;
    unsigned int numBlocks = (len + numThreads - 1) / numThreads;
    unsigned int numData = 7;
    unsigned int memSize = sizeof(int) * numData;
    int gpuData[] = {0, 0, -(1<<8), 1<<8, 0xff, 0, 0xff};

    // allocate device memory for result
    int *dOData;
    dpct::dpct_malloc((void **)&dOData, memSize);

    for (int i = 0; i < 1; i++) {
      // copy host memory to device to initialize to zero
        dpct::dpct_memcpy(dOData, gpuData, memSize, dpct::host_to_device);

      // execute the kernel
        {
            dpct::buffer_t dOData_buf_ct0 = dpct::get_buffer(dOData);
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                auto dOData_acc_ct0 =
                    dOData_buf_ct0.get_access<sycl::access::mode::read_write>(
                        cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                          sycl::range<3>(1, 1, numThreads),
                                      sycl::range<3>(1, 1, numThreads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        testKernel((int *)(&dOData_acc_ct0[0]), item_ct1);
                    });
            });
        }
    }

    //Copy result from device to host
    dpct::dpct_memcpy(gpuData, dOData, memSize, dpct::device_to_host);

    computeGold(gpuData, numThreads * numBlocks);

    dpct::dpct_free(dOData);
    return 0;
}
