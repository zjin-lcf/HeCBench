#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

/* Simple kernel demonstrating atomic functions in device code. */

#ifndef _SIMPLEATOMICS_KERNEL_H_
#define _SIMPLEATOMICS_KERNEL_H_

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for atomic instructions
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
void
testKernel(int *g_odata, sycl::nd_item<3> item_ct1)
{
    // access thread id
    const unsigned int tid =
        item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
        item_ct1.get_local_id(2);

    // Atomic addition
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[0])).fetch_add(10);

    // Atomic subtraction (final should be 0)
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[1])).fetch_sub(10);

    // Atomic maximum
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[2])).fetch_max(tid);

    // Atomic minimum
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[3])).fetch_min(tid);

    // Atomic AND
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[4]))
        .fetch_and(2 * tid + 7);

    // Atomic OR
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[5])).fetch_or(1 << tid);

    // Atomic XOR
    sycl::atomic<int>(sycl::global_ptr<int>(&g_odata[6])).fetch_xor(tid);
}

#endif // #ifndef _SIMPLEATOMICS_KERNEL_H_
