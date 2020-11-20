#include "hip/hip_runtime.h"
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
__global__ void
testKernel(int *g_odata)
{
    // access thread id
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Atomic addition
    atomicAdd(&g_odata[0], 10);

    // Atomic subtraction (final should be 0)
    atomicSub(&g_odata[1], 10);

    // Atomic maximum
    atomicMax(&g_odata[2], tid);

    // Atomic minimum
    atomicMin(&g_odata[3], tid);

    // Atomic AND
    atomicAnd(&g_odata[4], 2*tid+7);

    // Atomic OR
    atomicOr(&g_odata[5], 1 << tid);

    // Atomic XOR
    atomicXor(&g_odata[6], tid);
}

#endif // #ifndef _SIMPLEATOMICS_KERNEL_H_
