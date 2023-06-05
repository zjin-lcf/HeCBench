/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifndef _SCAN_H_
#define _SCAN_H_

#include <sycl/sycl.hpp>

using uint4 = sycl::uint4;
using uint2 = sycl::uint2;

#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
#define MAX_LOCAL_GROUP_SIZE 256
static const int WORKGROUP_SIZE = 256;
static const uint   MAX_BATCH_ELEMENTS = 64 * 1048576;
static const uint MIN_SHORT_ARRAY_SIZE = 4;
static const uint MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
static const uint MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
static const uint MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

uint factorRadix2(uint& log2L, uint L);

void scanExclusiveLarge(
    sycl::queue &q,
    uint *d_Dst,
    uint *d_Src,
    uint *d_Buf,
    const uint batchSize,
    const uint arrayLength,
    const uint numElements);
#endif
