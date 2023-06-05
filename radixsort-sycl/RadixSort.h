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
#ifndef _RADIXSORT_H_
#define _RADIXSORT_H_

#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include "Scan.h"

static const uint WARP_SIZE = 32;
static const uint bitStep = 4;
static const uint CTA_SIZE = 128;

void radixSortKeys(sycl::queue &q,
                   uint *d_keys,
                   uint *d_tempKeys,
                   uint *d_counters,
                   uint *d_blockOffsets,
                   uint *d_countersSum,
                   uint *d_buffer,
                   const uint numElements,
                   const uint keyBits,
                   const uint batchSize);

#endif
