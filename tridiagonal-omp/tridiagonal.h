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
 
 /*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifndef COMMON_H
#define COMMON_H

#define TRANSPOSE_BLOCK_DIM    16
#define BLOCK_DIM TRANSPOSE_BLOCK_DIM
#define REORDER
#define SYSTEM_SIZE  128

//#define OUTPUT_RESULTS
#ifdef DEBUG
#define BENCH_ITERATIONS  1
#else
#define BENCH_ITERATIONS  100
#endif

typedef struct { float x; float y; float z; float w; } float4 ;

#endif
