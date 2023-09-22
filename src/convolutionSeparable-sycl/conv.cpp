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

#include <assert.h>
#include <sycl/sycl.hpp>
#include "conv.h"

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

#define ROWS_BLOCKDIM_X       16
#define COLUMNS_BLOCKDIM_X    16
#define ROWS_BLOCKDIM_Y       4
#define COLUMNS_BLOCKDIM_Y    8
#define ROWS_RESULT_STEPS     8
#define COLUMNS_RESULT_STEPS  8
#define ROWS_HALO_STEPS       1
#define COLUMNS_HALO_STEPS    1

void convolutionRows(
    sycl::queue &q,
    float *dst,
    float *src,
    float *kernel,
    const int imageW,
    const int imageH,
    const int pitch
)
{
    assert ( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert ( imageH % ROWS_BLOCKDIM_Y == 0 );

    sycl::range<2> lws (ROWS_BLOCKDIM_Y, ROWS_BLOCKDIM_X);
    sycl::range<2> gws (imageH, imageW / ROWS_RESULT_STEPS);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 2>
        l_Data(sycl::range<2>{ROWS_BLOCKDIM_Y, (ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X}, cgh);

      cgh.parallel_for<class conv_rows>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        int gidX = item.get_group(1); 
        int gidY = item.get_group(0); 
        int lidX = item.get_local_id(1); 
        int lidY = item.get_local_id(0); 
        //Offset to the left halo edge
        const int baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
        const int baseY = gidY * ROWS_BLOCKDIM_Y + lidY;

        const float* src_new = src + baseY * pitch + baseX;
        float* dst_new = dst + baseY * pitch + baseX;

        //Load main data
        #pragma unroll
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
            l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = ldg(&src_new[i * ROWS_BLOCKDIM_X]);

        //Load left halo
        #pragma unroll
        for(int i = 0; i < ROWS_HALO_STEPS; i++)
            l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? ldg(&src_new[i * ROWS_BLOCKDIM_X]) : 0;

        //Load right halo
        #pragma unroll
        for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
            l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? ldg(&src_new[i * ROWS_BLOCKDIM_X]) : 0;

        //Compute and store results
        item.barrier(sycl::access::fence_space::local_space);

        #pragma unroll
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
            float sum = 0;

            #pragma unroll
            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += ldg(&kernel[KERNEL_RADIUS - j]) * l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X + j];

            dst_new[i * ROWS_BLOCKDIM_X] = sum;
        }
      });
    });
}

void convolutionColumns(
    sycl::queue &q,
    float *dst,
    float *src,
    float *kernel,
    const int imageW,
    const int imageH,
    const int pitch
)
{
    assert ( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert ( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    sycl::range<2> lws (COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_X);
    sycl::range<2> gws (imageH / COLUMNS_RESULT_STEPS, imageW);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 2>
        l_Data(sycl::range<2>{COLUMNS_BLOCKDIM_X, (COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1}, cgh);

      cgh.parallel_for<class conv_cols>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {

        int gidX = item.get_group(1); 
        int gidY = item.get_group(0); 
        int lidX = item.get_local_id(1); 
        int lidY = item.get_local_id(0); 

        //Offset to the upper halo edge
        const int baseX = gidX * COLUMNS_BLOCKDIM_X + lidX;
        const int baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;

        const float* src_new = src + baseY * pitch + baseX;
        float* dst_new = dst + baseY * pitch + baseX;

        //Load main data
        #pragma unroll
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = ldg(&src_new[i * COLUMNS_BLOCKDIM_Y * pitch]);

        //Load upper halo
        #pragma unroll
        for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? ldg(&src_new[i * COLUMNS_BLOCKDIM_Y * pitch]) : 0;

        //Load lower halo
        #pragma unroll
        for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? ldg(&src_new[i * COLUMNS_BLOCKDIM_Y * pitch]) : 0;

        //Compute and store results
        item.barrier(sycl::access::fence_space::local_space);

        #pragma unroll
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
            float sum = 0;

            #pragma unroll
            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += ldg(&kernel[KERNEL_RADIUS - j]) * l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y + j];

            dst_new[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
        }
      });
    });
}
