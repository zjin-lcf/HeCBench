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

#include "common.h"
#include "conv.h"
#include <cassert>

#define ROWS_BLOCKDIM_X       16
#define COLUMNS_BLOCKDIM_X    16
#define ROWS_BLOCKDIM_Y       4
#define COLUMNS_BLOCKDIM_Y    8
#define ROWS_RESULT_STEPS     8
#define COLUMNS_RESULT_STEPS  8
#define ROWS_HALO_STEPS       1
#define COLUMNS_HALO_STEPS    1


void convolutionRows(
    queue &q,
    buffer<float,1> &d_Dst,
    buffer<float,1> &d_Src,
    buffer<float,1> &d_Kernel,
    const unsigned int imageW,
    const unsigned int imageH,
    const unsigned int pitch
)
{
    assert ( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert ( imageH % ROWS_BLOCKDIM_Y == 0 );

    range<2> lws (ROWS_BLOCKDIM_Y, ROWS_BLOCKDIM_X);
    range<2> gws (imageH, imageW / ROWS_RESULT_STEPS);

    q.submit([&] (handler &cgh) {
      auto dst = d_Dst.get_access<sycl_write>(cgh);
      auto src = d_Src.get_access<sycl_read>(cgh);
      auto kernel = d_Kernel.get_access<sycl_read>(cgh);
      accessor<float, 2, sycl_read_write, access::target::local> 
        l_Data({ROWS_BLOCKDIM_Y, (ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X}, cgh);

      cgh.parallel_for<class conv_rows>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        int gidX = item.get_group(1); 
        int gidY = item.get_group(0); 
        int lidX = item.get_local_id(1); 
        int lidY = item.get_local_id(0); 
        //Offset to the left halo edge
        const int baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
        const int baseY = gidY * ROWS_BLOCKDIM_Y + lidY;

        const float* src_new = src.get_pointer() + baseY * pitch + baseX;
        float* dst_new = dst.get_pointer() + baseY * pitch + baseX;

        //Load main data
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
            l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = src_new[i * ROWS_BLOCKDIM_X];

        //Load left halo
        for(int i = 0; i < ROWS_HALO_STEPS; i++)
            l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? src_new[i * ROWS_BLOCKDIM_X] : 0;

        //Load right halo
        for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
            l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? src_new[i * ROWS_BLOCKDIM_X] : 0;

        //Compute and store results
        item.barrier(access::fence_space::local_space);
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
            float sum = 0;

            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += kernel[KERNEL_RADIUS - j] * l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X + j];

            dst_new[i * ROWS_BLOCKDIM_X] = sum;
        }
      });
    });
}

void convolutionColumns(
    queue &q,
    buffer<float,1> &d_Dst,
    buffer<float,1> &d_Src,
    buffer<float,1> &d_Kernel,
    const unsigned int imageW,
    const unsigned int imageH,
    const unsigned int pitch
)
{
    assert ( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert ( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    range<2> lws (COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_X);
    range<2> gws (imageH / COLUMNS_RESULT_STEPS, imageW);

    q.submit([&] (handler &cgh) {
      auto dst = d_Dst.get_access<sycl_discard_write>(cgh);
      auto src = d_Src.get_access<sycl_read>(cgh);
      auto kernel = d_Kernel.get_access<sycl_read>(cgh);
      accessor<float, 2, sycl_read_write, access::target::local> 
        l_Data({COLUMNS_BLOCKDIM_X, (COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1}, cgh);

      cgh.parallel_for<class conv_cols>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {

        int gidX = item.get_group(1); 
        int gidY = item.get_group(0); 
        int lidX = item.get_local_id(1); 
        int lidY = item.get_local_id(0); 

        //Offset to the upper halo edge
        const int baseX = gidX * COLUMNS_BLOCKDIM_X + lidX;
        const int baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;

        const float* src_new = src.get_pointer() + baseY * pitch + baseX;
        float* dst_new = dst.get_pointer() + baseY * pitch + baseX;

        //Load main data
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = src_new[i * COLUMNS_BLOCKDIM_Y * pitch];

        //Load upper halo
        for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? src_new[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

        //Load lower halo
        for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y]  = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? src_new[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

        //Compute and store results
        item.barrier(access::fence_space::local_space);
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
            float sum = 0;

            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += kernel[KERNEL_RADIUS - j] * l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y + j];

            dst_new[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
        }
      });
    });
}
