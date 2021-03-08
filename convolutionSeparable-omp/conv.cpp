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

#include "conv.h"
#include <omp.h>
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
    float* dst,
    const float* src,
    const float* kernel,
    const unsigned int imageW,
    const unsigned int imageH,
    const unsigned int pitch
)
{
    assert ( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert ( imageH % ROWS_BLOCKDIM_Y == 0 );

    int teamX = (imageW / ROWS_RESULT_STEPS) / ROWS_BLOCKDIM_X;
    int teamY = imageH / ROWS_BLOCKDIM_Y;
    int numTeams = teamX * teamY;

    #pragma omp target teams num_teams(numTeams) thread_limit(ROWS_BLOCKDIM_Y*ROWS_BLOCKDIM_X)
    {
      float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];
      #pragma omp parallel 
      {
        int gidX = omp_get_team_num() % teamX; 
        int gidY = omp_get_team_num() / teamX;  
        int lidX = omp_get_thread_num() % ROWS_BLOCKDIM_X;
        int lidY = omp_get_thread_num() / ROWS_BLOCKDIM_X;
        //Offset to the left halo edge
        const int baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
        const int baseY = gidY * ROWS_BLOCKDIM_Y + lidY;

        const float* src_new = src + baseY * pitch + baseX;
        float* dst_new = dst + baseY * pitch + baseX;

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
        #pragma omp barrier
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
            float sum = 0;

            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += kernel[KERNEL_RADIUS - j] * l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X + j];

            dst_new[i * ROWS_BLOCKDIM_X] = sum;
        }
      }
    }
}

void convolutionColumns(
    float* dst,
    const float* src,
    const float* kernel,
    const unsigned int imageW,
    const unsigned int imageH,
    const unsigned int pitch
){
    assert ( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    assert ( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert ( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    int teamX = imageW / COLUMNS_BLOCKDIM_X;
    int teamY = imageH / COLUMNS_RESULT_STEPS / COLUMNS_BLOCKDIM_Y;
    int numTeams = teamX * teamY;

    #pragma omp target teams num_teams(numTeams) thread_limit(COLUMNS_BLOCKDIM_Y*COLUMNS_BLOCKDIM_X)
    {
      float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
      #pragma omp parallel 
      {
        int gidX = omp_get_team_num() % teamX; 
        int gidY = omp_get_team_num() / teamX;  
        int lidX = omp_get_thread_num() % COLUMNS_BLOCKDIM_X;
        int lidY = omp_get_thread_num() / COLUMNS_BLOCKDIM_X;

        //Offset to the upper halo edge
        const int baseX = gidX * COLUMNS_BLOCKDIM_X + lidX;
        const int baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;

        const float* src_new = src + baseY * pitch + baseX;
        float* dst_new = dst + baseY * pitch + baseX;

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
        #pragma omp barrier
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
            float sum = 0;

            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += kernel[KERNEL_RADIUS - j] * l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y + j];

            dst_new[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
        }
      }
    }
}
