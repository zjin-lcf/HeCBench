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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "DCT8x8.h"

inline void DCT8(float *D){
    float X07P = D[0] + D[7];
    float X16P = D[1] + D[6];
    float X25P = D[2] + D[5];
    float X34P = D[3] + D[4];

    float X07M = D[0] - D[7];
    float X61M = D[6] - D[1];
    float X25M = D[2] - D[5];
    float X43M = D[4] - D[3];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    D[0] = C_norm * (X07P34PP + X16P25PP);
    D[2] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    D[4] = C_norm * (X07P34PP - X16P25PP);
    D[6] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    D[1] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    D[3] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    D[5] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    D[7] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}


inline void IDCT8(float *D){
    float Y04P   = D[0] + D[4];
    float Y2b6eP = C_b * D[2] + C_e * D[6];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * D[7] + C_a * D[1] + C_c * D[3] + C_d * D[5];
    float Y7a1fM3d5cMP = C_a * D[7] - C_f * D[1] + C_d * D[3] - C_c * D[5];

    float Y04M   = D[0] - D[4];
    float Y2e6bM = C_e * D[2] - C_b * D[6];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * D[1] - C_d * D[7] - C_f * D[3] - C_a * D[5];
    float Y1d7cP3a5fMM = C_d * D[1] + C_c * D[7] - C_a * D[3] + C_f * D[5];

    D[0] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    D[7] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    D[4] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    D[3] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    D[1] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    D[5] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    D[2] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    D[6] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}



////////////////////////////////////////////////////////////////////////////////
// 8x8 DCT kernels
////////////////////////////////////////////////////////////////////////////////

void DCT8x8_kernel(
    float* d_Dst,
    const float* d_Src,
    const unsigned int stride,
    const unsigned int imageH,
    const unsigned int imageW
,
    sycl::nd_item<3> item_ct1,
    float *l_Transpose){
    const unsigned int localX = item_ct1.get_local_id(2); // get_local_id(0);
    const unsigned int localY =
        BLOCK_SIZE * item_ct1.get_local_id(1); // get_local_id(1);
    const unsigned int modLocalX = localX & (BLOCK_SIZE - 1);
    const unsigned int globalX = item_ct1.get_group(2) * BLOCK_X +
                                 localX; // get_group_id(0) * BLOCK_X + localX;
    const unsigned int globalY = item_ct1.get_group(1) * BLOCK_Y +
                                 localY; // get_group_id(1) * BLOCK_Y + localY;

    //Process only full blocks
    if( (globalX - modLocalX + BLOCK_SIZE - 1 >= imageW) || (globalY + BLOCK_SIZE - 1 >= imageH) )
        return;

    float *l_V = &l_Transpose[localY * (BLOCK_X+1) + localX];
    float *l_H = &l_Transpose[(localY + modLocalX) * (BLOCK_X+1) + localX - modLocalX];
    d_Src += globalY * stride + globalX;
    d_Dst += globalY * stride + globalX;

    float D[8];
    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        l_V[i * (BLOCK_X + 1)] = d_Src[i * stride];

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_H[i];
    DCT8(D);
    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        l_H[i] = D[i];

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_V[i * (BLOCK_X + 1)];
    DCT8(D);

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        d_Dst[i * stride] = D[i];
}

void IDCT8x8_kernel(
    float* d_Dst,
    const float* d_Src,
    const unsigned int stride,
    const unsigned int imageH,
    const unsigned int imageW
,
    sycl::nd_item<3> item_ct1,
    float *l_Transpose){
    const unsigned int localX = item_ct1.get_local_id(2); // get_local_id(0);
    const unsigned int localY =
        BLOCK_SIZE * item_ct1.get_local_id(1); // get_local_id(1);
    const unsigned int modLocalX = localX & (BLOCK_SIZE - 1);
    const unsigned int globalX = item_ct1.get_group(2) * BLOCK_X +
                                 localX; // get_group_id(0) * BLOCK_X + localX;
    const unsigned int globalY = item_ct1.get_group(1) * BLOCK_Y +
                                 localY; // get_group_id(1) * BLOCK_Y + localY;

    //Process only full blocks
    if( (globalX - modLocalX + BLOCK_SIZE - 1 >= imageW) || (globalY + BLOCK_SIZE - 1 >= imageH) )
        return;

    float *l_V = &l_Transpose[localY * (BLOCK_X+1) + localX];
    float *l_H = &l_Transpose[(localY + modLocalX) * (BLOCK_X+1) + localX - modLocalX];
    d_Src += globalY * stride + globalX;
    d_Dst += globalY * stride + globalX;

    float D[8];
    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        l_V[i * (BLOCK_X + 1)] = d_Src[i * stride];

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_H[i];
    IDCT8(D);
    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        l_H[i] = D[i];

    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_V[i * (BLOCK_X + 1)];
    IDCT8(D);
    for(unsigned int i = 0; i < BLOCK_SIZE; i++)
        d_Dst[i * stride] = D[i];
}

inline unsigned int iDivUp(unsigned int dividend, unsigned int divisor){
    return dividend / divisor + (dividend % divisor != 0);
}

void DCT8x8(float *d_Dst, const float *d_Src, unsigned int stride,
            unsigned int imageH, unsigned int imageW, int dir) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    size_t blockSize[2];
    size_t gridSize[2];
    blockSize[0] = BLOCK_X;
    blockSize[1] = BLOCK_Y / BLOCK_SIZE;
    gridSize[0] = iDivUp(imageW, BLOCK_X);
    gridSize[1] = iDivUp(imageH, BLOCK_Y);

    sycl::range<3> grid(1, gridSize[1], gridSize[0]);
    sycl::range<3> block(1, blockSize[1], blockSize[0]);

    if (dir == DCT_FORWARD)  {
      /*
      DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      std::pair<dpct::buffer_t, size_t> d_Dst_buf_ct0 =
          dpct::get_buffer_and_offset(d_Dst);
      size_t d_Dst_offset_ct0 = d_Dst_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> d_Src_buf_ct1 =
          dpct::get_buffer_and_offset(d_Src);
      size_t d_Src_offset_ct1 = d_Src_buf_ct1.second;
      q_ct1.submit([&](sycl::handler &cgh) {
         sycl::accessor<float, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
             l_Transpose_acc_ct1(sycl::range<1>(528 /*BLOCK_Y * (BLOCK_X+1)*/),
                                 cgh);
         auto d_Dst_acc_ct0 =
             d_Dst_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                 cgh);
         auto d_Src_acc_ct1 =
             d_Src_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                 cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(grid * block, block),
             [=](sycl::nd_item<3> item_ct1) {
                float *d_Dst_ct0 =
                    (float *)(&d_Dst_acc_ct0[0] + d_Dst_offset_ct0);
                const float *d_Src_ct1 =
                    (const float *)(&d_Src_acc_ct1[0] + d_Src_offset_ct1);
                DCT8x8_kernel(d_Dst_ct0, d_Src_ct1, stride, imageH, imageW,
                              item_ct1, l_Transpose_acc_ct1.get_pointer());
             });
      });
    }
    else {
      /*
      DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      std::pair<dpct::buffer_t, size_t> d_Dst_buf_ct0 =
          dpct::get_buffer_and_offset(d_Dst);
      size_t d_Dst_offset_ct0 = d_Dst_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> d_Src_buf_ct1 =
          dpct::get_buffer_and_offset(d_Src);
      size_t d_Src_offset_ct1 = d_Src_buf_ct1.second;
      q_ct1.submit([&](sycl::handler &cgh) {
         sycl::accessor<float, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
             l_Transpose_acc_ct1(sycl::range<1>(528 /*BLOCK_Y * (BLOCK_X+1)*/),
                                 cgh);
         auto d_Dst_acc_ct0 =
             d_Dst_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                 cgh);
         auto d_Src_acc_ct1 =
             d_Src_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                 cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(grid * block, block),
             [=](sycl::nd_item<3> item_ct1) {
                float *d_Dst_ct0 =
                    (float *)(&d_Dst_acc_ct0[0] + d_Dst_offset_ct0);
                const float *d_Src_ct1 =
                    (const float *)(&d_Src_acc_ct1[0] + d_Src_offset_ct1);
                IDCT8x8_kernel(d_Dst_ct0, d_Src_ct1, stride, imageH, imageW,
                               item_ct1, l_Transpose_acc_ct1.get_pointer());
             });
      });
    }
}
