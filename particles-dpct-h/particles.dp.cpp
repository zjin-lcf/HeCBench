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

//Standard utilities and systems includes
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include "particles.h"
#include "particles_kernels.dp.cpp"

//Simulation parameters

static const size_t wgSize = 64;


static size_t uSnap(size_t a, size_t b){
  return ((a % b) == 0) ? a : (a - (a % b) + b);
}

void integrateSystem(sycl::float4 *d_Pos, sycl::float4 *d_Vel,
                     const simParams_t &params, const float deltaTime,
                     const unsigned int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);
  sycl::range<3> gws(1, 1, globalWorkSize / wgSize);
  sycl::range<3> lws(1, 1, wgSize);

  /*
  DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> d_Pos_buf_ct0 =
        dpct::get_buffer_and_offset(d_Pos);
    size_t d_Pos_offset_ct0 = d_Pos_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_Vel_buf_ct1 =
        dpct::get_buffer_and_offset(d_Vel);
    size_t d_Vel_offset_ct1 = d_Vel_buf_ct1.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto d_Pos_acc_ct0 =
          d_Pos_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Vel_acc_ct1 =
          d_Vel_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
            sycl::float4 *d_Pos_ct0 =
                (sycl::float4 *)(&d_Pos_acc_ct0[0] + d_Pos_offset_ct0);
            sycl::float4 *d_Vel_ct1 =
                (sycl::float4 *)(&d_Vel_acc_ct1[0] + d_Vel_offset_ct1);
            integrateSystemK(d_Pos_ct0, d_Vel_ct1, params, deltaTime,
                             numParticles, item_ct1);
          });
    });
  }
}

void calcHash(unsigned int *d_Hash, unsigned int *d_Index, sycl::float4 *d_Pos,
              const simParams_t &params, const int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);
  sycl::range<3> gws(1, 1, globalWorkSize / wgSize);
  sycl::range<3> lws(1, 1, wgSize);

  /*
  DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> d_Hash_buf_ct0 =
        dpct::get_buffer_and_offset(d_Hash);
    size_t d_Hash_offset_ct0 = d_Hash_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_Index_buf_ct1 =
        dpct::get_buffer_and_offset(d_Index);
    size_t d_Index_offset_ct1 = d_Index_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_Pos_buf_ct2 =
        dpct::get_buffer_and_offset(d_Pos);
    size_t d_Pos_offset_ct2 = d_Pos_buf_ct2.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto d_Hash_acc_ct0 =
          d_Hash_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Index_acc_ct1 =
          d_Index_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Pos_acc_ct2 =
          d_Pos_buf_ct2.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
            unsigned int *d_Hash_ct0 =
                (unsigned int *)(&d_Hash_acc_ct0[0] + d_Hash_offset_ct0);
            unsigned int *d_Index_ct1 =
                (unsigned int *)(&d_Index_acc_ct1[0] + d_Index_offset_ct1);
            const sycl::float4 *d_Pos_ct2 =
                (const sycl::float4 *)(&d_Pos_acc_ct2[0] + d_Pos_offset_ct2);
            calcHashK(d_Hash_ct0, d_Index_ct1, d_Pos_ct2, params, numParticles,
                      item_ct1);
          });
    });
  }
}

void memSet(
    unsigned int* d_Data,
    unsigned int val,
    unsigned int N)
{
  size_t globalWorkSize = uSnap(N, wgSize);

  sycl::range<3> gws(1, 1, globalWorkSize / wgSize);
  sycl::range<3> lws(1, 1, wgSize);

  /*
  DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> d_Data_buf_ct0 =
        dpct::get_buffer_and_offset(d_Data);
    size_t d_Data_offset_ct0 = d_Data_buf_ct0.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto d_Data_acc_ct0 =
          d_Data_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
            unsigned int *d_Data_ct0 =
                (unsigned int *)(&d_Data_acc_ct0[0] + d_Data_offset_ct0);
            memSetK(d_Data_ct0, val, N, item_ct1);
          });
    });
  }
}

void findCellBoundsAndReorder(unsigned int *d_CellStart,
                              unsigned int *d_CellEnd,
                              sycl::float4 *d_ReorderedPos,
                              sycl::float4 *d_ReorderedVel,
                              unsigned int *d_Hash, unsigned int *d_Index,
                              sycl::float4 *d_Pos, sycl::float4 *d_Vel,
                              const unsigned int numParticles,
                              const unsigned int numCells)
{
  memSet(d_CellStart, 0xFFFFFFFFU, numCells);

  size_t globalWorkSize = uSnap(numParticles, wgSize);
  sycl::range<3> gws(1, 1, globalWorkSize / wgSize);
  sycl::range<3> lws(1, 1, wgSize);

  /*
  DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> d_CellStart_buf_ct0 =
        dpct::get_buffer_and_offset(d_CellStart);
    size_t d_CellStart_offset_ct0 = d_CellStart_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_CellEnd_buf_ct1 =
        dpct::get_buffer_and_offset(d_CellEnd);
    size_t d_CellEnd_offset_ct1 = d_CellEnd_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_ReorderedPos_buf_ct2 =
        dpct::get_buffer_and_offset(d_ReorderedPos);
    size_t d_ReorderedPos_offset_ct2 = d_ReorderedPos_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_ReorderedVel_buf_ct3 =
        dpct::get_buffer_and_offset(d_ReorderedVel);
    size_t d_ReorderedVel_offset_ct3 = d_ReorderedVel_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_Hash_buf_ct4 =
        dpct::get_buffer_and_offset(d_Hash);
    size_t d_Hash_offset_ct4 = d_Hash_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_Index_buf_ct5 =
        dpct::get_buffer_and_offset(d_Index);
    size_t d_Index_offset_ct5 = d_Index_buf_ct5.second;
    std::pair<dpct::buffer_t, size_t> d_Pos_buf_ct6 =
        dpct::get_buffer_and_offset(d_Pos);
    size_t d_Pos_offset_ct6 = d_Pos_buf_ct6.second;
    std::pair<dpct::buffer_t, size_t> d_Vel_buf_ct7 =
        dpct::get_buffer_and_offset(d_Vel);
    size_t d_Vel_offset_ct7 = d_Vel_buf_ct7.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          dpct_local_acc_ct1(
              sycl::range<1>((wgSize + 1) * sizeof(unsigned int)), cgh);
      auto d_CellStart_acc_ct0 =
          d_CellStart_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_CellEnd_acc_ct1 =
          d_CellEnd_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_ReorderedPos_acc_ct2 =
          d_ReorderedPos_buf_ct2.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_ReorderedVel_acc_ct3 =
          d_ReorderedVel_buf_ct3.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_Hash_acc_ct4 =
          d_Hash_buf_ct4.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Index_acc_ct5 =
          d_Index_buf_ct5.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Pos_acc_ct6 =
          d_Pos_buf_ct6.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_Vel_acc_ct7 =
          d_Vel_buf_ct7.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
            unsigned int *d_CellStart_ct0 =
                (unsigned int *)(&d_CellStart_acc_ct0[0] +
                                 d_CellStart_offset_ct0);
            unsigned int *d_CellEnd_ct1 =
                (unsigned int *)(&d_CellEnd_acc_ct1[0] + d_CellEnd_offset_ct1);
            sycl::float4 *d_ReorderedPos_ct2 =
                (sycl::float4 *)(&d_ReorderedPos_acc_ct2[0] +
                                 d_ReorderedPos_offset_ct2);
            sycl::float4 *d_ReorderedVel_ct3 =
                (sycl::float4 *)(&d_ReorderedVel_acc_ct3[0] +
                                 d_ReorderedVel_offset_ct3);
            const unsigned int *d_Hash_ct4 =
                (const unsigned int *)(&d_Hash_acc_ct4[0] + d_Hash_offset_ct4);
            const unsigned int *d_Index_ct5 =
                (const unsigned int *)(&d_Index_acc_ct5[0] +
                                       d_Index_offset_ct5);
            const sycl::float4 *d_Pos_ct6 =
                (const sycl::float4 *)(&d_Pos_acc_ct6[0] + d_Pos_offset_ct6);
            const sycl::float4 *d_Vel_ct7 =
                (const sycl::float4 *)(&d_Vel_acc_ct7[0] + d_Vel_offset_ct7);
            findCellBoundsAndReorderK(d_CellStart_ct0, d_CellEnd_ct1,
                                      d_ReorderedPos_ct2, d_ReorderedVel_ct3,
                                      d_Hash_ct4, d_Index_ct5, d_Pos_ct6,
                                      d_Vel_ct7, numParticles, item_ct1,
                                      dpct_local_acc_ct1.get_pointer());
          });
    });
  }
}

void collide(sycl::float4 *d_Vel, sycl::float4 *d_ReorderedPos,
             sycl::float4 *d_ReorderedVel, unsigned int *d_Index,
             unsigned int *d_CellStart, unsigned int *d_CellEnd,
             const simParams_t &params, const unsigned int numParticles,
             const unsigned int numCells)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);

  sycl::range<3> gws(1, 1, globalWorkSize / wgSize);
  sycl::range<3> lws(1, 1, wgSize);

  /*
  DPCT1049:8: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> d_Vel_buf_ct0 =
        dpct::get_buffer_and_offset(d_Vel);
    size_t d_Vel_offset_ct0 = d_Vel_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_ReorderedPos_buf_ct1 =
        dpct::get_buffer_and_offset(d_ReorderedPos);
    size_t d_ReorderedPos_offset_ct1 = d_ReorderedPos_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> d_ReorderedVel_buf_ct2 =
        dpct::get_buffer_and_offset(d_ReorderedVel);
    size_t d_ReorderedVel_offset_ct2 = d_ReorderedVel_buf_ct2.second;
    std::pair<dpct::buffer_t, size_t> d_Index_buf_ct3 =
        dpct::get_buffer_and_offset(d_Index);
    size_t d_Index_offset_ct3 = d_Index_buf_ct3.second;
    std::pair<dpct::buffer_t, size_t> d_CellStart_buf_ct4 =
        dpct::get_buffer_and_offset(d_CellStart);
    size_t d_CellStart_offset_ct4 = d_CellStart_buf_ct4.second;
    std::pair<dpct::buffer_t, size_t> d_CellEnd_buf_ct5 =
        dpct::get_buffer_and_offset(d_CellEnd);
    size_t d_CellEnd_offset_ct5 = d_CellEnd_buf_ct5.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto d_Vel_acc_ct0 =
          d_Vel_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_ReorderedPos_acc_ct1 =
          d_ReorderedPos_buf_ct1.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_ReorderedVel_acc_ct2 =
          d_ReorderedVel_buf_ct2.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_Index_acc_ct3 =
          d_Index_buf_ct3.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_CellStart_acc_ct4 =
          d_CellStart_buf_ct4.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_CellEnd_acc_ct5 =
          d_CellEnd_buf_ct5.first.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
            sycl::float4 *d_Vel_ct0 =
                (sycl::float4 *)(&d_Vel_acc_ct0[0] + d_Vel_offset_ct0);
            const sycl::float4 *d_ReorderedPos_ct1 =
                (const sycl::float4 *)(&d_ReorderedPos_acc_ct1[0] +
                                       d_ReorderedPos_offset_ct1);
            const sycl::float4 *d_ReorderedVel_ct2 =
                (const sycl::float4 *)(&d_ReorderedVel_acc_ct2[0] +
                                       d_ReorderedVel_offset_ct2);
            const unsigned int *d_Index_ct3 =
                (const unsigned int *)(&d_Index_acc_ct3[0] +
                                       d_Index_offset_ct3);
            const unsigned int *d_CellStart_ct4 =
                (const unsigned int *)(&d_CellStart_acc_ct4[0] +
                                       d_CellStart_offset_ct4);
            const unsigned int *d_CellEnd_ct5 =
                (const unsigned int *)(&d_CellEnd_acc_ct5[0] +
                                       d_CellEnd_offset_ct5);
            collideK(d_Vel_ct0, d_ReorderedPos_ct1, d_ReorderedVel_ct2,
                     d_Index_ct3, d_CellStart_ct4, d_CellEnd_ct5, params,
                     numParticles, item_ct1);
          });
    });
  }
}
