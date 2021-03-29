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
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(gws * lws, lws),
                     [=](sycl::nd_item<3> item_ct1) {
                       integrateSystemK(d_Pos, d_Vel, params, deltaTime,
                                        numParticles, item_ct1);
                     });
  });
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
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
          calcHashK(d_Hash, d_Index, d_Pos, params, numParticles, item_ct1);
        });
  });
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
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(gws * lws, lws),
                     [=](sycl::nd_item<3> item_ct1) {
                       memSetK(d_Data, val, N, item_ct1);
                     });
  });
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
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        dpct_local_acc_ct1(sycl::range<1>((wgSize + 1) * sizeof(unsigned int)),
                           cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
          findCellBoundsAndReorderK(d_CellStart, d_CellEnd, d_ReorderedPos,
                                    d_ReorderedVel, d_Hash, d_Index, d_Pos,
                                    d_Vel, numParticles, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
        });
  });
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
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(gws * lws, lws), [=](sycl::nd_item<3> item_ct1) {
          collideK(d_Vel, d_ReorderedPos, d_ReorderedVel, d_Index, d_CellStart,
                   d_CellEnd, params, numParticles, item_ct1);
        });
  });
}
