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
#include "common.h"
#include "particles.h"
#include "particles_kernels.cpp"

//Simulation parameters

static const size_t wgSize = 64;

static size_t uSnap(size_t a, size_t b){
    return ((a % b) == 0) ? a : (a - (a % b) + b);
}

void integrateSystem(
    queue &q,
    buffer<sycl::float4,1> &d_Pos,
    buffer<sycl::float4,1> &d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles
){
    size_t globalWorkSize = uSnap(numParticles, wgSize);
    range<1> gws (globalWorkSize);
    range<1> lws (wgSize);

    q.submit([&] (handler &cgh) {
      auto pos = d_Pos.get_access<sycl_read_write>(cgh);
      auto vel = d_Vel.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class Integrate>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        integrateSystemK(item, pos.get_pointer(), vel.get_pointer(), 
                         params, deltaTime, numParticles);
      });
    });
}

void calcHash(
    queue &q,
    buffer<unsigned int, 1> &d_Hash,
    buffer<unsigned int, 1> &d_Index,
    buffer<sycl::float4, 1> &d_Pos,
    const simParams_t &params,
    const int numParticles
){
    size_t globalWorkSize = uSnap(numParticles, wgSize);
    range<1> gws (globalWorkSize);
    range<1> lws (wgSize);

    q.submit([&] (handler &cgh) {
      auto pos = d_Pos.get_access<sycl_read>(cgh);
      auto hash = d_Hash.get_access<sycl_discard_write>(cgh);
      auto index = d_Index.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class CalcHash>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        calcHashK(item, hash.get_pointer(), index.get_pointer(), 
                  pos.get_pointer(), params, numParticles);
      });
    });
}

void memSet(
    queue &q,
    buffer<unsigned int, 1> &d_Data,
    unsigned int val,
    unsigned int N
){
    size_t globalWorkSize = uSnap(N, wgSize);

    range<1> gws (globalWorkSize);
    range<1> lws (wgSize);

    q.submit([&] (handler &cgh) {
      auto data = d_Data.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class Memset>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        memSetK(item, data.get_pointer(), val, N);
      });
    });
}

void findCellBoundsAndReorder(
    queue &q,
    buffer<unsigned int, 1> &d_CellStart,
    buffer<unsigned int, 1> &d_CellEnd,
    buffer<sycl::float4, 1> &d_ReorderedPos,
    buffer<sycl::float4, 1> &d_ReorderedVel,
    buffer<unsigned int, 1> &d_Hash,
    buffer<unsigned int, 1> &d_Index,
    buffer<sycl::float4, 1> &d_Pos,
    buffer<sycl::float4, 1> &d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells
){
    memSet(q, d_CellStart, 0xFFFFFFFFU, numCells);
    
    size_t globalWorkSize = uSnap(numParticles, wgSize);
    range<1> gws (globalWorkSize);
    range<1> lws (wgSize);

    q.submit([&] (handler &cgh) {
      auto pos = d_Pos.get_access<sycl_read>(cgh);
      auto rpos = d_ReorderedPos.get_access<sycl_discard_write>(cgh);
      auto rvel = d_ReorderedVel.get_access<sycl_discard_write>(cgh);
      auto cellstart = d_CellStart.get_access<sycl_discard_write>(cgh);
      auto cellend = d_CellEnd.get_access<sycl_discard_write>(cgh);
      auto index = d_Index.get_access<sycl_read>(cgh);
      auto hash = d_Hash.get_access<sycl_read>(cgh);
      auto vel = d_Vel.get_access<sycl_read>(cgh);
      accessor<unsigned int, 1, sycl_read_write, access::target::local> localHash (wgSize + 1, cgh);
      cgh.parallel_for<class FindCellBoundsAndReorder>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        findCellBoundsAndReorderK(item, 
                                 cellstart.get_pointer(),
                                 cellend.get_pointer(), 
                                 rpos.get_pointer(),
                                 rvel.get_pointer(), 
                                 hash.get_pointer(), 
                                 index.get_pointer(), 
                                 pos.get_pointer(), 
                                 vel.get_pointer(),
                                 localHash.get_pointer(),
                                 numParticles);
      });
    });
}

void collide(
    queue &q,
    buffer<sycl::float4, 1> &d_Vel,
    buffer<sycl::float4, 1> &d_ReorderedPos,
    buffer<sycl::float4, 1> &d_ReorderedVel,
    buffer<unsigned int, 1> &d_Index,
    buffer<unsigned int, 1> &d_CellStart,
    buffer<unsigned int, 1> &d_CellEnd,
    const simParams_t &params,
    const unsigned int   numParticles,
    const unsigned int   numCells
){
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    range<1> gws (globalWorkSize);
    range<1> lws (wgSize);

    q.submit([&] (handler &cgh) {
      auto vel = d_Vel.get_access<sycl_discard_write>(cgh);
      auto rpos = d_ReorderedPos.get_access<sycl_read>(cgh);
      auto rvel = d_ReorderedVel.get_access<sycl_read>(cgh);
      auto cellstart = d_CellStart.get_access<sycl_read>(cgh);
      auto cellend = d_CellEnd.get_access<sycl_read>(cgh);
      auto index = d_Index.get_access<sycl_read>(cgh);
      cgh.parallel_for<class Collide>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        collideK(item, vel.get_pointer(), rpos.get_pointer(), rvel.get_pointer(), 
                index.get_pointer(), cellstart.get_pointer(), cellend.get_pointer(), 
                params, numParticles);
      });
    });
}
