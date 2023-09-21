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
#include "particles.h"
#include "particles_kernels.cpp"

//Simulation parameters

static const size_t wgSize = 64;

static size_t uSnap(size_t a, size_t b){
    return ((a % b) == 0) ? a : (a - (a % b) + b);
}

void integrateSystem(
    sycl::queue &q,
    float4 *d_Pos,
    float4 *d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles
){
    size_t globalWorkSize = uSnap(numParticles, wgSize);
    sycl::range<1> gws (globalWorkSize);
    sycl::range<1> lws (wgSize);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class Integrate>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        integrateSystemK(item, d_Pos, d_Vel, params, deltaTime, numParticles);
      });
    });
}

void calcHash(
    sycl::queue &q,
    unsigned int *d_Hash,
    unsigned int *d_Index,
    float4 *d_Pos,
    const simParams_t &params,
    const int numParticles
){
    size_t globalWorkSize = uSnap(numParticles, wgSize);
    sycl::range<1> gws (globalWorkSize);
    sycl::range<1> lws (wgSize);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class CalcHash>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        calcHashK(item, d_Hash, d_Index, d_Pos, params, numParticles);
      });
    });
}

void memSet(
    sycl::queue &q,
    unsigned int *d_Data,
    unsigned int val,
    unsigned int N
){
    size_t globalWorkSize = uSnap(N, wgSize);

    sycl::range<1> gws (globalWorkSize);
    sycl::range<1> lws (wgSize);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class Memset>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        memSetK(item, d_Data, val, N);
      });
    });
}

void findCellBoundsAndReorder(
    sycl::queue &q,
    unsigned int *d_CellStart,
    unsigned int *d_CellEnd,
    float4 *d_ReorderedPos,
    float4 *d_ReorderedVel,
    unsigned int *d_Hash,
    unsigned int *d_Index,
    float4 *d_Pos,
    float4 *d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells
){
    memSet(q, d_CellStart, 0xFFFFFFFFU, numCells);
    
    size_t globalWorkSize = uSnap(numParticles, wgSize);
    sycl::range<1> gws (globalWorkSize);
    sycl::range<1> lws (wgSize);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<unsigned int, 1> localHash (sycl::range<1>(wgSize + 1), cgh);
      cgh.parallel_for<class FindCellBoundsAndReorder>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        findCellBoundsAndReorderK(item, 
                                 d_CellStart,
                                 d_CellEnd, 
                                 d_ReorderedPos,
                                 d_ReorderedVel, 
                                 d_Hash, 
                                 d_Index, 
                                 d_Pos, 
                                 d_Vel,
                                 localHash.get_pointer(),
                                 numParticles);
      });
    });
}

void collide(
    sycl::queue &q,
    float4 *d_Vel,
    float4 *d_ReorderedPos,
    float4 *d_ReorderedVel,
    unsigned int *d_Index,
    unsigned int *d_CellStart,
    unsigned int *d_CellEnd,
    const simParams_t &params,
    const unsigned int numParticles,
    const unsigned int numCells
){
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    sycl::range<1> gws (globalWorkSize);
    sycl::range<1> lws (wgSize);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class Collide>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        collideK(item, d_Vel, d_ReorderedPos, d_ReorderedVel, 
                d_Index, d_CellStart, d_CellEnd, 
                params, numParticles);
      });
    });
}
