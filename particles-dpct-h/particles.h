#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef PARTICLES_H
#define PARTICLES_H

#define UMAD(a, b, c)  ( (a) * (b) + (c) )

//Simulation parameters
typedef struct dpct_type_e85abf {
    sycl::float3 colliderPos;
    float  colliderRadius;

    sycl::float3 gravity;
    float globalDamping;
    float particleRadius;

    sycl::uint3 gridSize;
    unsigned int numCells;
    sycl::float3 worldOrigin;
    sycl::float3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

void integrateSystem(sycl::float4 *d_Pos, sycl::float4 *d_Vel,
                     const simParams_t &params, const float deltaTime,
                     const unsigned int numParticles);

void calcHash(unsigned int *d_Hash, unsigned int *d_Index, sycl::float4 *d_Pos,
              const simParams_t &params, const int numParticles);

void memSet(
    unsigned int* d_Data,
    unsigned int val,
    unsigned int N);

void findCellBoundsAndReorder(unsigned int *d_CellStart,
                              unsigned int *d_CellEnd,
                              sycl::float4 *d_ReorderedPos,
                              sycl::float4 *d_ReorderedVel,
                              unsigned int *d_Hash, unsigned int *d_Index,
                              sycl::float4 *d_Pos, sycl::float4 *d_Vel,
                              const unsigned int numParticles,
                              const unsigned int numCells);

void collide(sycl::float4 *d_Vel, sycl::float4 *d_ReorderedPos,
             sycl::float4 *d_ReorderedVel, unsigned int *d_Index,
             unsigned int *d_CellStart, unsigned int *d_CellEnd,
             const simParams_t &params, const unsigned int numParticles,
             const unsigned int numCells);

void bitonicSort(
    unsigned int* d_DstKey,
    unsigned int* d_DstVal,
    unsigned int* d_SrcKey,
    unsigned int* d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir);
#endif
