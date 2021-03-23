#ifndef PARTICLES_H
#define PARTICLES_H

#define UMAD(a, b, c)  ( (a) * (b) + (c) )

//Simulation parameters
typedef struct{
    cl::sycl::float3 colliderPos;
    float  colliderRadius;

    cl::sycl::float3 gravity;
    float globalDamping;
    float particleRadius;

    cl::sycl::uint3 gridSize;
    unsigned int numCells;
    cl::sycl::float3 worldOrigin;
    cl::sycl::float3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

void integrateSystem(
    queue &q,
    buffer<cl::sycl::float4,1> &d_Pos,
    buffer<cl::sycl::float4,1> &d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles);

void calcHash(
    queue &q,
    buffer<unsigned int, 1> &d_Hash,
    buffer<unsigned int, 1> &d_Index,
    buffer<cl::sycl::float4, 1> &d_Pos,
    const simParams_t &params,
    const int numParticles);

void memSet(
    queue &q,
    buffer<unsigned int, 1> &d_Data,
    unsigned int val,
    unsigned int N);

void findCellBoundsAndReorder(
    queue &q,
    buffer<unsigned int, 1> &d_CellStart,
    buffer<unsigned int, 1> &d_CellEnd,
    buffer<cl::sycl::float4, 1> &d_ReorderedPos,
    buffer<cl::sycl::float4, 1> &d_ReorderedVel,
    buffer<unsigned int, 1> &d_Hash,
    buffer<unsigned int, 1> &d_Index,
    buffer<cl::sycl::float4, 1> &d_Pos,
    buffer<cl::sycl::float4, 1> &d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells);

void collide(
    queue &q,
    buffer<cl::sycl::float4, 1> &d_Vel,
    buffer<cl::sycl::float4, 1> &d_ReorderedPos,
    buffer<cl::sycl::float4, 1> &d_ReorderedVel,
    buffer<unsigned int, 1> &d_Index,
    buffer<unsigned int, 1> &d_CellStart,
    buffer<unsigned int, 1> &d_CellEnd,
    const simParams_t &params,
    const unsigned int   numParticles,
    const unsigned int   numCells);

void bitonicSort(
    queue &q,
    buffer<unsigned int, 1> &d_DstKey,
    buffer<unsigned int, 1> &d_DstVal,
    buffer<unsigned int, 1> &d_SrcKey,
    buffer<unsigned int, 1> &d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir);
#endif
