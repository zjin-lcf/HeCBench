#ifndef PARTICLES_H
#define PARTICLES_H

#define UMAD(a, b, c)  ( (a) * (b) + (c) )

//Simulation parameters
typedef struct{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

void integrateSystem(
    queue &q,
    buffer<float4,1> &d_Pos,
    buffer<float4,1> &d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const uint numParticles);

void calcHash(
    queue &q,
    buffer<unsigned int, 1> &d_Hash,
    buffer<unsigned int, 1> &d_Index,
    buffer<float4, 1> &d_Pos,
    const simParams_t &params,
    const int numParticles);

void memSet(
    queue &q,
    buffer<unsigned int, 1> &d_Data,
    uint val,
    uint N);

void findCellBoundsAndReorder(
    queue &q,
    buffer<unsigned int, 1> &d_CellStart,
    buffer<unsigned int, 1> &d_CellEnd,
    buffer<float4, 1> &d_ReorderedPos,
    buffer<float4, 1> &d_ReorderedVel,
    buffer<unsigned int, 1> &d_Hash,
    buffer<unsigned int, 1> &d_Index,
    buffer<float4, 1> &d_Pos,
    buffer<float4, 1> &d_Vel,
    const uint numParticles,
    const uint numCells);

void collide(
    queue &q,
    buffer<float4, 1> &d_Vel,
    buffer<float4, 1> &d_ReorderedPos,
    buffer<float4, 1> &d_ReorderedVel,
    buffer<unsigned int, 1> &d_Index,
    buffer<unsigned int, 1> &d_CellStart,
    buffer<unsigned int, 1> &d_CellEnd,
    const simParams_t &params,
    const uint   numParticles,
    const uint   numCells);

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
