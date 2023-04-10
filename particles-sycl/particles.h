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
    unsigned int numCells;
    float3 worldOrigin;
    float3 cellSize;

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
    float4 *__restrict d_Pos,
    float4 *__restrict d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles);

void calcHash(
    queue &q,
    unsigned int *__restrict d_Hash,
    unsigned int *__restrict d_Index,
    float4 *__restrict d_Pos,
    const simParams_t &params,
    const int numParticles);

void memSet(
    queue &q,
    unsigned int *d_Data,
    unsigned int val,
    unsigned int N);

void findCellBoundsAndReorder(
    queue &q,
    unsigned int *__restrict d_CellStart,
    unsigned int *__restrict d_CellEnd,
    float4 *__restrict d_ReorderedPos,
    float4 *__restrict d_ReorderedVel,
    unsigned int *__restrict d_Hash,
    unsigned int *__restrict d_Index,
    float4 *__restrict d_Pos,
    float4 *__restrict d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells);

void collide(
    queue &q,
    float4 *__restrict d_Vel,
    float4 *__restrict d_ReorderedPos,
    float4 *__restrict d_ReorderedVel,
    unsigned int *__restrict d_Index,
    unsigned int *__restrict d_CellStart,
    unsigned int *__restrict d_CellEnd,
    const simParams_t &params,
    const unsigned int numParticles,
    const unsigned int numCells);

void bitonicSort(
    queue &q,
    unsigned int *__restrict d_DstKey,
    unsigned int *__restrict d_DstVal,
    unsigned int *__restrict d_SrcKey,
    unsigned int *__restrict d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir);
#endif
