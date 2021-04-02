#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

////////////////////////////////////////////////////////////////////////////////
// Euler integration
////////////////////////////////////////////////////////////////////////////////
void integrateSystemK(sycl::float4 *d_Pos, // input/output
                      sycl::float4 *d_Vel, // input/output
                      const simParams_t params, const float deltaTime,
                      const unsigned int numParticles,
                      sycl::nd_item<3> item_ct1)
{
  const unsigned int index =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  if(index >= numParticles) return;

  sycl::float4 pos = d_Pos[index];
  sycl::float4 vel = d_Vel[index];

  pos.w() = 1.0f;
  vel.w() = 0.0f;

  //Gravity
  sycl::float4 g = {params.gravity.x(), params.gravity.y(), params.gravity.z(),
                    0};
  vel += g * deltaTime;
  vel *= params.globalDamping;

  //Advance pos
  pos += vel * deltaTime;

  //Collide with cube
  if (pos.x() < -1.0f + params.particleRadius) {
    pos.x() = -1.0f + params.particleRadius;
    vel.x() *= params.boundaryDamping;
  }
  if (pos.x() > 1.0f - params.particleRadius) {
    pos.x() = 1.0f - params.particleRadius;
    vel.x() *= params.boundaryDamping;
  }

  if (pos.y() < -1.0f + params.particleRadius) {
    pos.y() = -1.0f + params.particleRadius;
    vel.y() *= params.boundaryDamping;
  }
  if (pos.y() > 1.0f - params.particleRadius) {
    pos.y() = 1.0f - params.particleRadius;
    vel.y() *= params.boundaryDamping;
  }

  if (pos.z() < -1.0f + params.particleRadius) {
    pos.z() = -1.0f + params.particleRadius;
    vel.z() *= params.boundaryDamping;
  }
  if (pos.z() > 1.0f - params.particleRadius) {
    pos.z() = 1.0f - params.particleRadius;
    vel.z() *= params.boundaryDamping;
  }

  //Store new position and velocity
  d_Pos[index] = pos;
  d_Vel[index] = vel;
}


////////////////////////////////////////////////////////////////////////////////
// Save particle grid cell hashes and indices
////////////////////////////////////////////////////////////////////////////////

sycl::int4 getGridPos(const sycl::float4 p, const simParams_t &params)
{
  sycl::int4 gridPos;
  gridPos.x() =
      (int)sycl::floor((p.x() - params.worldOrigin.x()) / params.cellSize.x());
  gridPos.y() =
      (int)sycl::floor((p.y() - params.worldOrigin.y()) / params.cellSize.y());
  gridPos.z() =
      (int)sycl::floor((p.z() - params.worldOrigin.z()) / params.cellSize.z());
  gridPos.w() = 0;
  return gridPos;
}

//Calculate address in grid from position (clamping to edges)

unsigned int getGridHash(sycl::int4 gridPos, const simParams_t &params)
{
  //Wrap addressing, assume power-of-two grid dimensions
  gridPos.x() = gridPos.x() & (params.gridSize.x() - 1);
  gridPos.y() = gridPos.y() & (params.gridSize.y() - 1);
  gridPos.z() = gridPos.z() & (params.gridSize.z() - 1);
  return UMAD(UMAD(gridPos.z(), params.gridSize.y(), gridPos.y()),
              params.gridSize.x(), gridPos.x());
}


//Calculate grid hash value for each particle
void calcHashK(unsigned int *d_Hash,      // output
               unsigned int *d_Index,     // output
               const sycl::float4 *d_Pos, // input: positions
               const simParams_t params, unsigned int numParticles,
               sycl::nd_item<3> item_ct1)
{
  const unsigned int index =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  if(index >= numParticles) return;

  sycl::float4 p = d_Pos[index];

  //Get address in grid
  sycl::int4 gridPos = getGridPos(p, params);
  unsigned int gridHash = getGridHash(gridPos, params);

  //Store grid hash and particle index
  d_Hash[index] = gridHash;
  d_Index[index] = index;
}



////////////////////////////////////////////////////////////////////////////////
// Find cell bounds and reorder positions+velocities by sorted indices
////////////////////////////////////////////////////////////////////////////////
void memSetK(
    unsigned int* d_Data,
    const unsigned int val,
    const unsigned int N,
    sycl::nd_item<3> item_ct1)
{
  unsigned int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                   item_ct1.get_local_id(2);
  if(i < N)
    d_Data[i] = val;
}

void findCellBoundsAndReorderK(
    unsigned int *d_CellStart,    // output: cell start index
    unsigned int *d_CellEnd,      // output: cell end index
    sycl::float4 *d_ReorderedPos, // output: reordered by cell hash positions
    sycl::float4 *d_ReorderedVel, // output: reordered by cell hash velocities
    const unsigned int *d_Hash,   // input: sorted grid hashes
    const unsigned int *d_Index,  // input: particle indices sorted by hash
    const sycl::float4 *d_Pos,    // input: positions array sorted by hash
    const sycl::float4 *d_Vel,    // input: velocity array sorted by hash
    const unsigned int numParticles, sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
  unsigned int hash;
  const unsigned int index =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  const unsigned int lid = item_ct1.get_local_id(2);

  auto localHash = (unsigned int *)dpct_local;

  //Handle case when no. of particles not multiple of block size
  if(index < numParticles){
    hash = d_Hash[index];

    //Load hash data into local memory so that we can look 
    //at neighboring particle's hash value without loading
    //two hash values per thread
    localHash[lid + 1] = hash;

    //First thread in block must load neighbor particle hash
    if(index > 0 && lid == 0)
      localHash[0] = d_Hash[index - 1];
  }

  item_ct1.barrier();

  if(index < numParticles){
    //Border case
    if(index == 0)
      d_CellStart[hash] = 0;

    //Main case
    else{
      if(hash != localHash[lid])
        d_CellEnd[localHash[lid]]  = d_CellStart[hash] = index;
    };

    //Another border case
    if(index == numParticles - 1)
      d_CellEnd[hash] = numParticles;


    //Now use the sorted index to reorder the pos and vel arrays
    unsigned int sortedIndex = d_Index[index];
    sycl::float4 pos = d_Pos[sortedIndex];
    sycl::float4 vel = d_Vel[sortedIndex];

    d_ReorderedPos[index] = pos;
    d_ReorderedVel[index] = vel;
  }
}



////////////////////////////////////////////////////////////////////////////////
// Process collisions (calculate accelerations)
////////////////////////////////////////////////////////////////////////////////

sycl::float4 collideSpheres(sycl::float4 posA, sycl::float4 posB,
                            sycl::float4 velA, sycl::float4 velB, float radiusA,
                            float radiusB, float spring, float damping,
                            float shear, float attraction)
{
  //Calculate relative position
  sycl::float4 relPos = {posB.x() - posA.x(), posB.y() - posA.y(),
                         posB.z() - posA.z(), 0};
  float dist = sycl::sqrt(relPos.x() * relPos.x() + relPos.y() * relPos.y() +
                          relPos.z() * relPos.z());
  float collideDist = radiusA + radiusB;

  sycl::float4 force = {0, 0, 0, 0};
  if(dist < collideDist){
    sycl::float4 norm = {relPos.x() / dist, relPos.y() / dist,
                         relPos.z() / dist, 0};

    //Relative velocity
    sycl::float4 relVel = {velB.x() - velA.x(), velB.y() - velA.y(),
                           velB.z() - velA.z(), 0};

    //Relative tangential velocity
    float relVelDotNorm =
        relVel.x() * norm.x() + relVel.y() * norm.y() + relVel.z() * norm.z();
    sycl::float4 tanVel = {relVel.x() - relVelDotNorm * norm.x(),
                           relVel.y() - relVelDotNorm * norm.y(),
                           relVel.z() - relVelDotNorm * norm.z(), 0};

    //Spring force (potential)
    float springFactor = -spring * (collideDist - dist);
    force = {springFactor * norm.x() + damping * relVel.x() +
                 shear * tanVel.x() + attraction * relPos.x(),
             springFactor * norm.y() + damping * relVel.y() +
                 shear * tanVel.y() + attraction * relPos.y(),
             springFactor * norm.z() + damping * relVel.z() +
                 shear * tanVel.z() + attraction * relPos.z(),
             0};
  }

  return force;
}

void collideK(sycl::float4 *d_Vel,                // output: new velocity
              const sycl::float4 *d_ReorderedPos, // input: reordered positions
              const sycl::float4 *d_ReorderedVel, // input: reordered velocities
              const unsigned int *d_Index, // input: reordered particle indices
              const unsigned int *d_CellStart, // input: cell boundaries
              const unsigned int *d_CellEnd, const simParams_t params,
              const unsigned int numParticles, sycl::nd_item<3> item_ct1)
{
  unsigned int index =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);
  if(index >= numParticles)
    return;

  sycl::float4 pos = d_ReorderedPos[index];
  sycl::float4 vel = d_ReorderedVel[index];
  sycl::float4 force = {0, 0, 0, 0};

  //Get address in grid
  sycl::int4 gridPos = getGridPos(pos, params);

  //Accumulate surrounding cells
  for(int z = -1; z <= 1; z++)
    for(int y = -1; y <= 1; y++)
      for(int x = -1; x <= 1; x++){
        //Get start particle index for this cell
        sycl::int4 t = {x, y, z, 0};
        unsigned int   hash = getGridHash(gridPos + t, params);
        unsigned int startI = d_CellStart[hash];

        //Skip empty cell
        if(startI == 0xFFFFFFFFU) continue;

        //Iterate over particles in this cell
        unsigned int endI = d_CellEnd[hash];
        for(unsigned int j = startI; j < endI; j++){
          if(j == index) continue;

          sycl::float4 pos2 = d_ReorderedPos[j];
          sycl::float4 vel2 = d_ReorderedVel[j];

          //Collide two spheres
          force += collideSpheres(
              pos, pos2,
              vel, vel2,
              params.particleRadius, params.particleRadius, 
              params.spring, params.damping, params.shear, params.attraction
              );
        }
      }

  //Collide with cursor sphere
  force += collideSpheres(
      pos, {params.colliderPos.x(), params.colliderPos.y(), params.colliderPos.z(), 0},
      vel, {0, 0, 0, 0},
      params.particleRadius, params.colliderRadius,
      params.spring, params.damping, params.shear, params.attraction
      );

  //Write new velocity back to original unsorted location
  d_Vel[d_Index[index]] = vel + force;
}
