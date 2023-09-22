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
void integrateSystemK(
    sycl::nd_item<1> &item,
    float4 *d_Pos,  //input/output
    float4 *d_Vel,  //input/output
    const simParams_t &params,
    float deltaTime,
    unsigned int numParticles)
{
  const unsigned int index = item.get_global_id(0);
  if(index >= numParticles) return;

  float4 pos = d_Pos[index];
  float4 vel = d_Vel[index];

  pos.w() = 1.0f;
  vel.w() = 0.0f;

  //Gravity
  float4 g = {params.gravity.x(), params.gravity.y(), params.gravity.z(), 0};
  vel += g * deltaTime;
  vel *= params.globalDamping;

  //Advance pos
  pos += vel * deltaTime;

  //Collide with cube
  if(pos.x() < -1.0f + params.particleRadius){
    pos.x() = -1.0f + params.particleRadius;
    vel.x() *= params.boundaryDamping;
  }
  if(pos.x() > 1.0f - params.particleRadius){
    pos.x() = 1.0f - params.particleRadius;
    vel.x() *= params.boundaryDamping;
  }

  if(pos.y() < -1.0f + params.particleRadius){
    pos.y() = -1.0f + params.particleRadius;
    vel.y() *= params.boundaryDamping;
  }
  if(pos.y() > 1.0f - params.particleRadius){
    pos.y() = 1.0f - params.particleRadius;
    vel.y() *= params.boundaryDamping;
  }

  if(pos.z() < -1.0f + params.particleRadius){
    pos.z() = -1.0f + params.particleRadius;
    vel.z() *= params.boundaryDamping;
  }
  if(pos.z() > 1.0f - params.particleRadius){
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
int4 getGridPos(float4 p, const simParams_t &params){
  int4 gridPos;
  gridPos.x() = (int)sycl::floor((p.x() - params.worldOrigin.x()) / params.cellSize.x());
  gridPos.y() = (int)sycl::floor((p.y() - params.worldOrigin.y()) / params.cellSize.y());
  gridPos.z() = (int)sycl::floor((p.z() - params.worldOrigin.z()) / params.cellSize.z());
  gridPos.w() = 0;
  return gridPos;
}

//Calculate address in grid from position (clamping to edges)
unsigned int getGridHash(int4 gridPos, const simParams_t &params){
  //Wrap addressing, assume power-of-two grid dimensions
  gridPos.x() = gridPos.x() & (params.gridSize.x() - 1);
  gridPos.y() = gridPos.y() & (params.gridSize.y() - 1);
  gridPos.z() = gridPos.z() & (params.gridSize.z() - 1);
  return UMAD( UMAD(gridPos.z(), params.gridSize.y(), gridPos.y()), params.gridSize.x(), gridPos.x() );
}

//Calculate grid hash value for each particle
void calcHashK(
    sycl::nd_item<1> item,
    unsigned int *d_Hash, //output
    unsigned int *d_Index, //output
    const float4 *d_Pos, //input: positions
    const simParams_t &params,
    unsigned int numParticles)
{
  const unsigned int index = item.get_global_id(0);
  if(index >= numParticles) return;

  float4 p = d_Pos[index];

  //Get address in grid
  int4  gridPos = getGridPos(p, params);
  unsigned int gridHash = getGridHash(gridPos, params);

  //Store grid hash and particle index
  d_Hash[index] = gridHash;
  d_Index[index] = index;
}

////////////////////////////////////////////////////////////////////////////////
// Find cell bounds and reorder positions+velocities by sorted indices
////////////////////////////////////////////////////////////////////////////////
void memSetK(
    sycl::nd_item<1> &item,
    unsigned int *d_Data,
    const unsigned int val,
    const unsigned int N)
{
  if(item.get_global_id(0) < N)
    d_Data[item.get_global_id(0)] = val;
}

void findCellBoundsAndReorderK(
    sycl::nd_item<1> &item,
    unsigned int *d_CellStart,     //output: cell start index
    unsigned int *d_CellEnd,       //output: cell end index
    float4 *d_ReorderedPos,  //output: reordered by cell hash positions
    float4 *d_ReorderedVel,  //output: reordered by cell hash velocities

    const unsigned int *d_Hash,    //input: sorted grid hashes
    const unsigned int *d_Index,   //input: particle indices sorted by hash
    const float4 *d_Pos,     //input: positions array sorted by hash
    const float4 *d_Vel,     //input: velocity array sorted by hash
    unsigned int *localHash,          //get_group_size(0) + 1 elements
    const unsigned int    numParticles)
{
  unsigned int hash;
  const unsigned int index = item.get_global_id(0);
  const unsigned int lid = item.get_local_id(0);

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

  item.barrier(sycl::access::fence_space::local_space);

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
    float4 pos = d_Pos[sortedIndex];
    float4 vel = d_Vel[sortedIndex];

    d_ReorderedPos[index] = pos;
    d_ReorderedVel[index] = vel;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Process collisions (calculate accelerations)
////////////////////////////////////////////////////////////////////////////////
float4 collideSpheres(
    float4 posA,
    float4 posB,
    float4 velA,
    float4 velB,
    float radiusA,
    float radiusB,
    float spring,
    float damping,
    float shear,
    float attraction)
{
  //Calculate relative position
  float4 relPos = {posB.x() - posA.x(), posB.y() - posA.y(), posB.z() - posA.z(), 0};
  float        dist = sycl::sqrt(relPos.x() * relPos.x() + relPos.y() * relPos.y() + relPos.z() * relPos.z());
  float collideDist = radiusA + radiusB;

  float4 force = {0, 0, 0, 0};
  if(dist < collideDist){
    float4 norm = {relPos.x() / dist, relPos.y() / dist, relPos.z() / dist, 0};

    //Relative velocity
    float4 relVel = {velB.x() - velA.x(), velB.y() - velA.y(), velB.z() - velA.z(), 0};

    //Relative tangential velocity
    float relVelDotNorm = relVel.x() * norm.x() + relVel.y() * norm.y() + relVel.z() * norm.z();
    float4 tanVel = {relVel.x() - relVelDotNorm * norm.x(), relVel.y() - relVelDotNorm * norm.y(), 
      relVel.z() - relVelDotNorm * norm.z(), 0};

    //Spring force (potential)
    float springFactor = -spring * (collideDist - dist);
    force = {
      springFactor * norm.x() + damping * relVel.x() + shear * tanVel.x() + attraction * relPos.x(),
      springFactor * norm.y() + damping * relVel.y() + shear * tanVel.y() + attraction * relPos.y(),
      springFactor * norm.z() + damping * relVel.z() + shear * tanVel.z() + attraction * relPos.z(),
      0
    };
  }

  return force;
}

void collideK(
    sycl::nd_item<1> &item,
    float4 *d_Vel,          //output: new velocity
    const float4 *d_ReorderedPos, //input: reordered positions
    const float4 *d_ReorderedVel, //input: reordered velocities
    const unsigned int *d_Index,        //input: reordered particle indices
    const unsigned int *d_CellStart,    //input: cell boundaries
    const unsigned int *d_CellEnd,
    const simParams_t &params,
    const unsigned int numParticles)
{
  unsigned int index = item.get_global_id(0);
  if(index >= numParticles) return;

  float4   pos = d_ReorderedPos[index];
  float4   vel = d_ReorderedVel[index];
  float4 force = {0, 0, 0, 0};

  //Get address in grid
  int4 gridPos = getGridPos(pos, params);

  //Accumulate surrounding cells
  for(int z = -1; z <= 1; z++)
    for(int y = -1; y <= 1; y++)
      for(int x = -1; x <= 1; x++){
        //Get start particle index for this cell
        int4 t = {x, y, z, 0};
        unsigned int   hash = getGridHash(gridPos + t, params);
        unsigned int startI = d_CellStart[hash];

        //Skip empty cell
        if(startI == 0xFFFFFFFFU) continue;

        //Iterate over particles in this cell
        unsigned int endI = d_CellEnd[hash];
        for(unsigned int j = startI; j < endI; j++){
          if(j == index) continue;

          float4 pos2 = d_ReorderedPos[j];
          float4 vel2 = d_ReorderedVel[j];

          //Collide two spheres
          force += collideSpheres(
              pos, pos2,
              vel, vel2,
              params.particleRadius, params.particleRadius, 
              params.spring, params.damping, params.shear, params.attraction);
        }
      }

  //Collide with cursor sphere
  force += collideSpheres(
      pos, {params.colliderPos.x(), params.colliderPos.y(), params.colliderPos.z(), 0},
      vel, {0, 0, 0, 0},
      params.particleRadius, params.colliderRadius,
      params.spring, params.damping, params.shear, params.attraction);

  //Write new velocity back to original unsorted location
  d_Vel[d_Index[index]] = vel + force;
}
