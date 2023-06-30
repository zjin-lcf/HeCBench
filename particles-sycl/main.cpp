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

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define GRID_SIZE         64
#define NUM_PARTICLES     16384

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "particles.h"

// Simulation parameters
const float timestep = 0.5f;              // Time slice for re-computation iteration
//const float gravity = 0.0005f;            // Strength of gravity
//const float damping = 1.0f;
const float fParticleRadius = 0.023f;     // Radius of individual particles
const float fColliderRadius = 0.17f;      // Radius of collider for interacting with particles in 'm' mode
//const float collideSpring = 0.4f;         // Elastic spring constant for impact between particles
//const float collideDamping = 0.025f;      // Inelastic loss component for impact between particles
//const float collideShear = 0.12f;         // Friction constant for particles in contact
//const float collideAttraction = 0.0012f;  // Attraction between particles (~static or Van der Waals)

// Forward Function declarations
//*****************************************************************************
inline float frand(void){
  return (float)rand() / (float)RAND_MAX;
}

void initGrid(float *hPos, float *hVel, float particleRadius, float spacing,
    unsigned int numParticles)
{
  float jitter = particleRadius * 0.01f;
  unsigned int s = (int) ceilf(powf((float) numParticles, 1.0f / 3.0f));
  unsigned int gridSize[3];
  gridSize[0] = gridSize[1] = gridSize[2] = s;

  srand(1973);
  for(unsigned int z=0; z<gridSize[2]; z++)
  {
    for(unsigned int y=0; y<gridSize[1]; y++)
    {
      for(unsigned int x=0; x<gridSize[0]; x++)
      {
        unsigned int i = (z * gridSize[0] * gridSize[1]) + (y * gridSize[1]) + x;
        if (i < numParticles)
        {
          hPos[i * 4] = (spacing * x) + particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
          hPos[i * 4 + 1] = (spacing * y) + particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
          hPos[i * 4 + 2] = (spacing * z) + particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
          hPos[i * 4 + 3] = 1.0f;
          hVel[i * 4] = 0.0f;
          hVel[i * 4 + 1] = 0.0f;
          hVel[i * 4 + 2] = 0.0f;
          hVel[i * 4 + 3] = 0.0f;
        }
      }
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }
  const int iterations = atoi(argv[1]);               // Number of iterations
  unsigned int numParticles = NUM_PARTICLES;
  unsigned int gridDim = GRID_SIZE;

  // Set and log grid size and particle count, after checking optional command-line inputs
  uint3 gridSize;
  gridSize.x() = gridSize.y() = gridSize.z() = gridDim;

  unsigned int numGridCells = gridSize.x() * gridSize.y() * gridSize.z();
  //float3 worldSize = {2.0f, 2.0f, 2.0f};

  simParams_t params;

  // set simulation parameters
  params.gridSize = gridSize;
  params.numCells = numGridCells;
  params.numBodies = numParticles;
  params.particleRadius = fParticleRadius;
  params.colliderPos = {1.2f, -0.8f, 0.8f};
  params.colliderRadius = fColliderRadius;

  params.worldOrigin = {1.0f, -1.0f, -1.0f};
  float cellSize = params.particleRadius * 2.0f;  // cell size equal to particle diameter
  params.cellSize = {cellSize, cellSize, cellSize};

  params.spring = 0.5f;
  params.damping = 0.02f;
  params.shear = 0.1f;
  params.attraction = 0.0f;
  params.boundaryDamping = -0.5f;

  params.gravity = {0.0f, -0.0003f, 0.0f};
  params.globalDamping = 1.0f;

  printf(" grid: %d x %d x %d = %d cells\n", gridSize.x(), gridSize.y(), gridSize.z(), numGridCells);
  printf(" particles: %d\n\n", numParticles);

  float* hPos          = (float*)malloc(numParticles * 4 * sizeof(float));
  float* hVel          = (float*)malloc(numParticles * 4 * sizeof(float));
  float* hReorderedPos = (float*)malloc(numParticles * 4 * sizeof(float));
  float* hReorderedVel = (float*)malloc(numParticles * 4 * sizeof(float));
  unsigned int* hHash      = (unsigned int*)malloc(numParticles * sizeof(unsigned int));
  unsigned int* hIndex     = (unsigned int*)malloc(numParticles * sizeof(unsigned int));
  unsigned int* hCellStart = (unsigned int*)malloc(numGridCells * sizeof(unsigned int));
  unsigned int* hCellEnd   = (unsigned int*)malloc(numGridCells * sizeof(unsigned int));

  // configure grid
  initGrid(hPos, hVel, params.particleRadius, params.particleRadius * 2.0f, numParticles);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float4 *dPos = sycl::malloc_device<float4>(numParticles, q);
  q.memcpy(dPos, (float4*)hPos, sizeof(float4) * numParticles);

  float4 *dVel = sycl::malloc_device<float4>(numParticles, q);
  q.memcpy(dVel, (float4*)hVel, sizeof(float4) * numParticles);

  float4 *dReorderedPos = sycl::malloc_device<float4>(numParticles, q);
  float4 *dReorderedVel = sycl::malloc_device<float4>(numParticles, q);

  unsigned int *dHash = sycl::malloc_device<unsigned int>(numParticles, q);
  unsigned int *dIndex = sycl::malloc_device<unsigned int>(numParticles, q);
  unsigned int *dCellStart = sycl::malloc_device<unsigned int>(numGridCells, q);
  unsigned int *dCellEnd = sycl::malloc_device<unsigned int>(numGridCells, q);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++)
  {
    integrateSystem(
        q,
        dPos,
        dVel,
        params,
        timestep,
        numParticles);

    calcHash(
        q,
        dHash,
        dIndex,
        dPos,
        params,
        numParticles);

    bitonicSort(q, dHash, dIndex, dHash, dIndex, 1, numParticles, 0);

    //Find start and end of each cell and
    //Reorder particle data for better cache coherency
    findCellBoundsAndReorder(
        q,
        dCellStart,
        dCellEnd,
        dReorderedPos,
        dReorderedVel,
        dHash,
        dIndex,
        dPos,
        dVel,
        numParticles,
        numGridCells);

    collide(
        q,
        dVel,
        dReorderedPos,
        dReorderedVel,
        dIndex,
        dCellStart,
        dCellEnd,
        params,
        numParticles,
        numGridCells);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total execution time of %d loop iterations: %f (s)\n", iterations, time * 1e-9f);
  printf("Average execution time of a loop iteration: %f (us)\n", (time * 1e-3f) / iterations);

#ifdef DEBUG
  q.memcpy((float4*)hVel, dVel, numParticles * sizeof(float4));
  q.memcpy((float4*)hPos, dPos, numParticles * sizeof(float4));
  q.wait();
  for (unsigned int i = 0; i < numParticles; i++) {
    printf("%d: ", i);
    printf("pos: (%.4f, %.4f, %.4f, %.4f)\n",
        hPos[4*i], hPos[4*i+1], hPos[4*i+2], hPos[4*i+3]);
    printf("vel: (%.4f, %.4f, %.4f, %.4f)\n",
        hVel[4*i], hVel[4*i+1], hVel[4*i+2], hVel[4*i+3]);
  }
#endif

  free(hPos         );
  free(hVel         );
  free(hReorderedPos);
  free(hReorderedVel);
  free(hHash        );
  free(hIndex       );
  free(hCellStart   );
  free(hCellEnd     );
  sycl::free(dPos, q);
  sycl::free(dVel, q);
  sycl::free(dReorderedPos, q);
  sycl::free(dReorderedVel, q);
  sycl::free(dHash, q);
  sycl::free(dIndex, q);
  sycl::free(dCellStart, q);
  sycl::free(dCellEnd, q);

  return EXIT_SUCCESS;
}
