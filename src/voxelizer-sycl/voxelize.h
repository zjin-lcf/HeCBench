#pragma once

// Commun functions for both the solid and non-solid voxelization methods

#include <cuda.h>
//#include "helper_cuda.h"

#ifdef USE_GPU
#define GLM_FORCE_CUDA
#endif

// #define GLM_FORCE_PURE (not needed anymore with recent GLM versions)
#include <glm/glm.hpp>

#include <iostream>
#include "util.h"
#include "morton_LUTs.h"

// Encode morton code using LUT table
inline uint64_t mortonEncode_LUT(
  const unsigned int *morton256_x,
  const unsigned int *morton256_y,
  const unsigned int *morton256_z,
  unsigned int x, unsigned int y, unsigned int z)
{
  uint64_t answer = 0;
  answer = morton256_z[(z >> 16) & 0xFF] |
    morton256_y[(y >> 16) & 0xFF] |
    morton256_x[(x >> 16) & 0xFF];

  answer = answer << 48 |
    morton256_z[(z >> 8) & 0xFF] |
    morton256_y[(y >> 8) & 0xFF] |
    morton256_x[(x >> 8) & 0xFF];

  answer = answer << 24 |
    morton256_z[(z)& 0xFF] |
    morton256_y[(y)& 0xFF] |
    morton256_x[(x)& 0xFF];

  return answer;
}
