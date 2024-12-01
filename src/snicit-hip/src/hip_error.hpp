#pragma once
#include <assert.h>
#include <hip/hip_runtime.h>

// Convenience function for checking HIP runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
hipError_t checkHip(hipError_t result)
{
  if (result != hipSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("HIP Runtime Error : "s + hipGetErrorString(result));
    //assert(result == hipSuccess);
  }
  return result;
}
