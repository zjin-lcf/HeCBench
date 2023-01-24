#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <iomanip>
#include "noise_params.hpp"
#include <hip/hip_runtime.h>

class Perlin final {
  public:
    /** Calculates Perlin noise with given parameters and stores the result in `hPixels`.
     *  The computation is performed via the given CUDA streams.
     *  @return The timing statistics of the computation
     */
    void calculate(uint8_t *hPixels, NoiseParams params, hipStream_t *streams, int nStreams);
};
