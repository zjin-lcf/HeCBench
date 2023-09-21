#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <sycl/sycl.hpp>
#include "noise_params.hpp"

class Perlin final {
  public:
    /** Calculates Perlin noise with given parameters and stores the result in `hPixels`.
     *  The computation is performed via the given streams.
     *  @return The timing statistics of the computation
     */
    void calculate(sycl::queue &default_stream, 
                   const int *d_hash,
                   const float *d_gradientX,
                   const float *d_gradientY,
                   uint8_t *hPixels, NoiseParams params,
                   sycl::queue *streams, int nStreams);
};
