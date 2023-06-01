// Based on the Perlin noise generator written by G. Parolini && I. Cislaghi

#include <iostream>
#include <unistd.h>
#include "utils.hpp"
#include "noise.hpp"

int main(int argc, char **argv) {

  NoiseParams params;
  // set default values for the parameters
  params.ppu = 250.f;
  params.seed = 0;
  params.octaves = 3;
  params.lacunarity = 2;
  params.persistence = 0.5;
  
#ifdef USE_GPU
  sycl::queue default_stream (sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue default_stream (sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint8_t *hPixels = sycl::malloc_host<uint8_t>(4 * WIN_WIDTH * WIN_HEIGHT, default_stream);
  CHECK(hPixels);

  int *d_hash = sycl::malloc_device<int>(256, default_stream);
  CHECK(d_hash);
  default_stream.memcpy(d_hash, _hash, 256 * sizeof(int));

  float *d_gradientX = sycl::malloc_device<float>(N_GRADIENTS, default_stream);
  CHECK(d_gradientX);
  default_stream.memcpy(d_gradientX, gradientX, N_GRADIENTS * sizeof(float)); 

  float *d_gradientY = sycl::malloc_device<float>(N_GRADIENTS, default_stream);
  CHECK(d_gradientY);
  default_stream.memcpy(d_gradientY, gradientY, N_GRADIENTS * sizeof(float)); 

  for (int nStreams = 1; nStreams <= 32; nStreams *= 2) {

    std::cout << std::endl;
    std::cout << "Using " << nStreams << " streams." << std::endl;

    Perlin perlin;

    sycl::queue *streams = new sycl::queue [nStreams];
    CHECK(streams);

    for (int i = 0; i < nStreams; ++i) {
#ifdef USE_GPU
      streams[i] = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
      streams[i] = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
    }

    perlin.calculate(default_stream, d_hash, d_gradientX, d_gradientY,
                     hPixels, params, streams, nStreams);

    uint64_t checksum = 0;
    for (uint64_t i = 0; i < 4 * WIN_WIDTH * WIN_HEIGHT; ++i)
      checksum += hPixels[i];
    std::cout << "checksum = " << checksum / (4 * WIN_WIDTH * WIN_HEIGHT) << std::endl;

    delete[] streams;
  }

  sycl::free(hPixels, default_stream);
  sycl::free(d_hash, default_stream);
  sycl::free(d_gradientX, default_stream);
  sycl::free(d_gradientY, default_stream);

  return 0;
}
