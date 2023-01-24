#include <cmath>
#include <chrono>
#include <iostream>
#include "noise.hpp"
#include "utils.hpp"

inline float lerp(float a, float b, float t)
{
  return a + t*(b-a);
}

inline float dot(sycl::float2 a, sycl::float2 b)
{
  return a.x() * b.x() + a.y() * b.y();
}

// Reference implementation: http://catlikecoding.com/unity/tutorials/noise/

// This function (6t^5 - 15t^4 + 10t^3) has null first and second derivative at the "joint points"
// when used to calculate the distance between x point in the cell and its surrounding corners.
inline float smooth(float t) {
  return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

/*
 * @param x Grid coordinate x
 * @param y Grid coordinate y (top-left = (0, 0))
 * @return Perlin noise at coordinate (x, y).
 */
float noiseAt(float x, float y, int seed, const int *_hash,
              const float *gradientX, const float *gradientY) {

  // Get top-left corner indices
  const int ix = static_cast<int>(x),
            iy = static_cast<int>(y);

  // Weights
  const float wx = x - ix,
              wy = y - iy;

  // Get gradients at cell corners
  const int ix0 = ix & 255, iy0 = iy & 255;
  const int ix1 = (ix0 + 1) & 255, iy1 = (iy0 + 1) & 255;
  const int h0 = _hash[ix0], h1 = _hash[ix1];
  const int iTL = (_hash[h0 + iy0] + seed) % N_GRADIENTS,
            iTR = (_hash[h1 + iy0] + seed) % N_GRADIENTS,
            iBL = (_hash[h0 + iy1] + seed) % N_GRADIENTS,
            iBR = (_hash[h1 + iy1] + seed) % N_GRADIENTS;
  const sycl::float2 gTopLeft = sycl::float2(gradientX[iTL], gradientY[iTL]);
  const sycl::float2 gTopRight = sycl::float2(gradientX[iTR], gradientY[iTR]);
  const sycl::float2 gBotLeft = sycl::float2(gradientX[iBL], gradientY[iBL]);
  const sycl::float2 gBotRight = sycl::float2(gradientX[iBR], gradientY[iBR]);

  // Calculate dots between distance and gradient vectors
  const float dTopLeft = dot(gTopLeft, sycl::float2(wx, wy));
  const float dTopRight = dot(gTopRight, sycl::float2(wx - 1, wy));
  const float dBotLeft = dot(gBotLeft, sycl::float2(wx, wy - 1));
  const float dBotRight = dot(gBotRight, sycl::float2(wx - 1, wy - 1));

  // Calculate the smoothed distance between given point and the top-left corner
  const float tx = smooth(wx), ty = smooth(wy);

  // Interpolate with the other corners
  const float leftInterp  = lerp(dTopLeft, dBotLeft, ty);
  const float rightInterp = lerp(dTopRight, dBotRight, ty);

  return (lerp(leftInterp, rightInterp, tx) + 1.0) * 0.5;
}

/* Calculates several octaves of Perlin noise at coordinate (x, y)
 * (according to the given `params`), sums them together and returns the result.
 */
float sumOctaves(float x, float y, NoiseParams params, const int *_hash,
                 const float *gradientX, const float *gradientY) {
  float frequency = 1;
  float sum = noiseAt(x * frequency, y * frequency, params.seed, _hash,
                      gradientX, gradientY);
  float amplitude = 1;
  float range = 1;
  for (int i = 1; i < params.octaves; i++) {
    frequency *= params.lacunarity;
    amplitude *= params.persistence;
    range += amplitude;
    sum += amplitude * noiseAt(x * frequency, y * frequency, params.seed, _hash,
                               gradientX, gradientY);
  }
  return sum / range;
}

void perlin(int yStart, int height,
            NoiseParams params, uint8_t *outPixels,
            sycl::nd_item<3> &item,
            const int *_hash,
            const float *gradientX,
            const float *gradientY)
{
  // Pixel coordinates
  const auto px = item.get_global_id(2);
  const auto py = item.get_global_id(1) + yStart;

  if (px >= WIN_WIDTH || py >= yStart + height) return;

  auto noise = sumOctaves(px / params.ppu, py / params.ppu, params, _hash,
                          gradientX, gradientY);

  // Convert noise to pixel
  const auto baseIdx = 4 * LIN(px, py, WIN_WIDTH);

  const auto val = noise * 255;

  outPixels[baseIdx + 0] = val;
  outPixels[baseIdx + 1] = val;
  outPixels[baseIdx + 2] = val;
  outPixels[baseIdx + 3] = 255;
}

void Perlin::calculate(sycl::queue &default_stream, 
                       const int *d_hash,
                       const float *d_gradientX,
                       const float *d_gradientY,
                       uint8_t *hPixels,
                       NoiseParams params,
                       sycl::queue *streams,
                       int nStreams)
{
  const auto partialHeight = WIN_HEIGHT / nStreams;
  const sycl::range<3> threads (1, 16, 16);
  const sycl::range<3> blocks (1, std::ceil(partialHeight / 16.0),
                               std::ceil(WIN_WIDTH / 16.0));

  std::cout << "Each stream will calculate " << partialHeight * WIN_WIDTH << " pixels." << std::endl;

  uint8_t *dPixels = sycl::malloc_device<uint8_t>(4 * WIN_WIDTH * WIN_HEIGHT, default_stream);
  CHECK(dPixels); 

  std::cout << "For each stream, ";
  std::cout << "block size = " << "(" << threads[2] << "," << threads[1] << "," << threads[0] << ")"
            << ", grid size = " << "(" << blocks[2] << "," << blocks[1] << "," << blocks[0] << ")"
            << ", total threads = "
            << threads[2] * blocks[2] * threads[1] * blocks[1] << std::endl;

  default_stream.wait();
  for (int i = 0; i < nStreams; ++i) {
    streams[i].wait();
  }

  auto start = std::chrono::steady_clock::now();

  // Note that all kernels use the same device, but they all write on
  // different parts of the memory, so no data race occurs.
  for (int i = 0; i < nStreams; ++i) {
    streams[i].submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(threads*blocks, threads), [=](sycl::nd_item<3> item) {
        perlin(partialHeight * i, partialHeight, params,
               dPixels, item, d_hash, d_gradientX, d_gradientY);
      });
    });
  }

  for (int i = 0; i < nStreams; ++i) {
    streams[i].wait();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Total kernel execution time " <<  time * 1e-6  << " (ms)" << std::endl;

  default_stream.memcpy(hPixels, dPixels, sizeof(uint8_t) * 4 * WIN_WIDTH * WIN_HEIGHT).wait();

  sycl::free(dPixels, default_stream);
}
