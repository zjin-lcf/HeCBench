/******************************************************************************
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
template <
  int ACTIVE_CHANNELS,
  int NUM_BINS,
  typename PixelType>
double run_smem_atomics(
    PixelType* image,
    int width,
    int height,
    unsigned int* d_hist,
    bool warmup)
{
  enum
  {
    NUM_PARTS = 1024
  };


  //dim3 block(32, 4);
  //dim3 grid(16, 16);
  //int total_blocks = grid.x * grid.y;
  const int total_blocks = 256;

  auto start = std::chrono::steady_clock::now();

  unsigned int *part_hist = (unsigned int *) malloc (total_blocks * NUM_PARTS * sizeof(unsigned int));
#pragma omp target data map (alloc: part_hist[0: total_blocks * NUM_PARTS]) \
  map(to: image[0:width*height]) \
  map(from: d_hist[0:NUM_BINS * ACTIVE_CHANNELS])
  {
#pragma omp target teams num_teams(total_blocks) thread_limit(128)
    {
      unsigned int smem[ACTIVE_CHANNELS * NUM_BINS + 3];
#pragma omp parallel
      {
        int gid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();
        int x = gid % 512;
        int y = gid / 512;
        int nx = 512;
        int ny = 64;
        int t = omp_get_thread_num();
        int nt = 128;
        int g = omp_get_team_num();

        unsigned int* out = part_hist + g*NUM_PARTS;

        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS + 3; i += nt)
          smem[i] = 0;

#pragma omp barrier

        // process pixels (updates our group's partial histogram in gmem)
        for (int col = x; col < width; col += nx)
        {
          for (int row = y; row < height; row += ny)
          {
            PixelType pixel = image[row * width + col];

            unsigned int bins[ACTIVE_CHANNELS];
            DecodePixel<NUM_BINS>(pixel, bins);

#pragma unroll
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL) {
#pragma omp atomic update
              smem[(NUM_BINS * CHANNEL) + bins[CHANNEL] + CHANNEL]++;
            }
          }
        }
#pragma omp barrier

        // store local output to global
        for (int i = t; i < NUM_BINS; i += nt)
        {
#pragma unroll
          for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            out[i + NUM_BINS * CHANNEL] = smem[i + NUM_BINS * CHANNEL + CHANNEL];
        }
      }
    }

#ifdef DEBUG
    printf("partial histogram:\n");
#pragma omp target update from(part_hist[0:total_blocks * NUM_PARTS])
    for (int i = 0; i < total_blocks * NUM_PARTS; i++) {
      printf("%u\n", part_hist[i]);
    }
#endif


#pragma omp target teams distribute parallel for thread_limit(128)
    for (int i = 0; i < ACTIVE_CHANNELS * NUM_BINS; i++)
    {
      unsigned int total = 0;
      for (int j = 0; j < total_blocks; j++)
        total += part_hist[i + NUM_PARTS * j];

      d_hist[i] = total;
    }
  }
  free(part_hist);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  float elapsed_millis = elapsed_seconds.count()  * 1000;
  return elapsed_millis;
}

