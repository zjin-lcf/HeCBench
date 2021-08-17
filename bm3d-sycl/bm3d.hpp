#define NOMINMAX

#include <algorithm> //min  max
#include <vector>
#include <math.h>

#include "common.h"
#include "indices.hpp"
#include "params.hpp"

//2DDCT - has to be consistent with dct8x8.cu
#define KER2_BLOCK_WIDTH          128

//Exception handling
#include <sstream>
//Debug
#include <fstream>
#include <iostream>

//Extern kernels

extern "C" void run_block_matching(
  queue &q,
  buffer<uchar, 1> &image,
  buffer<ushort, 1> &stacks,
  buffer<uint, 1> &num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const uint2 start_point,
  const range<2> lws,
  const range<2> gws,
  const uint shared_memory_size
);

extern "C" void run_get_block(
  queue &q,
  const uint2 start_point,
  buffer<uchar, 1> &image,
  buffer<ushort, 1> &stacks,
  buffer<uint, 1> &num_patches_in_stack,
  buffer<float, 1> &patch_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const range<2> lws,
  const range<2> gws
);

extern "C" void run_DCT2D8x8(
  queue &q,
  buffer<float, 1> &d_transformed_stacks,
  buffer<float, 1> &d_gathered_stacks,
  const uint size,
  const range<2> lws,
  const range<2> gws
);

extern "C" void run_hard_treshold_block(
  queue &q,
  const uint2 start_point,
  buffer<float, 1> &patch_stack,
  buffer<float, 1> &w_P,
  buffer<uint, 1> &num_patches_in_stack,
  const uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const range<2> lws,
  const range<2> gws,
  const uint shared_memory_size
);

extern "C" void run_IDCT2D8x8(
  queue &q,
  buffer<float, 1> &d_gathered_stacks,
  buffer<float, 1> &d_transformed_stacks,
  const uint size,
  const range<2> lws,
  const range<2> gws
);

extern "C" void run_aggregate_block(
  queue &q,
  const uint2 start_point,
  buffer<float, 1> &patch_stack,  
  buffer<float, 1> &w_P,
  buffer<ushort, 1> &stacks,
  buffer<float, 1> &kaiser_window,
  buffer<float, 1> &numerator,
  buffer<float, 1> &denominator,
  buffer<uint, 1> &num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const range<2> lws,
  const range<2> gws
);

extern "C" void run_aggregate_final(
  queue &q,
  buffer<float, 1> &numerator,
  buffer<float, 1> &denominator,
  const uint2 image_dim,
  buffer<uchar, 1> &denoised_image,
  const range<2> lws,
  const range<2> gws
);
