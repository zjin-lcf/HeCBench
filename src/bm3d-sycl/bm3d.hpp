#define NOMINMAX

#include <algorithm> //min  max
#include <vector>
#include <math.h>
#include <sycl/sycl.hpp>
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

void run_block_matching(
  sycl::queue &q,
  sycl::uchar *image,
  ushort *stacks,
  uint *num_patches_in_stack,
  const sycl::uint2 image_dim,
  const sycl::uint2 stacks_dim,
  const Params params,
  const sycl::uint2 start_point,
  const sycl::range<2> lws,
  const sycl::range<2> gws,
  const uint shared_memory_size
);

void run_get_block(
  sycl::queue &q,
  const sycl::uint2 start_point,
  sycl::uchar *image,
  ushort *stacks,
  uint *num_patches_in_stack,
  float *patch_stack,
  const sycl::uint2 image_dim,
  const sycl::uint2 stacks_dim,
  const Params params,
  const sycl::range<2> lws,
  const sycl::range<2> gws
);

void run_DCT2D8x8(
  sycl::queue &q,
  float *d_transformed_stacks,
  float *d_gathered_stacks,
  const uint size,
  const sycl::range<2> lws,
  const sycl::range<2> gws
);

void run_hard_treshold_block(
  sycl::queue &q,
  const sycl::uint2 start_point,
  float *patch_stack,
  float *w_P,
  uint *num_patches_in_stack,
  const sycl::uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const sycl::range<2> lws,
  const sycl::range<2> gws,
  const uint shared_memory_size
);

void run_IDCT2D8x8(
  sycl::queue &q,
  float *d_gathered_stacks,
  float *d_transformed_stacks,
  const uint size,
  const sycl::range<2> lws,
  const sycl::range<2> gws
);

void run_aggregate_block(
  sycl::queue &q,
  const sycl::uint2 start_point,
  float *patch_stack,  
  float *w_P,
  ushort *stacks,
  float *kaiser_window,
  float *numerator,
  float *denominator,
  uint *num_patches_in_stack,
  const sycl::uint2 image_dim,
  const sycl::uint2 stacks_dim,
  const Params params,
  const sycl::range<2> lws,
  const sycl::range<2> gws
);

void run_aggregate_final(
  sycl::queue &q,
  float *numerator,
  float *denominator,
  const sycl::uint2 image_dim,
  sycl::uchar *denoised_image,
  const sycl::range<2> lws,
  const sycl::range<2> gws
);
