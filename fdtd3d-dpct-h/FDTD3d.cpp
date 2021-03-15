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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <algorithm>
#include "FDTD3dGPU.h"
#include "shrUtils.h"
#include <cmath>

void finite_difference(float *output, const float* input, const float* coef, 
                                  const int dimx, const int dimy, const int dimz, const int padding,
                                  sycl::nd_item<3> item_ct1,
                                  dpct::accessor<float, dpct::local, 2> tile)
{
  bool valid = true;
  const int gtidx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                    item_ct1.get_local_id(2);
  const int gtidy = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                    item_ct1.get_local_id(1);
  const int ltidx = item_ct1.get_local_id(2);
  const int ltidy = item_ct1.get_local_id(1);
  const int workx = item_ct1.get_local_range().get(2);
  const int worky = item_ct1.get_local_range().get(1);

  const int stride_y = dimx + 2 * k_radius_default;
  const int stride_z = stride_y * (dimy + 2 * k_radius_default);

  int inputIndex  = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += k_radius_default * stride_y + k_radius_default + padding;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float infront[k_radius_default];
  float behind[k_radius_default];
  float current;

  const int tx = ltidx + k_radius_default;
  const int ty = ltidy + k_radius_default;

  if (gtidx >= dimx) valid = false;
  if (gtidy >= dimy) valid = false;

  // For simplicity we assume that the global size is equal to the actual
  // problem size; since the global size must be a multiple of the local size
  // this means the problem size must be a multiple of the local size (or
  // padded to meet this constraint).
  // Preload the "infront" and "behind" data
  for (int i = k_radius_default - 2 ; i >= 0 ; i--)
  {
    behind[i] = input[inputIndex];
    inputIndex += stride_z;
  }

  current = input[inputIndex];
  outputIndex = inputIndex;
  inputIndex += stride_z;

  for (int i = 0 ; i < k_radius_default ; i++)
  {
    infront[i] = input[inputIndex];
    inputIndex += stride_z;
  }

  // Step through the xy-planes
  for (int iz = 0 ; iz < dimz ; iz++)
  {
    // Advance the slice (move the thread-front)
    for (int i = k_radius_default - 1 ; i > 0 ; i--)
      behind[i] = behind[i - 1];
    behind[0] = current;
    current = infront[0];
    for (int i = 0 ; i < k_radius_default - 1 ; i++)
      infront[i] = infront[i + 1];
    infront[k_radius_default - 1] = input[inputIndex];

    inputIndex  += stride_z;
    outputIndex += stride_z;
    item_ct1.barrier();

      // Note that for the work items on the boundary of the problem, the
      // supplied index when reading the halo (below) may wrap to the
      // previous/next row or even the previous/next xy-plane. This is
      // acceptable since a) we disable the output write for these work
      // items and b) there is at least one xy-plane before/after the
      // current plane, so the access will be within bounds.

      // Update the data slice in the local tile
      // Halo above & below
      if (ltidy < k_radius_default)
      {
        tile[ltidy][tx]                  = input[outputIndex - k_radius_default * stride_y];
        tile[ltidy + worky + k_radius_default][tx] = input[outputIndex + worky * stride_y];
      }
    // Halo left & right
    if (ltidx < k_radius_default)
    {
      tile[ty][ltidx]                  = input[outputIndex - k_radius_default];
      tile[ty][ltidx + workx + k_radius_default] = input[outputIndex + workx];
    }
    tile[ty][tx] = current;
    item_ct1.barrier();

      // Compute the output value
      float value = coef[0] * current;
    for (int i = 1 ; i <= k_radius_default ; i++)
    {
      value += coef[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + 
          tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }

    // Store the output value
    if (valid) output[outputIndex] = value;
  }
}

bool fdtdGPU(float *output, const float *input, const float *coeff, 
    const int dimx, const int dimy, const int dimz, const int radius, 
    const int timesteps, const int argc, const char **argv)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
  bool ok = true;
  const int         outerDimx  = dimx + 2 * radius;
  const int         outerDimy  = dimy + 2 * radius;
  const int         outerDimz  = dimz + 2 * radius;
  const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
  size_t            gridSize[2];
  size_t            blockSize[2];

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;


  // Create memory buffer objects
  float* bufferOut;
  bufferOut = sycl::malloc_device<float>(paddedVolumeSize, q_ct1);

  float* bufferIn;
  bufferIn = sycl::malloc_device<float>(paddedVolumeSize, q_ct1);

  float* bufferCoef;
  bufferCoef = sycl::malloc_device<float>((radius + 1), q_ct1);
  q_ct1.memcpy(bufferCoef, coeff, (radius + 1) * sizeof(float)).wait();

  // Set the maximum work group size
  size_t maxWorkSize = 256;

  // Set the work group size
  blockSize[0] = k_localWorkX;  
  blockSize[1] = maxWorkSize / k_localWorkX;
  gridSize[0] = (unsigned int)ceil((float)dimx / blockSize[0]);
  gridSize[1] = (unsigned int)ceil((float)dimy / blockSize[1]);
  shrLog(" set block size to %dx%d\n", blockSize[0], blockSize[1]);
  shrLog(" set grid size to %dx%d\n", gridSize[0], gridSize[1]);
  sycl::range<3> grid(1, gridSize[1], gridSize[0]);
  sycl::range<3> block(1, blockSize[1], blockSize[0]);

  // Copy the input to the device input buffer 
  // offset = padding * 4, bytes = volumeSize * 4
  q_ct1.memcpy(bufferIn + padding, input, volumeSize * sizeof(float)).wait();

  // Copy the input to the device output buffer (actually only need the halo)
  q_ct1.memcpy(bufferOut + padding, input, volumeSize * sizeof(float)).wait();

  // Execute the FDTD
  shrLog(" GPU FDTD loop\n");
  for (int it = 0 ; it < timesteps ; it++)
  {
    // Launch the kernel
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::range<2> tile_range_ct1(
          24 /*k_blockDimMaxY + 2 * k_radius_default*/,
          40 /*k_blockDimMaxX + 2 * k_radius_default*/);

      sycl::accessor<float, 2, sycl::access::mode::read_write,
                     sycl::access::target::local>
          tile_acc_ct1(tile_range_ct1, cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                         finite_difference(
                             bufferOut, bufferIn, bufferCoef, dimx, dimy, dimz,
                             padding, item_ct1,
                             dpct::accessor<float, dpct::local, 2>(
                                 tile_acc_ct1, tile_range_ct1));
                       });
    });

    // Toggle the buffers
    float* tmp = bufferIn;
    bufferIn = bufferOut;
    bufferOut = tmp;
  }

  // Read the result back, result is in bufferSrc (after final toggle)
  q_ct1.memcpy(output, bufferIn + padding, volumeSize * sizeof(float)).wait();
  sycl::free(bufferIn, q_ct1);
  sycl::free(bufferOut, q_ct1);
  sycl::free(bufferCoef, q_ct1);
  return ok;
}
