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

#include <iostream>
#include <algorithm>
#include "FDTD3dGPU.h"
#include "shrUtils.h"
#include "common.h"

bool fdtdGPU(float *output, const float *input, const float *coeff, 
             const int dimx, const int dimy, const int dimz, const int radius, 
             const int timesteps, const int argc, const char **argv)
{
    bool ok = true;
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
    size_t            globalWorkSize[2];
    size_t            localWorkSize[2];

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    // Create memory buffer objects
    buffer<float, 1> bufferOut (paddedVolumeSize);
    buffer<float, 1> bufferIn (paddedVolumeSize);
    buffer<float, 1> bufferCoef (coeff, radius+1);


    // Get the maximum work group size
    size_t userWorkSize = 256;

    // Set the work group size
    localWorkSize[0] = k_localWorkX;
    localWorkSize[1] = userWorkSize / k_localWorkX;
    globalWorkSize[0] = localWorkSize[0] * (unsigned int)ceil((float)dimx / localWorkSize[0]);
    globalWorkSize[1] = localWorkSize[1] * (unsigned int)ceil((float)dimy / localWorkSize[1]);
    shrLog(" set local work group size to %dx%d\n", localWorkSize[0], localWorkSize[1]);
    shrLog(" set total work size to %dx%d\n", globalWorkSize[0], globalWorkSize[1]);
    range<2> gws (globalWorkSize[1], globalWorkSize[0]);
    range<2> lws (localWorkSize[1], localWorkSize[0]);

    // Copy the input to the device input buffer 
    // offset = padding * 4, bytes = volumeSize * 4
    q.submit([&] (handler &cgh) {
      auto in = bufferIn.get_access<sycl_write>(cgh, volumeSize, padding);
      cgh.copy(input, in);
    });

    // Copy the input to the device output buffer (actually only need the halo)
    q.submit([&] (handler &cgh) {
      auto in = bufferOut.get_access<sycl_write>(cgh, volumeSize, padding);
      cgh.copy(input, in);
    });
      
    // Execute the FDTD
    shrLog(" GPU FDTD loop\n");
    for (int it = 0 ; it < timesteps ; it++)
    {
        // Launch the kernel
      q.submit([&] (handler &cgh) {
        auto output = bufferOut.get_access<sycl_write>(cgh);
        auto input = bufferIn.get_access<sycl_read>(cgh);
        auto coef = bufferCoef.get_access<sycl_read>(cgh);
        accessor<float, 2, sycl_read_write, access::target::local> tile(
          {localWorkMaxY + 2*k_radius_default, localWorkMaxX + 2*k_radius_default} , cgh);
        cgh.parallel_for<class finite_difference>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          bool valid = true;
          const int gtidx = item.get_global_id(1);
          const int gtidy = item.get_global_id(0);
          const int ltidx = item.get_local_id(1);
          const int ltidy = item.get_local_id(0);
          const int workx = item.get_local_range(1);
          const int worky = item.get_local_range(0);
          
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
              item.barrier(access::fence_space::local_space);

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
              item.barrier(access::fence_space::local_space);

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
        });
      });

      // Toggle the buffers
      auto tmp = std::move(bufferIn);
      bufferIn = std::move(bufferOut);
      bufferOut = std::move(tmp);
    }

    // Read the result back, result is in bufferSrc (after final toggle)
    q.submit([&] (handler &cgh) {
      auto src = bufferIn.get_access<sycl_read>(cgh, volumeSize, padding);
      cgh.copy(src, output);
    });
    q.wait();

    return ok;
}
