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
#include <chrono>
#include <sycl/sycl.hpp>
#include "FDTD3dGPU.h"
#include "shrUtils.h"

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

void finite_difference(
  sycl::queue &q,
  sycl::range<3> &gws,
  sycl::range<3> &lws,
  const int slm_size,
        float*__restrict__ output,
  const float*__restrict__ input,
  const float*__restrict__ coef,
  const int dimx, const int dimy, const int dimz,
  const int padding)
{

  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 2> tile(
       sycl::range<2>(k_blockDimMaxY + 2*k_radius_default,
                      k_blockDimMaxX + 2*k_radius_default), cgh);
    auto kfn = [=] (sycl::nd_item<3> item) {

      bool valid = true;
      const int ltidx = item.get_local_id(2);
      const int ltidy = item.get_local_id(1);
      const int workx = item.get_local_range(2);
      const int worky = item.get_local_range(1);
      const int gtidx = item.get_group(2) * workx + ltidx;
      const int gtidy = item.get_group(1) * worky + ltidy;

      const int stride_y = dimx + 2 * k_radius_default;
      const int stride_z = stride_y * (dimy + 2 * k_radius_default);

      int inputIndex  = 0;
      int outputIndex = 0;

      inputIndex += k_radius_default * stride_y + k_radius_default + padding;

      inputIndex += gtidy * stride_y + gtidx;

      float infront[k_radius_default];
      float behind[k_radius_default];
      float current;

      const int tx = ltidx + k_radius_default;
      const int ty = ltidy + k_radius_default;

      if (gtidx >= dimx) valid = false;
      if (gtidy >= dimy) valid = false;

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

      for (int iz = 0 ; iz < dimz ; iz++)
      {
        for (int i = k_radius_default - 1 ; i > 0 ; i--)
          behind[i] = behind[i - 1];
        behind[0] = current;
        current = infront[0];
        for (int i = 0 ; i < k_radius_default - 1 ; i++)
          infront[i] = infront[i + 1];
        infront[k_radius_default - 1] = input[inputIndex];

        inputIndex  += stride_z;
        outputIndex += stride_z;
        item.barrier(sycl::access::fence_space::local_space);

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
        item.barrier(sycl::access::fence_space::local_space);

        float value = coef[0] * current;
        for (int i = 1 ; i <= k_radius_default ; i++)
        {
          value += coef[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] +
              tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value
        if (valid) output[outputIndex] = value;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}


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
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    // Create memory buffer objects
    float *d_output = sycl::malloc_device<float>(paddedVolumeSize, q);
    float *d_input = sycl::malloc_device<float>(paddedVolumeSize, q);
    float *d_coef = sycl::malloc_device<float>(radius+1, q);
    q.memcpy(d_coef, coeff, sizeof(float) * (radius+1));

    // Get the maximum work group size
    size_t userWorkSize = 256;

    // Set the work group size
    localWorkSize[0] = k_localWorkX;
    localWorkSize[1] = userWorkSize / k_localWorkX;
    globalWorkSize[0] = localWorkSize[0] * (unsigned int)ceil((float)dimx / localWorkSize[0]);
    globalWorkSize[1] = localWorkSize[1] * (unsigned int)ceil((float)dimy / localWorkSize[1]);
    shrLog(" set local work group size to %dx%d\n", localWorkSize[0], localWorkSize[1]);
    shrLog(" set total work size to %dx%d\n", globalWorkSize[0], globalWorkSize[1]);
    sycl::range<3> gws (1, globalWorkSize[1], globalWorkSize[0]);
    sycl::range<3> lws (1, localWorkSize[1], localWorkSize[0]);

    // Copy the input to the device input buffer
    // offset = padding * 4, bytes = volumeSize * 4
    q.memcpy(d_input + padding, input, volumeSize * sizeof(float));

    // Copy the input to the device output buffer (actually only need the halo)
    q.memcpy(d_output + padding, input, volumeSize * sizeof(float));

    // Execute the FDTD
    shrLog(" GPU FDTD loop\n");

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int it = 0 ; it < timesteps ; it++)
    {
       // Launch the kernel
       finite_difference(q, gws, lws, 0, d_output, d_input, d_coef, dimx, dimy, dimz, padding);

      // Toggle the buffers
      float* tmp = d_input;
      d_input = d_output;
      d_output = tmp;
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / timesteps);

    // Read the result back, result is in bufferSrc (after final toggle)
    q.memcpy(output, d_input + padding, volumeSize * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_coef, q);

    return ok;
}
