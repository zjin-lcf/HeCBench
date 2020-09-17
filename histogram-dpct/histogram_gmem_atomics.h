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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "test_util.h"

namespace histogram_gmem_atomics
{
    // Decode float4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__dpct_inline__ void DecodePixel(sycl::float4 pixel,
                                 unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        float* samples = reinterpret_cast<float*>(&pixel);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
    }

    // Decode uchar4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__dpct_inline__ void DecodePixel(sycl::uchar4 pixel,
                                 unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
    }

    // Decode uchar1 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__dpct_inline__ void DecodePixel(unsigned char pixel,
                                 unsigned int (&bins)[ACTIVE_CHANNELS])
    {
    bins[0] = (unsigned int)pixel;
    }

    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    void histogram_gmem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out,
        sycl::nd_item<3> item_ct1)
    {
        // global position and size
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    int y = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
            item_ct1.get_local_id(1);
    int nx = item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);
    int ny = item_ct1.get_local_range().get(1) * item_ct1.get_group_range(1);

        // threads in workgroup
    int t = item_ct1.get_local_id(2) +
            item_ct1.get_local_id(1) *
                item_ct1.get_local_range().get(
                    2); // thread index in workgroup, linear in 0..nt-1
    int nt = item_ct1.get_local_range().get(2) *
             item_ct1.get_local_range().get(1); // total threads in workgroup

        // group index in 0..ngroups-1
    int g = item_ct1.get_group(2) +
            item_ct1.get_group(1) * item_ct1.get_group_range(2);

        // initialize global memory
        unsigned int *gmem = out + g * NUM_PARTS;
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
            gmem[i] = 0;
    item_ct1.barrier();

        // process pixels (updates our group's partial histogram in gmem)
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                sycl::atomic<unsigned int>(
                    sycl::global_ptr<unsigned int>(
                        &gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]]))
                    .fetch_add(1);
            }
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    void histogram_gmem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out,
        sycl::nd_item<3> item_ct1)
    {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
        if (i > ACTIVE_CHANNELS * NUM_BINS)
            return; // out of range

        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];

        out[i] = total;
    }


}   // namespace histogram_gmem_atomics


template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_gmem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist,
    bool warmup)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    enum
    {
        NUM_PARTS = 1024
    };

    dpct::device_info props;
    dpct::dev_mgr::instance().get_device(0).get_device_info(props);

    sycl::range<3> block(32, 4, 1);
    sycl::range<3> grid(16, 16, 1);
    int total_blocks = grid[0] * grid[1];

    // allocate partial histogram
    unsigned int *d_part_hist;
    d_part_hist =
        sycl::malloc_device<unsigned int>(total_blocks * NUM_PARTS, q_ct1);

    sycl::range<3> block2(128, 1, 1);
    sycl::range<3> grid2((3 * NUM_BINS + block[0] - 1) / block[0], 1, 1);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    /*
    DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid * block;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block.get(2), block.get(1), block.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
	    histogram_gmem_atomics::histogram_gmem_atomics<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>(
                    d_image, width, height, d_part_hist, item_ct1);
            });
    });

    /*
    DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
        auto dpct_global_range = grid2 * block2;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block2.get(2), block2.get(1), block2.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
	    histogram_gmem_atomics::histogram_gmem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>(
                    d_part_hist, total_blocks, d_hist, item_ct1);
            });
    });

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    sycl::free(d_part_hist, q_ct1);

    return elapsed_millis;
}

