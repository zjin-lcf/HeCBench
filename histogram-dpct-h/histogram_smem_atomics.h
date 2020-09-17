#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    void histogram_smem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out,
        sycl::nd_item<3> item_ct1,
        unsigned int *smem)
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

        // initialize smem

        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS + 3; i += nt)
            smem[i] = 0;
    item_ct1.barrier();

        // process pixels
        // updates our group's partial histogram in smem
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                sycl::atomic<unsigned int,
                             sycl::access::address_space::local_space>(
                    sycl::local_ptr<unsigned int>(
                        &smem[(NUM_BINS * CHANNEL) + bins[CHANNEL] + CHANNEL]))
                    .fetch_add(1);
            }
        }

    item_ct1.barrier();

        // move to our workgroup's slice of output
        out += g * NUM_PARTS;

        // store local output to global
        for (int i = t; i < NUM_BINS; i += nt)
        {
            #pragma unroll
            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                out[i + NUM_BINS * CHANNEL] = smem[i + NUM_BINS * CHANNEL + CHANNEL];
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    void histogram_smem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out,
        sycl::nd_item<3> item_ct1)
    {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
        if (i > ACTIVE_CHANNELS * NUM_BINS) return; // out of range
        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];
        out[i] = total;
    }



template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_smem_atomics(
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
    dpct::dpct_malloc(&d_part_hist,
                      total_blocks * NUM_PARTS * sizeof(unsigned int));

    sycl::range<3> block2(128, 1, 1);
    sycl::range<3> grid2(
        (ACTIVE_CHANNELS * NUM_BINS + block2[0] - 1) / block2[0], 1, 1);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    {
        std::pair<dpct::buffer_t, size_t> d_image_buf_ct0 =
            dpct::get_buffer_and_offset(d_image);
        size_t d_image_offset_ct0 = d_image_buf_ct0.second;
        std::pair<dpct::buffer_t, size_t> d_part_hist_buf_ct3 =
            dpct::get_buffer_and_offset(d_part_hist);
        size_t d_part_hist_offset_ct3 = d_part_hist_buf_ct3.second;
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                smem_acc_ct1(sycl::range<1>(ACTIVE_CHANNELS * NUM_BINS + 3),
                             cgh);
            auto d_image_acc_ct0 =
                d_image_buf_ct0.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto d_part_hist_acc_ct3 =
                d_part_hist_buf_ct3.first
                    .get_access<sycl::access::mode::read_write>(cgh);

            auto dpct_global_range = grid * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    PixelType *d_image_ct0 =
                        (PixelType *)(&d_image_acc_ct0[0] + d_image_offset_ct0);
                    unsigned int *d_part_hist_ct3 =
                        (unsigned int *)(&d_part_hist_acc_ct3[0] +
                                         d_part_hist_offset_ct3);
                    histogram_smem_atomics<NUM_PARTS, ACTIVE_CHANNELS,
                                           NUM_BINS>(
                        d_image_ct0, width, height, d_part_hist_ct3, item_ct1,
                        smem_acc_ct1.get_pointer());
                });
        });
    }

    {
        std::pair<dpct::buffer_t, size_t> d_part_hist_buf_ct0 =
            dpct::get_buffer_and_offset(d_part_hist);
        size_t d_part_hist_offset_ct0 = d_part_hist_buf_ct0.second;
        std::pair<dpct::buffer_t, size_t> d_hist_buf_ct2 =
            dpct::get_buffer_and_offset(d_hist);
        size_t d_hist_offset_ct2 = d_hist_buf_ct2.second;
        q_ct1.submit([&](sycl::handler &cgh) {
            auto d_part_hist_acc_ct0 =
                d_part_hist_buf_ct0.first
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto d_hist_acc_ct2 =
                d_hist_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                    cgh);

            auto dpct_global_range = grid2 * block2;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(block2.get(2), block2.get(1),
                                                 block2.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    unsigned int *d_part_hist_ct0 =
                        (unsigned int *)(&d_part_hist_acc_ct0[0] +
                                         d_part_hist_offset_ct0);
                    unsigned int *d_hist_ct2 =
                        (unsigned int *)(&d_hist_acc_ct2[0] +
                                         d_hist_offset_ct2);
                    histogram_smem_accum<NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>(
                        d_part_hist_ct0, total_blocks, d_hist_ct2, item_ct1);
                });
        });
    }

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    dpct::dpct_free(d_part_hist);

    return elapsed_millis;
}

