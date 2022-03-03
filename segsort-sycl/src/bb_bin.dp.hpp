/*
* (c) 2015-2019 Virginia Polytechnic Institute & State University (Virginia Tech)
*          2020 Robin Kobus (kobus@uni-mainz.de)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

#ifndef _H_BB_BIN
#define _H_BB_BIN

#include "bb_segsort_common.dp.hpp"
#include <chrono>
#include <cmath>

#define SEGBIN_NUM 13

template<class T>
void warp_exclusive_sum(const T * in, T * out, const int n,
                        sycl::nd_item<3> item_ct1)
{
    const int lane = item_ct1.get_local_id(2) & 31;

    T data = (lane > 0 && lane < n) ? in[lane-1] : 0;

    for(int i = 1; i < 32; i *= 2) {
        T other = sycl::shift_group_right(item_ct1.get_sub_group(), data, i);
        if(lane > i)
            data += other;
    }

    if(lane < n) {
        out[lane] = data;
    }
}


template<class Offset>
void bb_bin_histo(
    int *d_bin_counter,
    const Offset *d_seg_begins, const Offset *d_seg_ends, const int num_segs,
    sycl::nd_item<3> item_ct1, int *local_histo)
{
    const int tid = item_ct1.get_local_id(2);
    const int gid = item_ct1.get_global_id(2);

    if (tid < SEGBIN_NUM + 1)
        local_histo[tid] = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (gid < num_segs)
    {
        const int size = d_seg_ends[gid] - d_seg_begins[gid];

        if (size <= 1)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[0]))
                .fetch_add(1);
        if (1  < size && size <= 2 )
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[1]))
                .fetch_add(1);
        if (2  < size && size <= 4 )
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[2]))
                .fetch_add(1);
        if (4  < size && size <= 8 )
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[3]))
                .fetch_add(1);
        if (8  < size && size <= 16)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[4]))
                .fetch_add(1);
        if (16 < size && size <= 32)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[5]))
                .fetch_add(1);
        if (32 < size && size <= 64)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[6]))
                .fetch_add(1);
        if (64 < size && size <= 128)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[7]))
                .fetch_add(1);
        if (128 < size && size <= 256)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[8]))
                .fetch_add(1);
        if (256 < size && size <= 512)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[9]))
                .fetch_add(1);
        if (512 < size && size <= 1024)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[10]))
                .fetch_add(1);
        if (1024 < size && size <= 2048)
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[11]))
                .fetch_add(1);
        if (2048 < size) {
            // atomicAdd(&local_histo[12], 1);
            sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(&local_histo[13]))
                .fetch_max(size);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if(tid < 32) {
        warp_exclusive_sum(local_histo, local_histo, SEGBIN_NUM, item_ct1);

        if (tid < SEGBIN_NUM)
            sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[tid]))
                .fetch_add(local_histo[tid]);
        if (tid == SEGBIN_NUM)
            sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[tid]))
                .fetch_max(local_histo[tid]);
    }
}



template<class Offset>
void bb_bin_group(
    int *d_bin_segs_id, int *d_bin_counter,
    const Offset *d_seg_begins, const Offset *d_seg_ends, const int num_segs,
    sycl::nd_item<3> item_ct1)
{
    const int gid = item_ct1.get_global_id(2);

    if (gid < num_segs)
    {
        const int size = d_seg_ends[gid] - d_seg_begins[gid];
        int position;
        if (size <= 1)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[0])).fetch_add(1);
        else if (size <= 2)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[1])).fetch_add(1);
        else if (size <= 4)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[2])).fetch_add(1);
        else if (size <= 8)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[3])).fetch_add(1);
        else if (size <= 16)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[4])).fetch_add(1);
        else if (size <= 32)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[5])).fetch_add(1);
        else if (size <= 64)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[6])).fetch_add(1);
        else if (size <= 128)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[7])).fetch_add(1);
        else if (size <= 256)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[8])).fetch_add(1);
        else if (size <= 512)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[9])).fetch_add(1);
        else if (size <= 1024)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[10])).fetch_add(1);
        else if (size <= 2048)
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[11])).fetch_add(1);
        else
            position = sycl::atomic<int>(sycl::global_ptr<int>(&d_bin_counter[12])).fetch_add(1);
        d_bin_segs_id[position] = gid;
    }
}

template <class Offset>
void bb_bin(const Offset *d_seg_begins, const Offset *d_seg_ends,
            const int num_segs, int *d_bin_segs_id, int *d_bin_counter,
            int *h_bin_counter, sycl::queue *stream, sycl::event event)
{
    stream->memset(d_bin_counter, 0, (SEGBIN_NUM + 1) * sizeof(int));
    std::chrono::time_point<std::chrono::steady_clock> event_ct1;

    const int num_threads = 256;
    const int num_blocks = ceil((double)num_segs/(double)num_threads);

  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        local_histo_acc_ct1(sycl::range<1>(SEGBIN_NUM + 1), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * num_threads),
                          sycl::range<3>(1, 1, num_threads)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          bb_bin_histo(d_bin_counter, d_seg_begins, d_seg_ends, num_segs,
                       item_ct1, local_histo_acc_ct1.get_pointer());
        });
  });

  // show_d(stream, d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

  stream->memcpy(h_bin_counter, d_bin_counter,
                 (SEGBIN_NUM + 1) * sizeof(int));

  event_ct1 = std::chrono::steady_clock::now();

  // group segment IDs (that belong to the same bin) together
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         bb_bin_group(d_bin_segs_id, d_bin_counter,
                                      d_seg_begins, d_seg_ends, num_segs,
                                      item_ct1);
                       }).wait();

    // show_d(stream, d_bin_segs_id, num_segs, "d_bin_segs_id:\n");

    // wait for h_bin_counter copy to host

    // show_h(h_bin_counter, SEGBIN_NUM+1, "h_bin_counter:\n");
}

#endif
