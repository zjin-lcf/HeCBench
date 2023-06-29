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

#ifndef _H_BB_SEGSORT
#define _H_BB_SEGSORT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include "bb_bin.dp.hpp"
#include "bb_comput_s.dp.hpp"
#include "bb_comput_l.dp.hpp"
#include "bb_segsort_common.dp.hpp"

template <class K, class T, class Offset>
void dispatch_kernels(K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
                      const Offset *d_seg_begins, const Offset *d_seg_ends,
                      const int *d_bin_segs_id, const int *h_bin_counter,
                      const int max_segsize, sycl::queue *stream)
{
    int subwarp_size, subwarp_num, factor;
    int threads_per_block;
    int num_blocks;

    threads_per_block = 256;
    subwarp_num = h_bin_counter[1]-h_bin_counter[0];
    num_blocks = (subwarp_num+threads_per_block-1)/threads_per_block;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[0];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) {
            gen_copy(keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                     d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 256;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[2]-h_bin_counter[1];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[1];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk256_wp2_tc1_r2_r2_orig(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[3]-h_bin_counter[2];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[2];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk128_wp2_tc2_r3_r4_orig(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[4]-h_bin_counter[3];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[3];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk128_wp2_tc4_r5_r8_orig(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 128;
    subwarp_size = 4;
    subwarp_num = h_bin_counter[5]-h_bin_counter[4];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[4];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk128_wp4_tc4_r9_r16_strd(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 128;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[6]-h_bin_counter[5];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[5];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk128_wp8_tc4_r17_r32_strd(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 128;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[7]-h_bin_counter[6];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[6];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk128_wp16_tc4_r33_r64_strd(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 256;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[8]-h_bin_counter[7];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[7];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk256_wp8_tc16_r65_r128_strd(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 256;
    subwarp_size = 32;
    subwarp_num = h_bin_counter[9]-h_bin_counter[8];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[8];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk256_wp32_tc8_r129_r256_strd(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1);
          });
    });

    threads_per_block = 128;
    subwarp_num = h_bin_counter[10]-h_bin_counter[9];
    num_blocks = subwarp_num;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<K, 1>
          smem_acc_ct1(sycl::range<1>(512), cgh);
      sycl::local_accessor<int, 1>
          tmem_acc_ct1(sycl::range<1>(512), cgh);

      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[9];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk128_tc4_r257_r512_orig(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1,
                (K *)smem_acc_ct1.get_pointer(), tmem_acc_ct1.get_pointer());
          });
    });

    threads_per_block = 256;
    subwarp_num = h_bin_counter[11]-h_bin_counter[10];
    num_blocks = subwarp_num;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<K, 1>
          smem_acc_ct1(sycl::range<1>(1024), cgh);
      sycl::local_accessor<int, 1>
          tmem_acc_ct1(sycl::range<1>(1024), cgh);

      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[10];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk256_tc4_r513_r1024_orig(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1,
                (K *)smem_acc_ct1.get_pointer(), tmem_acc_ct1.get_pointer());
          });
    });

    threads_per_block = 512;
    subwarp_num = h_bin_counter[12]-h_bin_counter[11];
    num_blocks = subwarp_num;
    if(subwarp_num > 0)
    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<K, 1>
          smem_acc_ct1(sycl::range<1>(2048), cgh);
      sycl::local_accessor<int, 1>
          tmem_acc_ct1(sycl::range<1>(2048), cgh);

      auto d_bin_segs_id_h_bin_counter_ct6 = d_bin_segs_id + h_bin_counter[11];

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * threads_per_block),
                            sycl::range<3>(1, 1, threads_per_block)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            gen_bk512_tc4_r1025_r2048_orig(
                keys_d, vals_d, keysB_d, valsB_d, d_seg_begins, d_seg_ends,
                d_bin_segs_id_h_bin_counter_ct6, subwarp_num, item_ct1,
                (K *)smem_acc_ct1.get_pointer(), tmem_acc_ct1.get_pointer());
          });
    });

    // sort long segments
    subwarp_num = h_bin_counter[13]-h_bin_counter[12];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[12], subwarp_num, max_segsize,
        stream);
}

template <class K, class T, class Offset>
void bb_segsort_run(K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
                    const Offset *d_seg_begins, const Offset *d_seg_ends,
                    const int num_segs, int *d_bin_segs_id, int *h_bin_counter,
                    int *d_bin_counter, sycl::queue *stream, sycl::event event)
{
    bb_bin(d_seg_begins, d_seg_ends, num_segs,
        d_bin_segs_id, d_bin_counter, h_bin_counter,
        stream, event);

    int max_segsize = h_bin_counter[13];
    // std::cout << "max segsize: " << max_segsize << '\n';
    h_bin_counter[13] = num_segs;

    dispatch_kernels(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id, h_bin_counter, max_segsize,
        stream);
}

template <class K, class T, class Offset>
int bb_segsort(sycl::queue &q, K *&keys_d, T *&vals_d, const int num_elements,
               const Offset *d_seg_begins, const Offset *d_seg_ends,
               const int num_segs) try {

    int *h_bin_counter = sycl::malloc_host<int>((SEGBIN_NUM + 1), q);
    int *d_bin_counter = sycl::malloc_device<int>((SEGBIN_NUM + 1), q);
    int *d_bin_segs_id = sycl::malloc_device<int>(num_segs, q);

    K *keysB_d = (K *)sycl::malloc_device(num_elements * sizeof(K), q);
    T *valsB_d = (T *)sycl::malloc_device(num_elements * sizeof(T), q);

    sycl::event event;

    auto start = std::chrono::steady_clock::now();

    bb_segsort_run(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, num_segs,
        d_bin_segs_id, h_bin_counter, d_bin_counter,
        &q, event);

    q.wait();
    auto end = std::chrono::steady_clock::now();
    float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Kernel execution time: %f (s)\n", time * 1e-9f);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);

    sycl::free(h_bin_counter, q);
    sycl::free(d_bin_counter, q);
    sycl::free(d_bin_segs_id, q);
    sycl::free(keysB_d, q);
    sycl::free(valsB_d, q);

    return 1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif
