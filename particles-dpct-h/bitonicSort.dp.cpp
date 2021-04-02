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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cassert>

#define LOCAL_SIZE_LIMIT 512U

#include "bitonicSort_kernels.dp.cpp"

static unsigned int factorRadix2(unsigned int& log2L, unsigned int L){
  if(!L){
    log2L = 0;
    return 0;
  }else{
    for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
    return L;
  }
}

void bitonicSort(
    unsigned int *d_dstKey,
    unsigned int *d_dstVal,
    unsigned int *d_srcKey,
    unsigned int *d_srcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir
    )
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  if(arrayLength < 2) return;

  //Only power-of-two array lengths are supported so far
  unsigned int log2L;
  unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  size_t localWorkSize, globalWorkSize;

  if(arrayLength <= LOCAL_SIZE_LIMIT)
  {
    assert( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0 );

    //Launch bitonicSortLocal
    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;
    sycl::range<3> bs_gws(1, 1, globalWorkSize / localWorkSize);
    sycl::range<3> bs_lws(1, 1, localWorkSize);

    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    {
      std::pair<dpct::buffer_t, size_t> d_dstKey_buf_ct0 =
          dpct::get_buffer_and_offset(d_dstKey);
      size_t d_dstKey_offset_ct0 = d_dstKey_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> d_dstVal_buf_ct1 =
          dpct::get_buffer_and_offset(d_dstVal);
      size_t d_dstVal_offset_ct1 = d_dstVal_buf_ct1.second;
      std::pair<dpct::buffer_t, size_t> d_srcKey_buf_ct2 =
          dpct::get_buffer_and_offset(d_srcKey);
      size_t d_srcKey_offset_ct2 = d_srcKey_buf_ct2.second;
      std::pair<dpct::buffer_t, size_t> d_srcVal_buf_ct3 =
          dpct::get_buffer_and_offset(d_srcVal);
      size_t d_srcVal_offset_ct3 = d_srcVal_buf_ct3.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            l_key_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            l_val_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
        auto d_dstKey_acc_ct0 =
            d_dstKey_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_dstVal_acc_ct1 =
            d_dstVal_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_srcKey_acc_ct2 =
            d_srcKey_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_srcVal_acc_ct3 =
            d_srcVal_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(bs_gws * bs_lws, bs_lws),
            [=](sycl::nd_item<3> item_ct1) {
              unsigned int *d_dstKey_ct0 =
                  (unsigned int *)(&d_dstKey_acc_ct0[0] + d_dstKey_offset_ct0);
              unsigned int *d_dstVal_ct1 =
                  (unsigned int *)(&d_dstVal_acc_ct1[0] + d_dstVal_offset_ct1);
              unsigned int *d_srcKey_ct2 =
                  (unsigned int *)(&d_srcKey_acc_ct2[0] + d_srcKey_offset_ct2);
              unsigned int *d_srcVal_ct3 =
                  (unsigned int *)(&d_srcVal_acc_ct3[0] + d_srcVal_offset_ct3);
              bitonicSortLocal(d_dstKey_ct0, d_dstVal_ct1, d_srcKey_ct2,
                               d_srcVal_ct3, arrayLength, dir, item_ct1,
                               l_key_acc_ct1.get_pointer(),
                               l_val_acc_ct1.get_pointer());
            });
      });
    }
  }
  else
  {
    //Launch bitonicSortLocal1
    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;
    sycl::range<3> bs1_gws(1, 1, globalWorkSize / localWorkSize);
    sycl::range<3> bs1_lws(1, 1, localWorkSize);
    /*
    DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    {
      std::pair<dpct::buffer_t, size_t> d_dstKey_buf_ct0 =
          dpct::get_buffer_and_offset(d_dstKey);
      size_t d_dstKey_offset_ct0 = d_dstKey_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> d_dstVal_buf_ct1 =
          dpct::get_buffer_and_offset(d_dstVal);
      size_t d_dstVal_offset_ct1 = d_dstVal_buf_ct1.second;
      std::pair<dpct::buffer_t, size_t> d_srcKey_buf_ct2 =
          dpct::get_buffer_and_offset(d_srcKey);
      size_t d_srcKey_offset_ct2 = d_srcKey_buf_ct2.second;
      std::pair<dpct::buffer_t, size_t> d_srcVal_buf_ct3 =
          dpct::get_buffer_and_offset(d_srcVal);
      size_t d_srcVal_offset_ct3 = d_srcVal_buf_ct3.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            l_key_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            l_val_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
        auto d_dstKey_acc_ct0 =
            d_dstKey_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_dstVal_acc_ct1 =
            d_dstVal_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_srcKey_acc_ct2 =
            d_srcKey_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_srcVal_acc_ct3 =
            d_srcVal_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(bs1_gws * bs1_lws, bs1_lws),
            [=](sycl::nd_item<3> item_ct1) {
              unsigned int *d_dstKey_ct0 =
                  (unsigned int *)(&d_dstKey_acc_ct0[0] + d_dstKey_offset_ct0);
              unsigned int *d_dstVal_ct1 =
                  (unsigned int *)(&d_dstVal_acc_ct1[0] + d_dstVal_offset_ct1);
              unsigned int *d_srcKey_ct2 =
                  (unsigned int *)(&d_srcKey_acc_ct2[0] + d_srcKey_offset_ct2);
              unsigned int *d_srcVal_ct3 =
                  (unsigned int *)(&d_srcVal_acc_ct3[0] + d_srcVal_offset_ct3);
              bitonicSortLocal1(d_dstKey_ct0, d_dstVal_ct1, d_srcKey_ct2,
                                d_srcVal_ct3, item_ct1,
                                l_key_acc_ct1.get_pointer(),
                                l_val_acc_ct1.get_pointer());
            });
      });
    }

    for(unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
    {
      for(unsigned stride = size / 2; stride > 0; stride >>= 1)
      {
        if(stride >= LOCAL_SIZE_LIMIT)
        {
          //Launch bitonicMergeGlobal
          localWorkSize  = LOCAL_SIZE_LIMIT / 4;
          globalWorkSize = batch * arrayLength / 2;
          sycl::range<3> bmg_gws(1, 1, globalWorkSize / localWorkSize);
          sycl::range<3> bmg_lws(1, 1, localWorkSize);

          /*
          DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the workgroup size if
          needed.
          */
          {
            std::pair<dpct::buffer_t, size_t> d_dstKey_buf_ct0 =
                dpct::get_buffer_and_offset(d_dstKey);
            size_t d_dstKey_offset_ct0 = d_dstKey_buf_ct0.second;
            std::pair<dpct::buffer_t, size_t> d_dstVal_buf_ct1 =
                dpct::get_buffer_and_offset(d_dstVal);
            size_t d_dstVal_offset_ct1 = d_dstVal_buf_ct1.second;
            std::pair<dpct::buffer_t, size_t> d_dstKey_buf_ct2 =
                dpct::get_buffer_and_offset(d_dstKey);
            size_t d_dstKey_offset_ct2 = d_dstKey_buf_ct2.second;
            std::pair<dpct::buffer_t, size_t> d_dstVal_buf_ct3 =
                dpct::get_buffer_and_offset(d_dstVal);
            size_t d_dstVal_offset_ct3 = d_dstVal_buf_ct3.second;
            q_ct1.submit([&](sycl::handler &cgh) {
              auto d_dstKey_acc_ct0 =
                  d_dstKey_buf_ct0.first
                      .get_access<sycl::access::mode::read_write>(cgh);
              auto d_dstVal_acc_ct1 =
                  d_dstVal_buf_ct1.first
                      .get_access<sycl::access::mode::read_write>(cgh);
              auto d_dstKey_acc_ct2 =
                  d_dstKey_buf_ct2.first
                      .get_access<sycl::access::mode::read_write>(cgh);
              auto d_dstVal_acc_ct3 =
                  d_dstVal_buf_ct3.first
                      .get_access<sycl::access::mode::read_write>(cgh);

              cgh.parallel_for(sycl::nd_range<3>(bmg_gws * bmg_lws, bmg_lws),
                               [=](sycl::nd_item<3> item_ct1) {
                                 unsigned int *d_dstKey_ct0 =
                                     (unsigned int *)(&d_dstKey_acc_ct0[0] +
                                                      d_dstKey_offset_ct0);
                                 unsigned int *d_dstVal_ct1 =
                                     (unsigned int *)(&d_dstVal_acc_ct1[0] +
                                                      d_dstVal_offset_ct1);
                                 unsigned int *d_dstKey_ct2 =
                                     (unsigned int *)(&d_dstKey_acc_ct2[0] +
                                                      d_dstKey_offset_ct2);
                                 unsigned int *d_dstVal_ct3 =
                                     (unsigned int *)(&d_dstVal_acc_ct3[0] +
                                                      d_dstVal_offset_ct3);
                                 bitonicMergeGlobal(d_dstKey_ct0, d_dstVal_ct1,
                                                    d_dstKey_ct2, d_dstVal_ct3,
                                                    arrayLength, size, stride,
                                                    dir, item_ct1);
                               });
            });
          }
        }
        else
        {
          //Launch bitonicMergeLocal
          localWorkSize  = LOCAL_SIZE_LIMIT / 2;
          globalWorkSize = batch * arrayLength / 2;

          sycl::range<3> bml_gws(1, 1, globalWorkSize / localWorkSize);
          sycl::range<3> bml_lws(1, 1, localWorkSize);

          assert(stride < LOCAL_SIZE_LIMIT);
          /*
          DPCT1049:3: The workgroup size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the workgroup size if
          needed.
          */
          {
            std::pair<dpct::buffer_t, size_t> d_dstKey_buf_ct0 =
                dpct::get_buffer_and_offset(d_dstKey);
            size_t d_dstKey_offset_ct0 = d_dstKey_buf_ct0.second;
            std::pair<dpct::buffer_t, size_t> d_dstVal_buf_ct1 =
                dpct::get_buffer_and_offset(d_dstVal);
            size_t d_dstVal_offset_ct1 = d_dstVal_buf_ct1.second;
            std::pair<dpct::buffer_t, size_t> d_dstKey_buf_ct2 =
                dpct::get_buffer_and_offset(d_dstKey);
            size_t d_dstKey_offset_ct2 = d_dstKey_buf_ct2.second;
            std::pair<dpct::buffer_t, size_t> d_dstVal_buf_ct3 =
                dpct::get_buffer_and_offset(d_dstVal);
            size_t d_dstVal_offset_ct3 = d_dstVal_buf_ct3.second;
            q_ct1.submit([&](sycl::handler &cgh) {
              sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                             sycl::access::target::local>
                  l_key_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
              sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                             sycl::access::target::local>
                  l_val_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
              auto d_dstKey_acc_ct0 =
                  d_dstKey_buf_ct0.first
                      .get_access<sycl::access::mode::read_write>(cgh);
              auto d_dstVal_acc_ct1 =
                  d_dstVal_buf_ct1.first
                      .get_access<sycl::access::mode::read_write>(cgh);
              auto d_dstKey_acc_ct2 =
                  d_dstKey_buf_ct2.first
                      .get_access<sycl::access::mode::read_write>(cgh);
              auto d_dstVal_acc_ct3 =
                  d_dstVal_buf_ct3.first
                      .get_access<sycl::access::mode::read_write>(cgh);

              cgh.parallel_for(sycl::nd_range<3>(bml_gws * bml_lws, bml_lws),
                               [=](sycl::nd_item<3> item_ct1) {
                                 unsigned int *d_dstKey_ct0 =
                                     (unsigned int *)(&d_dstKey_acc_ct0[0] +
                                                      d_dstKey_offset_ct0);
                                 unsigned int *d_dstVal_ct1 =
                                     (unsigned int *)(&d_dstVal_acc_ct1[0] +
                                                      d_dstVal_offset_ct1);
                                 unsigned int *d_dstKey_ct2 =
                                     (unsigned int *)(&d_dstKey_acc_ct2[0] +
                                                      d_dstKey_offset_ct2);
                                 unsigned int *d_dstVal_ct3 =
                                     (unsigned int *)(&d_dstVal_acc_ct3[0] +
                                                      d_dstVal_offset_ct3);
                                 bitonicMergeLocal(
                                     d_dstKey_ct0, d_dstVal_ct1, d_dstKey_ct2,
                                     d_dstVal_ct3, arrayLength, size, stride,
                                     dir, item_ct1, l_key_acc_ct1.get_pointer(),
                                     l_val_acc_ct1.get_pointer());
                               });
            });
          }
          break;
        }
      }
    }
  }
}
