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
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          l_key_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          l_val_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);

      cgh.parallel_for(sycl::nd_range<3>(bs_gws * bs_lws, bs_lws),
                       [=](sycl::nd_item<3> item_ct1) {
                         bitonicSortLocal(d_dstKey, d_dstVal, d_srcKey,
                                          d_srcVal, arrayLength, dir, item_ct1,
                                          l_key_acc_ct1.get_pointer(),
                                          l_val_acc_ct1.get_pointer());
                       });
    });
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
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          l_key_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          l_val_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);

      cgh.parallel_for(sycl::nd_range<3>(bs1_gws * bs1_lws, bs1_lws),
                       [=](sycl::nd_item<3> item_ct1) {
                         bitonicSortLocal1(d_dstKey, d_dstVal, d_srcKey,
                                           d_srcVal, item_ct1,
                                           l_key_acc_ct1.get_pointer(),
                                           l_val_acc_ct1.get_pointer());
                       });
    });

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
          q_ct1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(bmg_gws * bmg_lws, bmg_lws),
                             [=](sycl::nd_item<3> item_ct1) {
                               bitonicMergeGlobal(d_dstKey, d_dstVal, d_dstKey,
                                                  d_dstVal, arrayLength, size,
                                                  stride, dir, item_ct1);
                             });
          });
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
          q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                l_key_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);
            sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                l_val_acc_ct1(sycl::range<1>(512 /*LOCAL_SIZE_LIMIT*/), cgh);

            cgh.parallel_for(sycl::nd_range<3>(bml_gws * bml_lws, bml_lws),
                             [=](sycl::nd_item<3> item_ct1) {
                               bitonicMergeLocal(d_dstKey, d_dstVal, d_dstKey,
                                                 d_dstVal, arrayLength, size,
                                                 stride, dir, item_ct1,
                                                 l_key_acc_ct1.get_pointer(),
                                                 l_val_acc_ct1.get_pointer());
                             });
          });
          break;
        }
      }
    }
  }
}
