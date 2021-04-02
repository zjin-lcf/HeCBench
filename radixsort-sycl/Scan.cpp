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

#include "Scan.h"
#include "Scan_kernels.cpp"


void scanExclusiveLocal1(
    queue &q,
    buffer<unsigned int> &d_Dst,
    buffer<unsigned int> &d_Src,
    const unsigned int n,
    const unsigned int size
    ){

  size_t localWorkSize = WORKGROUP_SIZE;
  size_t globalWorkSize = (n * size) / 4;

  range<1> gws (globalWorkSize);
  range<1> lws (localWorkSize);

  q.submit([&] (handler &cgh) {
    auto dst = d_Dst.get_access<sycl_discard_write>(cgh);
    auto src = d_Src.get_access<sycl_read>(cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> l_Data(2 * WORKGROUP_SIZE , cgh);
    cgh.parallel_for<class scan_exclusive_local1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      scanExclusiveLocal1K(item, dst.get_pointer(), src.get_pointer(), l_Data.get_pointer(), size);
    });
  });
}

void scanExclusiveLocal2(
    queue &q,
    buffer<unsigned int> &d_Buf,
    buffer<unsigned int> &d_Dst,
    buffer<unsigned int> &d_Src,
    const unsigned int n,
    const unsigned int size)
{
  const unsigned int elements = n * size;
  size_t localWorkSize = WORKGROUP_SIZE;
  size_t globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);
  range<1> gws (globalWorkSize);
  range<1> lws (localWorkSize);


  q.submit([&] (handler &cgh) {
    auto buf = d_Buf.get_access<sycl_discard_write>(cgh);
    auto dst = d_Dst.get_access<sycl_read>(cgh);
    auto src = d_Src.get_access<sycl_read>(cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> l_Data(2 * WORKGROUP_SIZE , cgh);
    cgh.parallel_for<class scan_exclusive_local2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      scanExclusiveLocal2K(item, buf.get_pointer(), dst.get_pointer(), src.get_pointer(), 
                           l_Data.get_pointer(), elements, size);
    });
  });
}

void uniformUpdate(
    queue &q,
    buffer<unsigned int> &d_Dst,
    buffer<unsigned int> &d_Buf,
    const unsigned int n)
{
  range<1> gws (n * WORKGROUP_SIZE);
  range<1> lws (WORKGROUP_SIZE);

  q.submit([&] (handler &cgh) {
    auto d_dst = d_Dst.get_access<sycl_read_write>(cgh);
    auto d_buf = d_Buf.get_access<sycl_read>(cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> buf(1 , cgh);
    cgh.parallel_for<class uniform_update>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      uniformUpdateK(item, d_dst.get_pointer(), d_buf.get_pointer(), buf.get_pointer());
    });
  });
}

// main exclusive scan routine
void scanExclusiveLarge(
    queue &q,
    buffer<unsigned int> &d_Dst,
    buffer<unsigned int> &d_Src,
    buffer<unsigned int> &d_Buf,
    const unsigned int batchSize,
    const unsigned int arrayLength, 
    const unsigned int numElements)
{

  scanExclusiveLocal1(
      q,
      d_Dst,
      d_Src,
      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE),
      4 * WORKGROUP_SIZE);

  scanExclusiveLocal2(
      q,
      d_Buf,
      d_Dst,
      d_Src,
      batchSize,
      arrayLength / (4 * WORKGROUP_SIZE));

  uniformUpdate(
      q,
      d_Dst,
      d_Buf,
      (batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
}
