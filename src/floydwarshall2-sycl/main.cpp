/*
ECL-APSP code: ECL-APSP is an all-pairs-shortest-paths CUDA implementation of
the Floyd-Warshall algorithm that is quite fast. It operates on graphs stored
in binary CSR format.

Copyright (c) 2021, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Yiqian Liu and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/ECL-APSP/.
*/

#include <cstdio>
#include <limits>
#include <sys/time.h>
#include <sycl/sycl.hpp>
#include "graph.h"

using mtype = int;

static const int ws = 32; // warp size
static const int tile = 64; // tile size
static const int TPB = ws * ws; // threads per block is fixed at 1024

// initialize adj matrix
static void init1(const int nodes, mtype* const AdjMat, const int upper,
                  sycl::nd_item<1> &item)
{
  const int idx = item.get_global_id(0);
  const int i = idx / upper;
  if (i < upper) {
    const int j = idx % upper;
    AdjMat[idx] = ((i == j) && (i < nodes)) ? 0 : (INT_MAX / 2);
  }
}

// add edges to adj matrix
static void init2(const ECLgraph g, mtype* const AdjMat, const int upper,
                  sycl::nd_item<1> &item)
{
  const int i = item.get_global_id(0);
  if (i < g.nodes) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      AdjMat[i * upper + nei] = g.eweight[j];
    }
  }
}

// calculate the first tile
static 
void FW0_64(mtype* const __restrict AdjMat, const int upper, 
            mtype* const __restrict krows, mtype* const __restrict kcols,
            sycl::nd_item<1> &item, mtype *__restrict temp, mtype *__restrict krow)
{
  const int warp_a = item.get_local_id(0) / ws; // i: 0-31, upper half
  const int warp_b = warp_a + ws; // i: 32-63, lower half
  const int lane_a = item.get_local_id(0) % ws; // j: 0-31, left half
  const int lane_b = lane_a + ws; // j: 32-63, right half

  const int idx0_aa = warp_a * upper + lane_a; // upper left
  const int idx0_ab = warp_a * upper + lane_b; // upper right
  const int idx0_ba = warp_b * upper + lane_a; // lower left
  const int idx0_bb = warp_b * upper + lane_b; // lower right

  const int idx1_aa = lane_a * tile + warp_a;
  const int idx1_ab = lane_b * tile + warp_a;
  const int idx1_ba = lane_a * tile + warp_b;
  const int idx1_bb = lane_b * tile + warp_b;

  int idx2_a = lane_a;
  int idx2_b = lane_b;

  mtype ij_aa = AdjMat[idx0_aa];
  mtype ij_ab = AdjMat[idx0_ab];
  mtype ij_ba = AdjMat[idx0_ba];
  mtype ij_bb = AdjMat[idx0_bb];

  auto sg = item.get_sub_group();

  # pragma unroll 64
  for (int k = 0; k < tile; k++) {
    if (warp_a == k) krow[idx2_a] = ij_aa;
    if (warp_a == k) krow[idx2_b] = ij_ab;
    if (warp_b == k) krow[idx2_a] = ij_ba;
    if (warp_b == k) krow[idx2_b] = ij_bb;

    item.barrier(sycl::access::fence_space::local_space);

    mtype ik_a, ik_b;

    if (k < ws) {
      ik_a = sycl::select_from_group(sg, ij_aa, k);
      ik_b = sycl::select_from_group(item.get_sub_group(), ij_ba, k);
    } else {
      ik_a = sycl::select_from_group(item.get_sub_group(), ij_ab, k - ws);
      ik_b = sycl::select_from_group(item.get_sub_group(), ij_bb, k - ws);
    }

    const mtype kr_a = krow[idx2_a];
    const mtype kr_b = krow[idx2_b];

    ij_aa = sycl::min((int)ij_aa, (int)(ik_a + kr_a));
    ij_ab = sycl::min((int)ij_ab, (int)(ik_a + kr_b));
    ij_ba = sycl::min((int)ij_ba, (int)(ik_b + kr_a));
    ij_bb = sycl::min((int)ij_bb, (int)(ik_b + kr_b));

    if (warp_a == k) krows[idx0_aa] = ij_aa;
    if (warp_a == k) krows[idx0_ab] = ij_ab;
    if (warp_b == k) krows[idx0_ba] = ij_ba;
    if (warp_b == k) krows[idx0_bb] = ij_bb;

    if (lane_a == k) temp[idx1_aa] = ij_aa;
    if (lane_a == k) temp[idx1_ba] = ij_ba;
    if (lane_b == k) temp[idx1_ab] = ij_ab;
    if (lane_b == k) temp[idx1_bb] = ij_bb;

    idx2_a += tile;
    idx2_b += tile;
  }

  item.barrier(sycl::access::fence_space::local_space);

  kcols[idx0_aa] = temp[warp_a * tile + lane_a];
  kcols[idx0_ab] = temp[warp_a * tile + lane_b];
  kcols[idx0_ba] = temp[warp_b * tile + lane_a];
  kcols[idx0_bb] = temp[warp_b * tile + lane_b];
  AdjMat[idx0_aa] = ij_aa;
  AdjMat[idx0_ab] = ij_ab;
  AdjMat[idx0_ba] = ij_ba;
  AdjMat[idx0_bb] = ij_bb;
}

// calculate k^th row and col
static 
void FWrowcol_64(mtype* const __restrict AdjMat, const int upper, 
    mtype* const __restrict krows,
    mtype* const __restrict kcols,
    const int x, const int subm1,
    sycl::nd_item<1> &item, mtype *__restrict temp, mtype *__restrict krow)
{

  const int warp_a = item.get_local_id(0) / ws; // i: 0-31, upper half
  const int warp_b = warp_a + ws; // i: 32-63, lower half
  const int lane_a = item.get_local_id(0) % ws; // j: 0-31, left half
  const int lane_b = lane_a + ws; // j: 32-63, right half

  int y = item.get_group(0);

  if (y < subm1) {
    if (y >= x) y++;
    const int i_a = warp_a + x * tile;
    const int i_b = warp_b + x * tile;
    const int j_a = lane_a + y * tile;
    const int j_b = lane_b + y * tile;

    const int idx0_aa = i_a * upper + j_a;
    const int idx0_ab = i_a * upper + j_b;
    const int idx0_ba = i_b * upper + j_a;
    const int idx0_bb = i_b * upper + j_b;

    int idx1_a = warp_a;
    int idx1_b = warp_b;
    int idx2_a = lane_a;
    int idx2_b = lane_b;

    temp[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile];
    temp[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile];
    temp[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile];
    temp[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile];

    item.barrier(sycl::access::fence_space::local_space);

    mtype ij_aa = AdjMat[idx0_aa];
    mtype ij_ab = AdjMat[idx0_ab];
    mtype ij_ba = AdjMat[idx0_ba];
    mtype ij_bb = AdjMat[idx0_bb];

    const mtype orig_aa = ij_aa;
    const mtype orig_ab = ij_ab;
    const mtype orig_ba = ij_ba;
    const mtype orig_bb = ij_bb;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      if (warp_a == k) krow[idx2_a] = ij_aa;
      if (warp_a == k) krow[idx2_b] = ij_ab;
      if (warp_b == k) krow[idx2_a] = ij_ba;
      if (warp_b == k) krow[idx2_b] = ij_bb;

      item.barrier(sycl::access::fence_space::local_space);

      const mtype ik_a = temp[idx1_a];
      const mtype ik_b = temp[idx1_b];
      const mtype kr_a = krow[idx2_a];
      const mtype kr_b = krow[idx2_b];

      ij_aa = sycl::min((int)ij_aa, (int)(ik_a + kr_a));
      ij_ab = sycl::min((int)ij_ab, (int)(ik_a + kr_b));
      ij_ba = sycl::min((int)ij_ba, (int)(ik_b + kr_a));
      ij_bb = sycl::min((int)ij_bb, (int)(ik_b + kr_b));

      if (warp_a == k) krows[idx0_aa] = ij_aa;
      if (warp_a == k) krows[idx0_ab] = ij_ab;
      if (warp_b == k) krows[idx0_ba] = ij_ba;
      if (warp_b == k) krows[idx0_bb] = ij_bb;

      idx1_a += tile;
      idx1_b += tile;
      idx2_a += tile;
      idx2_b += tile;
    }
    if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
    if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
    if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
    if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
  } else {
    y -= subm1;
    if (y >= x) y++;
    const int i_a = warp_a + y * tile;
    const int i_b = warp_b + y * tile;

    const int j_a = lane_a + x * tile;
    const int j_b = lane_b + x * tile;

    const int idx0_aa = i_a * upper + j_a;
    const int idx0_ab = i_a * upper + j_b;
    const int idx0_ba = i_b * upper + j_a;
    const int idx0_bb = i_b * upper + j_b;

    const int idx1_aa = lane_a * tile + warp_a;
    const int idx1_ab = lane_b * tile + warp_a;
    const int idx1_ba = lane_a * tile + warp_b;
    const int idx1_bb = lane_b * tile + warp_b;

    int idx2_a = (x * tile) * upper + j_a;
    int idx2_b = (x * tile) * upper + j_b;

    mtype ij_aa = AdjMat[idx0_aa];
    mtype ij_ab = AdjMat[idx0_ab];
    mtype ij_ba = AdjMat[idx0_ba];
    mtype ij_bb = AdjMat[idx0_bb];

    const mtype orig_aa = ij_aa;
    const mtype orig_ab = ij_ab;
    const mtype orig_ba = ij_ba;
    const mtype orig_bb = ij_bb;

    auto sg = item.get_sub_group();

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      mtype ik_a, ik_b;
      if (k < ws) {
        ik_a = sycl::select_from_group(sg, ij_aa, k);
        ik_b = sycl::select_from_group(sg, ij_ba, k);
      }
      if (k >= ws) {
        ik_a = sycl::select_from_group(sg, ij_ab, k - ws);
        ik_b = sycl::select_from_group(sg, ij_bb, k - ws);
      }
      const mtype kr_a = krows[idx2_a];
      const mtype kr_b = krows[idx2_b];

      ij_aa = sycl::min((int)ij_aa, (int)(ik_a + kr_a));
      ij_ab = sycl::min((int)ij_ab, (int)(ik_a + kr_b));
      ij_ba = sycl::min((int)ij_ba, (int)(ik_b + kr_a));
      ij_bb = sycl::min((int)ij_bb, (int)(ik_b + kr_b));

      if (lane_a == k) temp[idx1_aa] = ij_aa;
      if (lane_a == k) temp[idx1_ba] = ij_ba;
      if (lane_b == k) temp[idx1_ab] = ij_ab;
      if (lane_b == k) temp[idx1_bb] = ij_bb;

      idx2_a += upper;
      idx2_b += upper;
    }

    item.barrier(sycl::access::fence_space::local_space);

    kcols[idx0_aa] = temp[warp_a * tile + lane_a];
    kcols[idx0_ab] = temp[warp_a * tile + lane_b];
    kcols[idx0_ba] = temp[warp_b * tile + lane_a];
    kcols[idx0_bb] = temp[warp_b * tile + lane_b];

    if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
    if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
    if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
    if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
  }
}

// calculate the remaining tiles
static 
void FWrem_64(mtype* const __restrict AdjMat,
    const int upper, mtype* const __restrict krows,
    mtype* const __restrict kcols, const int x, const int subm1,
    sycl::nd_item<1> &item, mtype *__restrict s_kj, mtype *__restrict s_ik)
{
  int y = item.get_group(0) / subm1;
  int z = item.get_group(0) % subm1;
  if (y >= x) y++;
  if (z >= x) z++;

  const int warp_a = item.get_local_id(0) / ws;
  const int warp_b = warp_a + ws;

  const int lane_a = item.get_local_id(0) % ws;
  const int lane_b = lane_a + ws;

  const int i_a = warp_a + y * tile;
  const int i_b = warp_b + y * tile;
  const int j_a = lane_a + z * tile;
  const int j_b = lane_b + z * tile;

  const int idx0_aa = i_a * upper + j_a; // upper left
  const int idx0_ab = i_a * upper + j_b; // upper right
  const int idx0_ba = i_b * upper + j_a; // lower left
  const int idx0_bb = i_b * upper + j_b; // lower right

  s_kj[warp_a * tile + lane_a] = krows[(x * tile + warp_a) * upper + j_a];
  s_kj[warp_a * tile + lane_b] = krows[(x * tile + warp_a) * upper + j_b];
  s_kj[warp_b * tile + lane_a] = krows[(x * tile + warp_b) * upper + j_a];
  s_kj[warp_b * tile + lane_b] = krows[(x * tile + warp_b) * upper + j_b];

  s_ik[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile];
  s_ik[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile];
  s_ik[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile];
  s_ik[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile];

  mtype ij_aa = AdjMat[idx0_aa];
  mtype ij_ab = AdjMat[idx0_ab];
  mtype ij_ba = AdjMat[idx0_ba];
  mtype ij_bb = AdjMat[idx0_bb];

  const mtype orig_aa = ij_aa;
  const mtype orig_ab = ij_ab;
  const mtype orig_ba = ij_ba;
  const mtype orig_bb = ij_bb;

  item.barrier(sycl::access::fence_space::local_space);

  int idx1_a = warp_a;
  int idx1_b = warp_b;

  int idx2_a = lane_a;
  int idx2_b = lane_b;

  #pragma unroll 64
  for (int k = 0; k < tile; k++) {
    const mtype sk_a = s_kj[idx2_a];
    const mtype sk_b = s_kj[idx2_b];

    ij_aa = sycl::min((int)ij_aa, (int)(s_ik[idx1_a] + sk_a));
    ij_ab = sycl::min((int)ij_ab, (int)(s_ik[idx1_a] + sk_b));
    ij_ba = sycl::min((int)ij_ba, (int)(s_ik[idx1_b] + sk_a));
    ij_bb = sycl::min((int)ij_bb, (int)(s_ik[idx1_b] + sk_b));

    idx1_a += tile;
    idx1_b += tile;

    idx2_a += tile;
    idx2_b += tile;
  }

  if ((y == z) && (y == x + 1) && (x != subm1)) { // the diagonal in next iteration
    const int idx1_aa = lane_a * tile + warp_a;
    const int idx1_ab = lane_b * tile + warp_a;
    const int idx1_ba = lane_a * tile + warp_b;
    const int idx1_bb = lane_b * tile + warp_b;

    int idx2_a = lane_a;
    int idx2_b = lane_b;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      if (warp_a == k) s_kj[idx2_a] = ij_aa;
      if (warp_a == k) s_kj[idx2_b] = ij_ab;
      if (warp_b == k) s_kj[idx2_a] = ij_ba;
      if (warp_b == k) s_kj[idx2_b] = ij_bb;

      item.barrier(sycl::access::fence_space::local_space);

      mtype ik_a, ik_b;
      auto sg = item.get_sub_group(); 
      if (k < ws) {
        ik_a = sycl::select_from_group(sg, ij_aa, k);
        ik_b = sycl::select_from_group(sg, ij_ba, k);
      }
      else {
        ik_a = sycl::select_from_group(sg, ij_ab, k - ws);
        ik_b = sycl::select_from_group(sg, ij_bb, k - ws);
      }

      const mtype sk_a = s_kj[idx2_a];
      const mtype sk_b = s_kj[idx2_b];

      ij_aa = sycl::min((int)ij_aa, (int)(ik_a + sk_a));
      ij_ab = sycl::min((int)ij_ab, (int)(ik_a + sk_b));
      ij_ba = sycl::min((int)ij_ba, (int)(ik_b + sk_a));
      ij_bb = sycl::min((int)ij_bb, (int)(ik_b + sk_b));

      if (warp_a == k) krows[idx0_aa] = ij_aa;
      if (warp_a == k) krows[idx0_ab] = ij_ab;
      if (warp_b == k) krows[idx0_ba] = ij_ba;
      if (warp_b == k) krows[idx0_bb] = ij_bb;

      if (lane_a == k) s_ik[idx1_aa] = ij_aa;
      if (lane_a == k) s_ik[idx1_ba] = ij_ba;
      if (lane_b == k) s_ik[idx1_ab] = ij_ab;
      if (lane_b == k) s_ik[idx1_bb] = ij_bb;
      idx2_a += tile;
      idx2_b += tile;
    }

    item.barrier(sycl::access::fence_space::local_space);

    kcols[idx0_aa] = s_ik[warp_a * tile + lane_a];
    kcols[idx0_ab] = s_ik[warp_a * tile + lane_b];
    kcols[idx0_ba] = s_ik[warp_b * tile + lane_a];
    kcols[idx0_bb] = s_ik[warp_b * tile + lane_b];
  }

  if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
  if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
  if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
  if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
}

static void FW_gpu_64(const ECLgraph g, mtype *const AdjMat, const int repeat) {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (wgs < TPB) {
    printf("The max work group size supported is less than %d required for the program\n", TPB);
    return;
  }

  // copy graph to GPU
  ECLgraph d_g = g;
  d_g.nindex = sycl::malloc_device<int>((g.nodes + 1), q);
  q.memcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1));

  d_g.nlist = sycl::malloc_device<int>(g.edges, q);
  q.memcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges);

  d_g.eweight = sycl::malloc_device<int>(g.edges, q);
  q.memcpy(d_g.eweight, g.eweight, sizeof(int) * g.edges);

  // allocate GPU memory
  const int sub = (g.nodes + tile - 1) / tile;

  const int upper = sub * tile; // upper bound of the GPU matrix
  mtype* d_AdjMat = sycl::malloc_device<mtype>(upper * upper, q);

  mtype* d_krows = sycl::malloc_device<mtype>(upper * upper, q);

  mtype* d_kcols = sycl::malloc_device<mtype>(upper * upper, q);

  printf("GPU matrix size: %.1f MB\n", sizeof(mtype) * upper * upper / (1024.0 * 1024.0));

  timeval start, end;

  sycl::range<1> init_gws ((upper*upper+TPB-1)/TPB*TPB);
  sycl::range<1> lws (TPB);
  sycl::range<1> init2_gws ((g.nodes+TPB-1)/TPB*TPB);

  gettimeofday(&start, NULL);

  q.wait();

  for (int i = 0; i < repeat; i++) {
    // run GPU init code
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(init_gws, lws), [=](sycl::nd_item<1> item) {
        init1(g.nodes, d_AdjMat, upper, item);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(init2_gws, lws), [=](sycl::nd_item<1> item) {
        init2(d_g, d_AdjMat, upper, item);
      });
    });
  }
  q.wait();
  gettimeofday(&end, NULL);
  const double inittime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("Average kernel (initialization) time: %10.6f s\n", inittime / repeat);

  const int subm1 = sub - 1;
  gettimeofday(&start, NULL);

  for (int i = 0; i < repeat; i++) {
    // compute 64*64 tile
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<mtype, 1> temp(sycl::range<1>(tile * tile), cgh);
      sycl::local_accessor<mtype, 1> krow(sycl::range<1>(tile * tile), cgh);
      cgh.parallel_for(sycl::nd_range<1>(lws, lws),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            FW0_64(d_AdjMat, upper, d_krows, d_kcols, item,
                   temp.get_pointer(), krow.get_pointer());
      });
    });

    if (sub > 1) {
      sycl::range<1> fw64_gws (2 * subm1 * TPB);
      sycl::range<1> fw64r_gws (subm1 * subm1 * TPB);

      for (int x = 0; x < sub; x++) {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<mtype, 1> temp(sycl::range<1>(tile * tile), cgh);
          sycl::local_accessor<mtype, 1> krow(sycl::range<1>(tile * tile), cgh);
          cgh.parallel_for(sycl::nd_range<1>(fw64_gws, lws),
              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                FWrowcol_64(d_AdjMat, upper, d_krows, d_kcols, x, subm1, item,
                            temp.get_pointer(),
                            krow.get_pointer());
          });
        });

        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<mtype, 1> s_kj(sycl::range<1>(tile * tile), cgh);
          sycl::local_accessor<mtype, 1> s_ik(sycl::range<1>(tile * tile), cgh);
          cgh.parallel_for(
              sycl::nd_range<1>(fw64r_gws, lws),
              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                FWrem_64(d_AdjMat, upper, d_krows, d_kcols, x, subm1, item,
                         s_kj.get_pointer(), s_ik.get_pointer());
          });
        });
      }
    }
  }
  q.wait();
  gettimeofday(&end, NULL);
  const double comptime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("Average kernel (compute) time: %10.6f s\n", comptime / repeat);

  // copy result back to CPU
  q.memcpy(AdjMat, d_AdjMat, sizeof(mtype) * upper * upper).wait();

  // clean up
  sycl::free(d_g.nindex, q);
  sycl::free(d_g.nlist, q);
  sycl::free(d_g.eweight, q);
  sycl::free(d_AdjMat, q);
  sycl::free(d_krows, q);
  sycl::free(d_kcols, q);
}

static void FW_cpu(const ECLgraph g, mtype* const AdjMat)
{
  timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < g.nodes; i++) {
    for (int j = 0; j < g.nodes; j++) {
      AdjMat[i * g.nodes + j] = ((i == j) ? 0 : (INT_MAX / 2));
    }
  }

  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      AdjMat[i * g.nodes + nei] = g.eweight[j];
    }
  }

  gettimeofday(&end, NULL);
  const double inittime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("CPU init time: %10.6f s\n", inittime);

  gettimeofday(&start, NULL);

  for (int k = 0; k < g.nodes; k++) {
    for (int i = 0; i < g.nodes; i++) {
      for (int j = 0; j < g.nodes; j++) {
        if (AdjMat[i * g.nodes + j] > AdjMat[i * g.nodes + k] + AdjMat[k * g.nodes + j]) {
          AdjMat[i * g.nodes + j] = AdjMat[i * g.nodes + k] + AdjMat[k * g.nodes + j];
        }
      }
    }
  }

  gettimeofday(&end, NULL);
  const double comptime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("CPU comp time: %10.6f s\n", comptime);
}

int main(int argc, char* argv[])
{
  printf("ECL-APSP v1.0 (%s)\n", __FILE__);
  printf("Copyright 2021 Texas State University\n");
  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <input_graph_name> <repeat>\n\n", argv[0]);
    return 1;
  }
  if (TPB != 1024) {
    fprintf(stderr, "Work-group size must be 1024\n\n");
    return 1;
  }

  // allocation of matrices may fail on a host
  mtype* AdjMat1 = NULL;
  mtype* AdjMat2 = NULL;

  // declare them before the goto statement
  int upper_64;
  int diffcount;
  int gn;

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  const int repeat = atoi(argv[2]);

  if (g.eweight == NULL) {
    fprintf(stderr, "ERROR: input graph has no edge weights\n\n");
    goto DONE;
  }

  // make all weights positive to avoid negative cycles
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
    }
  }

  // run on device
  upper_64 = ((g.nodes + tile - 1) / tile) * tile;  // round up
  AdjMat1 = (mtype*) malloc (sizeof(mtype) * upper_64 * upper_64);
  if (AdjMat1 == NULL) {
    fprintf(stderr, "ERROR: memory allocation (AdjMat1) fails\n\n");
    goto DONE;
  }

  FW_gpu_64(g, AdjMat1, repeat);

  // run on host
  AdjMat2 = (mtype*) malloc (sizeof(mtype) * g.nodes * g.nodes);
  if (AdjMat2 == NULL) {
    fprintf(stderr, "ERROR: memory allocation (AdjMat2) fails\n\n");
    goto DONE;
  }

  FW_cpu(g, AdjMat2);

  // compare results
  diffcount = 0;
  gn = g.nodes;
  for (int i = 0; i < gn; ++i) {
    for (int j = 0; j < gn; ++j) {
      if (AdjMat1[i * upper_64 + j] != AdjMat2[i * g.nodes + j]) {
        diffcount++;
      }
    }
  }

  if (diffcount > 0) {
    printf("ERROR: results differ in %d places!\n", diffcount);
  } else {
    printf("results match\n");
  }

  DONE:
  // clean up
  if (AdjMat1) free(AdjMat1);
  if (AdjMat2) free(AdjMat2);
  freeECLgraph(g);
  return 0;
}
