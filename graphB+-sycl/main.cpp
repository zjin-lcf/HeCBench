/*
graphB+ balancing algorithm for signed social network graphs

Copyright 2021, Texas State University. All rights reserved.

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

Authors: Ghadeer Alabandi and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/graphB/.
*/

#include <cstdio>
#include <chrono>
#include <climits>
#include <algorithm>
#include <set>
#include <map>
#include <sycl/sycl.hpp>
#include "kernels.h"

int main(int argc, char* argv[])
{
  printf("graphB+ balancing code for signed social network graphs (%s)\n", __FILE__);
  printf("Copyright 2021 Texas State University\n");

  // process command line and read input
  if (argc != 4) {printf("USAGE: %s input_file_name iteration_count output_file_name\n", argv[0]); exit(-1);}

#ifdef VERIFY
  printf("verification is on\n");
#else
  printf("verification is off\n");
#endif
  printf("input: %s\n", argv[1]);
  Graph g = readGraph(argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int iterations = atoi(argv[2]);

  // allocate all memory
  bool* const minus = new bool [g.edges];
  int* const parent = new int [g.nodes];
  int* const queue = new int [g.nodes];  // first used as queue, then as CC size
  int* const label = new int [g.nodes];  // first used as count, then as label, and finally as CC label
  int* const border = new int [g.nodes + 2];  // maybe make smaller
  int* const inCC = new int [g.nodes];  // how often node was in largest CC or at an even distance from largest CC
  int* const inTree = new int [g.edges];  // how often edge was in tree
  int* const negCnt = new int [g.edges];  // how often edge was negative
  EdgeInfo* const einfo = new EdgeInfo [g.edges + 16];
  int* const root = new int [g.nodes];  // tree roots

  for (int i = 0; i < g.nodes; i++) root[i] = i;
  std::partial_sort(root, root + std::min(iterations, g.nodes), root + g.nodes, [&](int a, int b) {
    return (g.nindex[a + 1] - g.nindex[a]) > (g.nindex[b + 1] - g.nindex[b]);
  });

  //GPU code
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto dev = q.get_device();
  auto devName = dev.get_info<sycl::info::device::name>();
  auto SMs = dev.get_info<sycl::info::device::max_compute_units>();
  auto mTpSM = 2048;
  auto clockRate = dev.get_info<sycl::info::device::max_clock_frequency>();
  auto memClockRate = dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
  printf("Device: %s with %d SMs and %d mTpSM (%u MHz and %u MHz)\n",
         devName.c_str(), SMs, mTpSM, clockRate, memClockRate);

  Graph d_g = g;
  EdgeInfo* d_einfo;
  int* d_label;
  int* d_parent;
  int* d_queue;
  int* d_border;
  int* d_tail;
  int* d_inCC;
  int* d_inTree;
  int* d_negCnt;
  int* d_ws1;
  int* d_ws2;
  unsigned long long* d_hi;
  int* d_wSize;
  bool* d_minus;
  bool* changed_gpu;

  if (nullptr == (d_g.eweight = sycl::malloc_device<int>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_g.nindex = sycl::malloc_device<int>((g.nodes + 1), q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_g.nlist = sycl::malloc_device<int>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_inTree = sycl::malloc_device<int>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_negCnt = sycl::malloc_device<int>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_inCC = sycl::malloc_device<int>(g.nodes, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_einfo = sycl::malloc_device<EdgeInfo>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_label = sycl::malloc_device<int>(g.nodes, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_parent = sycl::malloc_device<int>(g.nodes, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_border = sycl::malloc_device<int>((g.nodes + 2), q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_queue = sycl::malloc_device<int>(g.nodes, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_tail = sycl::malloc_device<int>(1, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_ws1 = sycl::malloc_device<int>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_ws2 = sycl::malloc_device<int>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_wSize = sycl::malloc_device<int>(1, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_minus = sycl::malloc_device<bool>(g.edges, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (changed_gpu = sycl::malloc_device<bool>(1, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (nullptr == (d_hi = sycl::malloc_device<unsigned long long>(1, q)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  q.memcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1)).wait();

  q.memcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges).wait();

  q.memcpy(d_g.eweight, g.eweight, sizeof(int) * g.edges).wait();

  const int blocks = SMs * mTpSM / ThreadsPerBlock;

  sycl::range<1> gws (blocks * ThreadsPerBlock);
  sycl::range<1> lws (ThreadsPerBlock);

  // use random pluses and minuses
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class init_1>(
    sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
    init(g.edges, g.nodes, d_g.nlist, d_g.eweight, d_inCC, d_einfo,
         d_inTree, d_negCnt, item);
    });
  }).wait();

  int min_d = INT_MAX;
  int max_d = INT_MIN;
  int sum_d = 0;
  double avg_d = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iterations; iter++) {
    // generate tree
    q.submit([&](sycl::handler &cgh) {
      auto root_node = root[iter % g.nodes];
      cgh.parallel_for<class init_2>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        init2(g.edges, g.nodes, root_node, d_g.nlist, d_parent,
              d_queue, d_label, d_tail, item);
      });
    });
    int level = 0;
    int tail = 1;
    border[0] = 0;
    border[1] = tail;
    while (border[level + 1] < g.nodes) {
      q.submit([&](sycl::handler &cgh) {
        auto border_level_cur = border[level];
        auto border_level_nxt = border[level + 1];
        cgh.parallel_for<class genSpanTree>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          generateSpanningTree(g.nodes, d_g.nindex, d_g.nlist, iter + 17,
                               d_einfo, d_parent, d_queue, level, d_tail,
                               border_level_cur, border_level_nxt,
                               item);
        });
      });

      q.memcpy(&tail, d_tail, sizeof(int)).wait();

      level++;
      border[level + 1] = tail;
    }

    const int levels = level + 1;

    //min , max and avg depth of the trees
    sum_d += level;
    if (level < min_d) min_d = level;
    if (level > max_d) max_d = level;

#ifdef VERIFY
    q.submit([&](sycl::handler &cgh) {
      sycl::stream stream_out(64 * 1024, 80, cgh);
      auto border_level = border[level + 1];
      cgh.parallel_for<class verify_spanTree>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        verify_generateSpanningTree(g.nodes, g.edges, d_g.nindex, d_g.nlist,
                                    iter, d_parent, level, d_tail,
                                    border_level, item, stream_out);
      });
    });
#endif
    //root count
    //#1

    for (int level = levels - 1; level > 0; level--) {
      q.submit([&](sycl::handler &cgh) {
        auto border_level_cur = border[level];
        auto border_level_nxt = border[level + 1];
        cgh.parallel_for<class rootCount>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          rootcount(d_parent, d_queue, d_label, level, border_level_cur,
                    border_level_nxt, item);
        });
      });
    }

#ifdef VERIFY
    q.memcpy((void *)&label[root[iter % g.nodes]],
             (void *)&d_label[root[iter % g.nodes]], sizeof(int)).wait();

    if (label[root[iter % g.nodes]] != g.nodes)
      printf("ERROR: root count mismatch\n");
#endif

    // tree label
    label[root[iter % g.nodes]] = 0;
    q.memset((void *)&d_label[root[iter % g.nodes]], 0, sizeof(int)).wait();

    //#2
    for (int level = 0; level < levels; level++) {
      q.submit([&](sycl::handler &cgh) {
#ifdef VERIFY
        sycl::stream stream_out(64 * 1024, 80, cgh);
#endif
        auto border_level_cur = border[level];
        auto border_level_nxt = border[level + 1];

        cgh.parallel_for<class treeLabel>(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) 
           [[intel::reqd_sub_group_size(32)]] {
          treelabel(g.nodes, d_g.nindex, d_g.nlist, d_einfo, d_inTree,
                    d_negCnt, d_parent, d_queue, d_label, level,
                    border_level_cur, border_level_nxt, item
#ifdef VERIFY
                    , stream_out
#endif
                   );
          });
      });
    }

    //#3
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class treeUpdate>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        inTreeUpdate(g.edges, d_g.nlist, d_inTree, item);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class initM>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        initMinus(g.edges, g.nodes, d_g.nindex, d_g.nlist, d_einfo, d_minus,
                  item);
      });
    });

    //#4
    q.submit([&](sycl::handler &cgh) {
#ifdef VERIFY
      sycl::stream stream_out(64 * 1024, 80, cgh);
#endif
      cgh.parallel_for<class pCyles>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        processCycles(g.nodes, d_g.nindex, d_g.nlist, d_label, d_einfo,
                      d_minus, item
#ifdef VERIFY
                      , stream_out
#endif
                     );
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class init_3>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        init3(g.nodes, d_g.nindex, d_g.nlist, d_label, d_queue, item);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class computeOne>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        compute1(g.nodes, d_g.nindex, d_g.nlist, d_label, d_minus, d_negCnt, item);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class flat>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        flatten(g.nodes, d_label, item);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class sizeCC>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        ccSize(g.nodes, d_label, d_queue, item, d_hi, d_wSize);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class maxCC>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        largestCC(g.nodes, d_queue, item, d_hi);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class hopCountCC>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        ccHopCount(g.nodes, d_g.nindex, d_g.nlist, d_label, d_queue, d_ws1,
                   d_ws2, item, d_hi, d_wSize);
      });
    });

    bool changed;
    do {
      changed = false;
      q.memset(changed_gpu, 0, sizeof(bool)).wait();

      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class BF1>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          BellmanFord(d_queue, changed_gpu, d_ws1, d_ws2, item, d_wSize);
        });
      });

      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class BF2>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          BellmanFord(d_queue, changed_gpu, d_ws1, d_ws2, item, d_wSize);
        });
      });

      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class BF3>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          BellmanFord(d_queue, changed_gpu, d_ws1, d_ws2, item, d_wSize);
        });
      });

      q.memcpy(&changed, changed_gpu, sizeof(bool)).wait();

    } while (changed);

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class incrCC>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        incrementCC(g.nodes, d_label, d_queue, d_inCC, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = elapsed_seconds.count();
  printf("Total graphB+ runtime:    %.6f s\n", runtime);

  q.memcpy(inCC, d_inCC, sizeof(int) * g.nodes).wait();


  // print results
  avg_d = sum_d/iterations;
  printf("number of trees %d\n", iterations);
  printf("Min depth of the trees %d\n Max depth of the trees %d\n Avg depth of the trees %.4f\n",min_d, max_d, avg_d);
  for (int i = 0; i < g.nodes; i++) {
    if (i >= 10) break;  // to limit output
    printf("%6d: %6d   (%5.1f%%)  %d\n", i, inCC[i], 100.0 * inCC[i] / iterations, g.origID[i]);
  }
  
  // output results to file
  FILE *f = fopen(argv[3], "wt");
  fprintf(f, "original node ID, percentage node was in agreeable majority\n");
  for (int i = 0; i < g.nodes; i++) {
    fprintf(f, "%d,%.1f\n", g.origID[i], 100.0 * inCC[i] / iterations);
  }
  fprintf(f, "source node ID, destination node ID, percentage edge was in tree, percentage edge was negative\n");
  for (int v = 0; v < g.nodes; v++) {
    for (int j = g.nindex[v]; j < g.nindex[v + 1]; j++) {
      const int n = g.nlist[j] >> 1;
      if (v < n) {  // only print one copy of each edge (other copy does not have correct negCnt)
        fprintf(f, "%d,%d,%.1f,%.1f\n", g.origID[v], g.origID[n], 100.0 * inTree[j] / iterations, 100.0 * negCnt[j] / iterations);
      }
    }
  }
  fclose(f);

  // finalize
  freeGraph(g);
  delete [] minus;
  delete [] einfo;
  delete [] parent;
  delete [] queue;
  delete [] label;
  delete [] border;
  delete [] inCC;
  delete [] inTree;
  delete [] negCnt;
  delete [] root;
  sycl::free(d_g.nlist, q);
  sycl::free(d_g.nindex, q);
  sycl::free(d_einfo, q);
  sycl::free(d_inCC, q);
  sycl::free(d_negCnt, q);
  sycl::free(d_inTree, q);
  sycl::free(d_label, q);
  sycl::free(d_parent, q);
  sycl::free(d_queue, q);
  sycl::free(d_border, q);
  sycl::free(d_tail, q);
  sycl::free(changed_gpu, q);
  sycl::free(d_ws1, q);
  sycl::free(d_ws2, q);
  sycl::free(d_wSize, q);
  sycl::free(d_minus, q);
  sycl::free(d_hi, q);

  return 0;
}
