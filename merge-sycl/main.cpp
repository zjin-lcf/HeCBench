#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "common.h"
#include "kernels.h"

#define CSV 0
#if(CSV)
#define PS(X, S) std::cout << X << ", " << S << ", "; fflush(stdout);
#define PV(X) std::cout << X << ", "; fflush(stdout);
#else
#define PS(X, S) std::cout << X << " " << S <<" :\n"; fflush(stdout);
#define PV(X) std::cout << "\t" << #X << " \t: " << X << "\n"; fflush(stdout);
#endif

/*
 * Produce 64-bits of pseudo-randomness
 * Note: not very "good" or "random" 
 */
template<typename vec_t>
vec_t rand64() {
  vec_t rtn;
  do {
    uint32_t * rtn32 = (uint32_t *)&rtn;
    rtn32[0] = rand();
    if(sizeof(vec_t) > 4) rtn32[1] = rand();
  } while(!(rtn < getPositiveInfinity<vec_t>() &&
        rtn > getNegativeInfinity<vec_t>()));
  return rtn;
}

template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
class divide;

template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
class merge;

/*
 * Perform <runs> merges of two sorted pseudorandom <vec_t> arrays of length <size> 
 * Checks the output of each merge for correctness
 */
#define PADDING 1024
template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
void mergeType(queue &q, const uint64_t size, const uint32_t runs) {
  // Prepare host and device vectors
  std::vector<vec_t> hA (size + PADDING);
  std::vector<vec_t> hB (size + PADDING);
  std::vector<vec_t> hC (2*size + PADDING);

  buffer<vec_t, 1> dA (size + PADDING);
  buffer<vec_t, 1> dB (size + PADDING);
  buffer<vec_t, 1> dC (2*size + PADDING);

  // diagonal_path_intersections;
  buffer<uint32_t, 1> dpi (2 * (blocks + 1));

  uint32_t errors = 0;

  double total_time = 0.0;

  for(uint32_t r = 0; r < runs; r++) {

    // Generate two sorted psuedorandom arrays
    for (uint64_t n = 0; n < size; n++) {
       hA[n] = rand64<vec_t>();
       hB[n] = rand64<vec_t>();
    }

    for (uint64_t n = size; n < size + PADDING; n++) {
      hA[n] = getPositiveInfinity<vec_t>();
      hB[n] = getPositiveInfinity<vec_t>();
    }

    std::sort(hA.begin(), hA.end());
    std::sort(hB.begin(), hB.end());

    q.submit([&] (handler &cgh) {
      auto acc = dA.template get_access<sycl_discard_write>(cgh);
      cgh.copy(hA.data(), acc);
    });
     
    q.submit([&] (handler &cgh) {
      auto acc = dB.template get_access<sycl_discard_write>(cgh);
      cgh.copy(hB.data(), acc);
    });

    // Perform the global diagonal intersection serach to divide work among SMs
    range<1> gws (blocks * 32);
    range<1> lws (32);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      auto a = dA.template get_access<sycl_read>(cgh);
      auto b = dB.template get_access<sycl_read>(cgh);
      auto d = dpi.template get_access<sycl_discard_write>(cgh);
      accessor<int32_t, 1, sycl_read_write, access::target::local> xt(1, cgh);
      accessor<int32_t, 1, sycl_read_write, access::target::local> yt(1, cgh);
      accessor<int32_t, 1, sycl_read_write, access::target::local> xb(1, cgh);
      accessor<int32_t, 1, sycl_read_write, access::target::local> yb(1, cgh);
      accessor<int32_t, 1, sycl_read_write, access::target::local> found(1, cgh);
      accessor<int32_t, 1, sycl_read_write, access::target::local> oneorzero(32, cgh);
      cgh.parallel_for<class divide<vec_t, blocks, threads, timing>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        workloadDiagonals<vec_t>(
          item, xt.get_pointer(), yt.get_pointer(), xb.get_pointer(), yb.get_pointer(),
          found.get_pointer(), oneorzero.get_pointer(), 
          a.get_pointer(), size, b.get_pointer(), size, d.get_pointer());
      });
    });

    // Merge between global diagonals independently on each block
    range<1> gws2 (blocks * threads);
    range<1> lws2 (threads);
    q.submit([&] (handler &cgh) {
      auto a = dA.template get_access<sycl_read>(cgh);
      auto b = dB.template get_access<sycl_read>(cgh);
      auto c = dC.template get_access<sycl_discard_write>(cgh);
      auto d = dpi.template get_access<sycl_read>(cgh);
      accessor<vec_t, 1, sycl_read_write, access::target::local> A((K+2)<<1, cgh);
      accessor<uint32_t, 1, sycl_read_write, access::target::local> xt(1, cgh);
      accessor<uint32_t, 1, sycl_read_write, access::target::local> yt(1, cgh);
      accessor<uint32_t, 1, sycl_read_write, access::target::local> xs(1, cgh);
      accessor<uint32_t, 1, sycl_read_write, access::target::local> ys(1, cgh);
      cgh.parallel_for<class merge<vec_t, blocks, threads, timing>>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
        mergeSinglePath<vec_t,false,false> (
          item, A.get_pointer(), xt.get_pointer(), yt.get_pointer(),
          xs.get_pointer(), ys.get_pointer(),
          a.get_pointer(), size, b.get_pointer(), size, 
          d.get_pointer(), c.get_pointer(), size * 2);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    // Test for errors
    q.submit([&] (handler &cgh) {
      auto acc = dC.template get_access<sycl_read>(cgh, range<1>(size));
      cgh.copy(acc, hC.data());
    }).wait();

    for(uint32_t i = 1; i < size; i++) {
      errors += hC[i] < hC[i-1];
    }
  }

  if (timing)
    printf("\nAverage kernel execution time: %f (s)\n", (total_time * 1e-9f) / runs);

  // Print error info
  PV(errors);
}

/* 
 * Performs <runs> merge tests for each type at a given size
 */
template<uint32_t blocks, uint32_t threads>
void mergeAllTypes(queue &q, const uint64_t size, const uint32_t runs) {
  // warmup
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, false>(q, size, runs); printf("\n");
  PS("float",    size)  mergeType<float,    blocks, threads, false>(q, size, runs); printf("\n");
  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, false>(q, size, runs); printf("\n");
  PS("double", size)    mergeType<double,   blocks, threads, false>(q, size, runs); printf("\n");

  // timing
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, true>(q, size, runs); printf("\n");
  PS("float",    size)  mergeType<float,    blocks, threads, true>(q, size, runs); printf("\n");
  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, true>(q, size, runs); printf("\n");
  PS("double", size)    mergeType<double,   blocks, threads, true>(q, size, runs); printf("\n");
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length of the arrays> <runs>\n", argv[0]);
    return 1;
  }
  // length is sufficiently large; 
  // otherwise there are invalid global reads in the kernel mergeSinglePath
  const uint64_t length = atol(argv[1]);

  const uint32_t runs = atoi(argv[2]);

  const int blocks = 112;
  const int threads = 128;  // do not change

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  mergeAllTypes<blocks, threads>(q, length, runs);

  return 0;
}

