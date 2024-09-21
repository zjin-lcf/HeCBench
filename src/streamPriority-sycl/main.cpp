/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <cstdio>
#include <iostream>
#include <sycl/sycl.hpp>

#define EACH_SIZE 256 * 1024

// # threadblocks
#define TBLOCKS 256
#define THREADS 256

// throw error on equality
#define ERR_EQ(X, Y)                                                           \
  do {                                                                         \
    if ((X) == (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

// throw error on difference
#define ERR_NE(X, Y)                                                           \
  do {                                                                         \
    if ((X) != (Y)) {                                                          \
      fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
    }                                                                          \
  } while (0)

// copy from source -> destination arrays
void memcpy_kernel(sycl::nd_item<1> &item, int *dst, int *src, size_t n, bool wait) {
  int num = item.get_local_range(0) * item.get_group_range(0);
  int id = item.get_global_id(0);

  for (size_t i = id; i < n / sizeof(int); i += num) {
    int v = src[i];
    if (wait) {
      while (v--) 
        dst[i] = v;
    }
    dst[i] = src[i];
  }
}

// initialise memory
void mem_init(int *buf, size_t n) {
  for (size_t i = 0; i < n / sizeof(int); i++) {
    buf[i] = i;
  }
}

long eval (sycl::queue &q, sycl::queue &s1, sycl::queue &s2) {
  size_t size = 1UL << 29;

  // initialise host data
  int *h_src_low;
  int *h_src_hi;
  ERR_EQ(h_src_low = (int *)malloc(size), NULL);
  ERR_EQ(h_src_hi = (int *)malloc(size), NULL);
  mem_init(h_src_low, size);
  mem_init(h_src_hi, size);

  // initialise device data
  int *h_dst_low;
  int *h_dst_hi;
  ERR_EQ(h_dst_low = (int *)malloc(size), NULL);
  ERR_EQ(h_dst_hi = (int *)malloc(size), NULL);
  memset(h_dst_low, 0, size);
  memset(h_dst_hi, 0, size);

  // copy source data -> device
  int *d_src_low = (int*) sycl::malloc_device(size, q);
  int *d_src_hi = (int*) sycl::malloc_device(size, q);
  q.memcpy(d_src_low, h_src_low, size);
  q.memcpy(d_src_hi, h_src_hi, size);

  // allocate memory for memcopy destination
  int *d_dst_low = (int*) sycl::malloc_device(size, q);
  int *d_dst_hi = (int*) sycl::malloc_device(size, q);

  sycl::range<1> gws (TBLOCKS * THREADS);
  sycl::range<1> lws (THREADS);

  // warmup
  for (size_t i = 0; i < size; i += EACH_SIZE) {
    size_t j = i / sizeof(int);
    s1.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          memcpy_kernel(item, d_dst_low + j, d_src_low + j, EACH_SIZE, true);
      });
    });
      
    s2.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          memcpy_kernel(item, d_dst_hi + j, d_src_hi + j, EACH_SIZE, false);
      });
    });
  }
  s1.wait();
  s2.wait();

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < size; i += EACH_SIZE) {
    size_t j = i / sizeof(int);
    s1.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          memcpy_kernel(item, d_dst_low + j, d_src_low + j, EACH_SIZE, true);
      });
    });
      
    s2.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          memcpy_kernel(item, d_dst_hi + j, d_src_hi + j, EACH_SIZE, false);
      });
    });
  }
  s2.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(h_dst_low, d_dst_low, size);
  q.memcpy(h_dst_hi, d_dst_hi, size);
  q.wait();

  // check results of kernels
  ERR_NE(memcmp(h_dst_low, h_src_low, size), 0);
  ERR_NE(memcmp(h_dst_hi, h_src_hi, size), 0);

  sycl::free(d_src_low, q);
  sycl::free(d_src_hi, q);
  sycl::free(d_dst_low, q);
  sycl::free(d_dst_hi, q);
  free(h_src_low);
  free(h_src_hi);
  free(h_dst_low);
  free(h_dst_hi);

  return time;
}

int main(int argc, char **argv) {

  printf("Starting [%s]...\n", argv[0]);

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

  auto device = q.get_device();
  
  sycl::queue s1(device, sycl::property_list{sycl::property::queue::in_order(),
                         sycl::ext::oneapi::property::queue::discard_events(),
                         sycl::ext::oneapi::property::queue::priority_low()});
  sycl::queue s2(device, sycl::property_list{sycl::property::queue::in_order(),
                         sycl::ext::oneapi::property::queue::discard_events(),
                         sycl::ext::oneapi::property::queue::priority_high()});

  auto time = eval(q, s1, s2);
  printf("Elapsed time of kernel launched to high priority stream: %.3lf ms\n", time * 1e-6);

  sycl::queue s3(device, sycl::property_list{sycl::property::queue::in_order(),
                         sycl::ext::oneapi::property::queue::discard_events()});
  sycl::queue s4(device, sycl::property_list{sycl::property::queue::in_order(),
                         sycl::ext::oneapi::property::queue::discard_events()});

  time = eval(q, s3, s4);
  printf("Elapsed time of kernel launched to no-priority stream: %.3lf ms\n", time * 1e-6);

  return 0;
}
