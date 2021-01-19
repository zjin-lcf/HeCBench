#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "constants.h"
#include "common.h"

#define N_THREADS_PER_BLOCK 256

void find_min_max_kernel( queue &q,
    const float *__restrict__ u, llint u_size, float *__restrict__ min_u, float *__restrict__ max_u
    ) {
  llint u_block = u_size / N_THREADS_PER_BLOCK;
  llint u_remainder = u_size % N_THREADS_PER_BLOCK;

  llint d_block = u_block;
  if (u_remainder != 0) { d_block += 1; }
  llint d_size = d_block * N_THREADS_PER_BLOCK;

  llint reminder_size = N_THREADS_PER_BLOCK - u_remainder;
  float *reminder = (float *)malloc(reminder_size * sizeof(float));
  memcpy(reminder, u, reminder_size * sizeof(float));

  float *h_max = (float*)malloc(d_block * sizeof(float));
  float *h_min = (float*)malloc(d_block * sizeof(float));

  buffer<float, 1> d_u (d_size);
  buffer<float, 1> d_max (d_block);
  buffer<float, 1> d_min (d_block);


  //cudaMemcpy(d_u, u, u_size * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_u+u_size, reminder, reminder_size * sizeof(float), cudaMemcpyHostToDevice);
  //find_min_max_u_kernel<<<d_block, N_THREADS_PER_BLOCK, sizeof(float) * N_THREADS_PER_BLOCK>>>(d_u, d_max, d_min);
  //cudaMemcpy(max, d_max, d_block * sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(min, d_min, d_block * sizeof(float), cudaMemcpyDeviceToHost);

  q.submit([&] (handler &h) {
      auto du = d_u.get_access<sycl_write>(h, range<1>(u_size));
      h.copy(u, du);
      });
  q.submit([&] (handler &h) {
      auto du = d_u.get_access<sycl_write>(h, range<1>(reminder_size), id<1>(u_size));
      h.copy(reminder, du);
      });
  q.submit([&] (handler &h) {
      auto g_u = d_u.get_access<sycl_read>(h);
      auto g_max = d_max.get_access<sycl_discard_write>(h);
      auto g_min = d_min.get_access<sycl_discard_write>(h);
      accessor <float, 1, sycl_read_write, access::target::local> sdata (N_THREADS_PER_BLOCK, h);
      h.parallel_for<class find_min_max>(nd_range<1>(d_size, N_THREADS_PER_BLOCK), [=] (nd_item<1> item) {
          unsigned int tid = item.get_local_id(0);
          unsigned int gid = item.get_group(0);
          unsigned int tidFromBack = item.get_local_range(0) - 1 - tid;
          unsigned int i = item.get_global_id(0);
          sdata[tid] = g_u[i];
          item.barrier(access::fence_space::local_space);

          for (unsigned int s = item.get_local_range(0)/2; s > 0; s >>= 1) {
            if (tid < s) {
              if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
              }
            }
            if (tidFromBack < s) {
              if (sdata[tid - s] < sdata[tid]) {
                sdata[tid] = sdata[tid - s];
              }
            }
            item.barrier(access::fence_space::local_space);
          }

          if (tid == 0) {
            g_max[gid] = sdata[0];
          }
          if (tidFromBack == 0) {
            g_min[gid] = sdata[tid];
          }
      });
  });
  q.submit([&] (handler &h) {
      auto dm = d_max.get_access<sycl_read>(h);
      h.copy(dm, h_max);
  });
  q.submit([&] (handler &h) {
      auto dm = d_min.get_access<sycl_read>(h);
      h.copy(dm, h_min);
  });
  q.wait();

  *min_u = FLT_MAX, *max_u = FLT_MIN;
  for (size_t i = 0; i < d_block; i++) {
    *min_u = std::fminf(*min_u, h_min[i]);
    *max_u = std::fmaxf(*max_u, h_max[i]);
  }

  free(reminder);
  free(h_max);
  free(h_min);
}
