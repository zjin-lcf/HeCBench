/*
 * (c) 2009-2010 Christoph Schied <Christoph.Schied@uni-ulm.de>
 *
 * This file is part of flame.
 *
 * flame is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * flame is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with flame.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "flame.hpp"

#define VARPARM(a,b) const_mem_params.variation_parameters[a][b]
#define FUNC_COLOR(a) const_mem_params.function_colors[a]
#define FUNC_WEIGHT(a) function_weights[a]
#define AFFINE_PRE(a, b) const_mem_params.pre_transform_params[a][b]
#define AFFINE_POST(a, b) const_mem_params.post_transform_params[a][b]

#include "kernels.dp.cpp"

int main(int argc, char **argv)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  printf("reset parameters..\n");
  ConstMemParams const_mem_params;
  float function_weights[NUM_FUNCTIONS];

  memset(&const_mem_params, 0, sizeof const_mem_params);

  for(int i = 0; i < NUM_FUNCTIONS; i++) {
    function_weights[i] = 0.0f;
    const_mem_params.variation_parameters[i][0].factor = 1.0f;
    const_mem_params.pre_transform_params[i][0] = 1.0f;
    const_mem_params.pre_transform_params[i][4] = 1.0f;
    const_mem_params.post_transform_params[i][0] = 1.0f;
    const_mem_params.post_transform_params[i][4] = 1.0f;
  }

  function_weights[0] = 1.0f;

  AFFINE_PRE(0, 0) = 1.4f;
  AFFINE_PRE(0, 1) = 0.6f;

  AFFINE_PRE(1, 2) = 0.4f;

  AFFINE_PRE(2, 0) = 0.3f;
  AFFINE_PRE(2, 2) = 1.0f;

  FUNC_COLOR(0) = 1.0f;
  FUNC_COLOR(2) = 0.4f;

  FUNC_WEIGHT(1) = 0.5f;
  FUNC_WEIGHT(2) = 0.6f;

  const_mem_params.enable_sierpinski = 1;

  VARPARM(2, 0).idx = 9;
  VARPARM(0, 0).idx = 13;
  VARPARM(1, 0).idx = 5;
  VARPARM(1, 1).idx = 7;

  VARPARM(2, 0).factor = 1.0f;
  VARPARM(0, 0).factor = 1.0f;
  VARPARM(1, 0).factor = 1.0f;
  VARPARM(1, 1).factor = -0.3f;

  AFFINE_PRE(2, 2) = -1.0f;
  AFFINE_PRE(2, 5) = -1.0f;

  srand(2);
  unsigned mersenne_state[624];
  for(int i = 0; i < 624; i++) mersenne_state[i] = rand();
  for(int i = 0; i < 10000; i++) mersenne_twister(mersenne_state);

  printf("generating random numbers\n");
  float *rn_tmp = new float[NUM_RANDOMS];
  for(int i = 0; i < NUM_RANDOMS; i++)
    rn_tmp[i] = mersenne_twister(mersenne_state) * (1.0f / 4294967296.0f);

  float *random_numbers;
  random_numbers = sycl::malloc_device<float>(NUM_RANDOMS, q_ct1);
  q_ct1.memcpy(random_numbers, rn_tmp, NUM_RANDOMS * sizeof(float)).wait();

  delete[] rn_tmp;

  printf("generating permutations\n");
  PermSortElement *to_sort = new PermSortElement[NUM_THREADS];

  unsigned short *perm_data = new unsigned short[NUM_THREADS * NUM_PERMUTATIONS];

  for(int i = 0; i < NUM_PERMUTATIONS; i++) {
    //if(!(i & 127)) printf(".");
    for(int j = 0; j < NUM_THREADS; j++) {
      to_sort[j].value = mersenne_twister(mersenne_state);
      to_sort[j].idx = j;
    }
    std::sort(to_sort, to_sort + NUM_THREADS);
    for(int j = 0; j < NUM_THREADS; j++) {
      perm_data[i * NUM_THREADS + j] = to_sort[j].idx;
    }
  }

  unsigned short *permutations;
  permutations = sycl::malloc_device<unsigned short>(
      NUM_THREADS * NUM_PERMUTATIONS, q_ct1);
  q_ct1
      .memcpy(permutations, perm_data,
              NUM_THREADS * NUM_PERMUTATIONS * sizeof(unsigned short))
      .wait();

  delete[] perm_data;

  sycl::short2 *short_points;
  short_points = sycl::malloc_device<sycl::short2>(NUM_THREADS, q_ct1);

  short *colors;
  colors = sycl::malloc_device<short>(NUM_THREADS, q_ct1);

  sycl::float2 *points_tmp = new sycl::float2[NUM_THREADS];
  for(int i = 0; i < NUM_THREADS; i++) {
    points_tmp[i].x() = (float(i) / NUM_THREADS - 0.5f) * 2.0f;
    points_tmp[i].y() = (radical_inverse(i, 2) - 0.5f) * 2.0f;
  }

  sycl::float2 *start_points;
  start_points = sycl::malloc_device<sycl::float2>(NUM_THREADS, q_ct1);
  q_ct1.memcpy(start_points, points_tmp, NUM_THREADS * sizeof(sycl::float2))
      .wait();

  delete[] points_tmp;

  sycl::float3 *vertices;
  vertices = sycl::malloc_device<sycl::float3>(
      NUM_POINTS_PER_THREAD * NUM_THREADS, q_ct1);

  printf("entering mainloop\n");
  struct timeval tv, tv2;
  gettimeofday(&tv, NULL);
  int perm_pos = 0;
  for (int n = 0; n < 100; n++) { // while (running)
    float sum = 0.0f;
    for(int i = 0; i < NUM_FUNCTIONS; i++)
      sum += function_weights[i];

    int num_threads_sum = 0;
    for(int i = 0; i < NUM_FUNCTIONS; i++) {
      int num_threads = (function_weights[i] / sum) * (NUM_THREADS);
      const_mem_params.thread_function_mapping[i] = num_threads;
      num_threads_sum += num_threads;
    }
    const_mem_params.thread_function_mapping[0] += NUM_THREADS - num_threads_sum;

    for(int i = 1; i < NUM_FUNCTIONS; i++) {
      const_mem_params.thread_function_mapping[i] +=
        const_mem_params.thread_function_mapping[i - 1];
    }

    sycl::range<3> grid(1, 1, NUM_THREADS / THREADS_PER_BLOCK);
    sycl::range<3> block(1, 1, THREADS_PER_BLOCK);

    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                         kernel_initialize(short_points, colors, permutations,
                                           perm_pos, start_points,
                                           random_numbers, const_mem_params,
                                           item_ct1);
                       });
    });
    perm_pos++;
    dev_ct1.queues_wait_and_throw();

    for(int i = 0; i < NUM_ITERATIONS; i++) {
      /*
      DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                           kernel_iterate(short_points, colors, permutations,
                                          perm_pos, random_numbers,
                                          const_mem_params, item_ct1);
                         });
      });
      perm_pos++;
      perm_pos %= NUM_PERMUTATIONS;
      dev_ct1.queues_wait_and_throw();
    }

    /*
    DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                         kernel_generate_points(vertices, short_points, colors,
                                                permutations, perm_pos,
                                                random_numbers,
                                                const_mem_params, item_ct1);
                       });
    });
    perm_pos++;
    dev_ct1.queues_wait_and_throw();

    perm_pos += NUM_POINTS_PER_THREAD - 1;
    perm_pos %= NUM_PERMUTATIONS;
  }
  gettimeofday(&tv2, NULL);
  float frametime = (tv2.tv_sec - tv.tv_sec) * 1000000 + tv2.tv_usec - tv.tv_usec;
  printf("Total frame time is %.1f us\n", frametime);

  // dump vertices
  sycl::float3 *pixels = (sycl::float3 *)malloc(
      NUM_POINTS_PER_THREAD * NUM_THREADS * sizeof(sycl::float3));
  q_ct1
      .memcpy(pixels, vertices,
              NUM_POINTS_PER_THREAD * NUM_THREADS * sizeof(sycl::float3))
      .wait();
  for (int i = 0; i < NUM_POINTS_PER_THREAD * NUM_THREADS; i++)
    printf("%d x=%.1f y=%.1f color=%.1f\n", i, pixels[i].x(), pixels[i].y(),
           pixels[i].z());

  free(pixels);

  sycl::free(start_points, q_ct1);
  sycl::free(short_points, q_ct1);
  sycl::free(colors, q_ct1);
  sycl::free(random_numbers, q_ct1);
  sycl::free(permutations, q_ct1);
  sycl::free(vertices, q_ct1);

  return 0;
}
