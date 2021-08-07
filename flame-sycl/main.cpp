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


#include <sys/time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "common.h"
#include "flame.hpp"

#define VARPARM(a,b) const_mem_params.variation_parameters[a][b]
#define FUNC_COLOR(a) const_mem_params.function_colors[a]
#define FUNC_WEIGHT(a) function_weights[a]
#define AFFINE_PRE(a, b) const_mem_params.pre_transform_params[a][b]
#define AFFINE_POST(a, b) const_mem_params.post_transform_params[a][b]


#include "kernels.cpp"

int main(int argc, char **argv)
{
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

  float2 *points_tmp = new float2[NUM_THREADS];
  for(int i = 0; i < NUM_THREADS; i++) {
    points_tmp[i].x() = (float(i) / NUM_THREADS - 0.5f) * 2.0f;
    points_tmp[i].y() = (radical_inverse(i, 2) - 0.5f) * 2.0f;
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> random_numbers(rn_tmp, NUM_RANDOMS);
  buffer<unsigned short, 1> permutations (perm_data, NUM_THREADS * NUM_PERMUTATIONS);
  buffer<short2, 1> short_points (NUM_THREADS);
  buffer<short, 1> colors (NUM_THREADS);
  buffer<float2, 1> start_points (points_tmp, NUM_THREADS);
  buffer<float3, 1> vertices (NUM_POINTS_PER_THREAD * NUM_THREADS);

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

    range<1> gws (NUM_THREADS);
    range<1> lws (THREADS_PER_BLOCK);

    q.submit([&] (handler &cgh) {
      auto points_acc = short_points.get_access<sycl_discard_write>(cgh);
      auto colors_acc = colors.get_access<sycl_discard_write>(cgh);
      auto start_points_acc = start_points.get_access<sycl_read>(cgh);
      auto perm_acc = permutations.get_access<sycl_read>(cgh);
      auto rn_acc = random_numbers.get_access<sycl_read>(cgh);
      cgh.parallel_for<class init>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
         kernel_initialize (points_acc.get_pointer(), 
                            colors_acc.get_pointer(), 
                            perm_acc.get_pointer(),
                            perm_pos, 
                            start_points_acc.get_pointer(), 
                            rn_acc.get_pointer(), 
                            const_mem_params, 
                            item);
      });
    });

    q.wait();
    perm_pos++;

    for(int i = 0; i < NUM_ITERATIONS; i++) {
      q.submit([&] (handler &cgh) {
        auto points_acc = short_points.get_access<sycl_read_write>(cgh);
        auto colors_acc = colors.get_access<sycl_read_write>(cgh);
        auto perm_acc = permutations.get_access<sycl_read>(cgh);
        auto rn_acc = random_numbers.get_access<sycl_read>(cgh);
        cgh.parallel_for<class iterate>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
           kernel_iterate (points_acc.get_pointer(), 
                           colors_acc.get_pointer(), 
                           perm_acc.get_pointer(),
                           perm_pos, 
                           rn_acc.get_pointer(),
                           const_mem_params,
                           item);
        });
      });
      q.wait();
      perm_pos++;
      perm_pos %= NUM_PERMUTATIONS;
    }

    q.submit([&] (handler &cgh) {
      auto vertices_acc = vertices.get_access<sycl_discard_write>(cgh);
      auto points_acc = short_points.get_access<sycl_read>(cgh);
      auto colors_acc = colors.get_access<sycl_read>(cgh);
      auto perm_acc = permutations.get_access<sycl_read>(cgh);
      auto rn_acc = random_numbers.get_access<sycl_read>(cgh);
      cgh.parallel_for<class generate>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
         kernel_generate_points (vertices_acc.get_pointer(), 
                                 points_acc.get_pointer(), 
                                 colors_acc.get_pointer(), 
                                 perm_acc.get_pointer(),
                                 perm_pos, 
                                 rn_acc.get_pointer(),
                                 const_mem_params,
                                 item);
      });
    });
    q.wait();
    perm_pos++;
    perm_pos += NUM_POINTS_PER_THREAD - 1;
    perm_pos %= NUM_PERMUTATIONS;
  }
  gettimeofday(&tv2, NULL);
  float frametime = (tv2.tv_sec - tv.tv_sec) * 1000000 + tv2.tv_usec - tv.tv_usec;
  printf("Total frame time is %.1f us\n", frametime);

  // dump vertices
  float3 *pixels = new float3[NUM_POINTS_PER_THREAD * NUM_THREADS];
  q.submit([&] (handler &cgh) {
    auto vertices_acc = vertices.get_access<sycl_read>(cgh);
    cgh.copy(vertices_acc, pixels);
  }).wait();

  for (int i = 0; i < NUM_POINTS_PER_THREAD * NUM_THREADS; i++)
    printf("%d x=%.1f y=%.1f color=%.1f\n", i, pixels[i].x(), pixels[i].y(), pixels[i].z());

  delete[] pixels;
  delete[] rn_tmp;
  delete[] points_tmp;
  delete[] perm_data;
  delete[] to_sort;
  return 0;
}
