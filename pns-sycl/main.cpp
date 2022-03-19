/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "common.h"


/*
// Place and Transition are implicitly included in the code
// as the grid is a fixed one
typedef struct {
        float mark;
} Place;

typedef struct {
        int from1, from2;
        int to1, to2;
} Transition;

// this starts from row 0 and col 0
P(r,c)    -> T(r,c)   -> P(r,c+1)  ->
  |            |            |
 \/           \/           \/
T(r+1,c-1)-> P(r+1,c) -> T(r+1,c)  ->
  |            |            |
 \/           \/           \/
P(r+2,c)  -> T(r+2,c) -> P(r+2,c+1)->
  |            |            |
 \/           \/           \/
T(r+3,c-1)-> P(r+3,c) -> T(r+3,c)->
  |            |            |
 \/           \/           \/

*/

#define __syncthreads() item.barrier(access::fence_space::local_space)
#include "rand_gen.cpp"
#include "petri_kernel.cpp"

static int N, s, t, N2, NSQUARE2;
uint32 host_mt[MERS_N];

void PetrinetOnDevice(queue &q);
void compute_statistics();

float results[4];
float* h_vars;
int* h_maxs;

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

int main(int argc, char** argv) 
{
  if (argc<4) 
    {
      printf("Usage: petri n s t\n"
	     "n: the place-transition grid is 2nX2n\n"
	     "s: the maximum steps in a trajectory\n"
	     "t: number of trajectories\n");
      return -1;
    }

  N = atoi(argv[1]);
  if (N<1)
    return -1;
  s = atoi(argv[2]);
  if (s<1)
    return -1;

  t = atoi(argv[3]);
  if (t<1)
    return -1;

  N2 = N+N;
  NSQUARE2 = N*N2;
  
  h_vars = (float*)malloc(t*sizeof(float));
  h_maxs = (int*)malloc(t*sizeof(int));
  
  // compute the simulation on the GPU
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  auto start = get_time();

  PetrinetOnDevice(q);

  auto end = get_time();
  printf("Total device execution time: %.2f s\n", (end - start) / 1e6f);

  compute_statistics();

  free(h_vars);
  free(h_maxs);
    
  printf("petri N=%d s=%d t=%d\n", N, s, t);
  printf("mean_vars: %f    var_vars: %f\n", results[0], results[1]);
  printf("mean_maxs: %f    var_maxs: %f\n", results[2], results[3]);

  return 0;
}

void compute_statistics() 
{
  float sum = 0;
  float sum_vars = 0;
  float sum_max = 0;
  float sum_max_vars = 0;
  int i;
  for (i=0; i<t; i++) 
    {
      sum += h_vars[i];
      sum_vars += h_vars[i]*h_vars[i];
      sum_max += h_maxs[i];
      sum_max_vars += h_maxs[i]*h_maxs[i];
    }
  results[0] = sum/t;
  results[1] = sum_vars/t - results[0]*results[0];
  results[2] = sum_max/t;
  results[3] = sum_max_vars/t - results[2]*results[2];
}

void PetrinetOnDevice(queue &q)
{
  // Allocate memory
  int i;
  int unit_size = NSQUARE2*(sizeof(int)+sizeof(char))+sizeof(float)+sizeof(int);
  int block_num = MAX_DEVICE_MEM/unit_size;

  printf("Number of thread blocks: %d\n", block_num);

  const int g_places_size = (unit_size - sizeof(float) - sizeof(int))*block_num / sizeof(int);

  buffer<int, 1> g_places (g_places_size);
  buffer<float, 1> g_vars (block_num);
  buffer<int, 1> g_maxs (block_num);

  // Setup the execution configuration
  range<1> gws (256 * block_num);
  range<1> lws (256);  // each block has 256 threads

  int *p_hmaxs = h_maxs;
  float *p_hvars = h_vars;

  int Nt = N;
  int st = s;

  // Launch the device computation threads!
  for (i = 0; i<t-block_num; i+=block_num) 
    {
      q.submit([&] (handler &cgh) {
        auto p = g_places.get_access<sycl_read_write>(cgh);
        auto v = g_vars.get_access<sycl_write>(cgh);
        auto m = g_maxs.get_access<sycl_write>(cgh);
        accessor<uint32, 1, sycl_read_write, access::target::local> mt(MERS_N, cgh);
        cgh.parallel_for<class pn_loop>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          PetrinetKernel(item, 
	                 mt.get_pointer(),
	                 p.get_pointer(),
                         v.get_pointer(),
                         m.get_pointer(),
                         Nt, st, 5489*(i+1));
        });
      });

      q.submit([&] (handler &cgh) {
        auto acc = g_maxs.get_access<sycl_read>(cgh);
        cgh.copy(acc, p_hmaxs);
      });

      q.submit([&] (handler &cgh) {
        auto acc = g_vars.get_access<sycl_read>(cgh);
        cgh.copy(acc, p_hvars);
      });

      q.wait();

      p_hmaxs += block_num;
      p_hvars += block_num;
    }
	
  range<1> gws1 (256*(t-i));
  q.submit([&] (handler &cgh) {
    auto p = g_places.get_access<sycl_read_write>(cgh);
    auto v = g_vars.get_access<sycl_write>(cgh);
    auto m = g_maxs.get_access<sycl_write>(cgh);
    accessor<uint32, 1, sycl_read_write, access::target::local> mt(MERS_N, cgh);
    cgh.parallel_for<class pn_final>(nd_range<1>(gws1, lws), [=] (nd_item<1> item) {
      PetrinetKernel(item, 
	             mt.get_pointer(),
                     p.get_pointer(),
                     v.get_pointer(),
                     m.get_pointer(),
                     Nt, st, 5489*(i+1));
    });
  });

  // Read result from the device
  q.submit([&] (handler &cgh) {
    auto acc = g_maxs.get_access<sycl_read>(cgh, range<1>(t-i));
    cgh.copy(acc, p_hmaxs);
  });

  q.submit([&] (handler &cgh) {
    auto acc = g_vars.get_access<sycl_read>(cgh, range<1>(t-i));
    cgh.copy(acc, p_hvars);
  });

  q.wait();
}
