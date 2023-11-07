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
#include <sycl/sycl.hpp>


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

#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)
#include "rand_gen.cpp"
#include "petri_kernel.cpp"

static int N, S, T, NSQUARE2;
uint32 host_mt[MERS_N];

void PetrinetOnDevice(long long &time);
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
      printf("Usage: %s N S T\n", argv[0]);
      printf("N: the place-transition grid is 2nX2n\n"
             "S: the maximum steps in a trajectory\n"
             "T: number of trajectories\n");
      return -1;
    }

  N = atoi(argv[1]);
  if (N<1)
    return -1;
  S = atoi(argv[2]);
  if (S<1)
    return -1;

  T = atoi(argv[3]);
  if (T<1)
    return -1;

  NSQUARE2 = N*(N+N);

  h_vars = (float*)malloc(T*sizeof(float));
  h_maxs = (int*)malloc(T*sizeof(int));

  long long ktime = 0;

  auto start = get_time();

  PetrinetOnDevice(ktime);

  auto end = get_time();

  printf("Total kernel execution time: %.2f s\n", ktime / 1e6f);
  printf("Total device execution time: %.2f s\n", (end - start) / 1e6f);

  compute_statistics();

  free(h_vars);
  free(h_maxs);

  printf("petri N=%d S=%d T=%d\n", N, S, T);
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
  for (i=0; i<T; i++)
    {
      sum += h_vars[i];
      sum_vars += h_vars[i]*h_vars[i];
      sum_max += h_maxs[i];
      sum_max_vars += h_maxs[i]*h_maxs[i];
    }
  results[0] = sum/T;
  results[1] = sum_vars/T - results[0]*results[0];
  results[2] = sum_max/T;
  results[3] = sum_max_vars/T - results[2]*results[2];
}

void PetrinetOnDevice(long long &time)
{
  // Allocate memory
  int i;
  int unit_size = NSQUARE2*(sizeof(int)+sizeof(char))+sizeof(float)+sizeof(int);
  int block_num = MAX_DEVICE_MEM/unit_size;

  printf("Number of thread blocks: %d\n", block_num);

  const int g_places_size = (unit_size - sizeof(float) - sizeof(int))*block_num / sizeof(int);

  // compute the simulation on the GPU
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    int *g_places = sycl::malloc_device<int>(g_places_size, q);
  float *g_vars = sycl::malloc_device<float>(block_num, q);
    int *g_maxs = sycl::malloc_device<int>(block_num, q);

  // Setup the execution configuration
  sycl::range<1> gws (256 * block_num);
  sycl::range<1> lws (256);  // each block has 256 threads

  int *p_hmaxs = h_maxs;
  float *p_hvars = h_vars;

  const int n = N;
  const int s = S;

  // Launch the device computation threads!
  for (i = 0; i<T-block_num; i+=block_num)
  {
    q.wait();
    auto start = get_time();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<uint32, 1> mt(sycl::range<1>(MERS_N), cgh);
      cgh.parallel_for<class pn_loop>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        PetrinetKernel(
          item,
          mt.get_pointer(),
          g_places,
          g_vars,
          g_maxs,
          n, s, 5489*(i+1));
      });
    }).wait();

    auto end = get_time();
    time += end - start;

    q.memcpy(p_hmaxs, g_maxs, block_num*sizeof(int));
    q.memcpy(p_hvars, g_vars, block_num*sizeof(float));

    p_hmaxs += block_num;
    p_hvars += block_num;
  }

  sycl::range<1> gws1 (256*(T-i));

  q.wait();
  auto start = get_time();

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint32, 1> mt(sycl::range<1>(MERS_N), cgh);
    cgh.parallel_for<class pn_final>(
      sycl::nd_range<1>(gws1, lws), [=] (sycl::nd_item<1> item) {
      PetrinetKernel(
        item,
        mt.get_pointer(),
        g_places,
        g_vars,
        g_maxs,
        n, s, 5489*(i+1));
    });
  }).wait();

  auto end = get_time();
  time += end - start;

  // Read result from the device
  q.memcpy(p_hmaxs, g_maxs, (T-i)*sizeof(int));
  q.memcpy(p_hvars, g_vars, (T-i)*sizeof(float));
  q.wait();

  sycl::free(g_places, q);
  sycl::free(g_vars, q);
  sycl::free(g_maxs, q);
}
