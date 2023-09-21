/*---------------------------------------------------------------
  Original author: Zebulun Arendsee
  March 26, 2013
----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <oneapi/mkl/rng/device.hpp>
//#include "oneapi/mkl.hpp"
#include "common.h"

#define PI 3.14159265359f
#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_ADD 256

/* 
   Box-Muller Transformation: Generate one standard normal variable.

   This algorithm can be more efficiently used by producing two
   random normal variables. However, for the CPU, much faster
   algorithms are possible (e.g. the Ziggurat Algorithm);

   This is actually the algorithm chosen by nVidia to calculate
   normal random variables on the GPU.
 */
float rnorm()
{
  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  float V1 = sqrtf(-2 * logf(U1)) * cosf(2 * PI * U2);
  return V1;
}

/*
   Generate random gamma variables on a CPU.
 */
float rgamma(float a, float b)
{
  float x;
  if (a > 1) {
    float d = a - 1.f / 3.f;
    float c = 1.f / sqrtf(9.f * d);
    bool flag = true;
    float V;
    while (flag) {
      float Z = rnorm();
      if (Z > -1.f / c) {
        V = powf(1.f + c * Z, 3.f);
        float U = rand() / (float)RAND_MAX;
        flag = logf(U) > (0.5f * Z * Z + d - d * V + d * logf(V));
      }
    }
    x = d * V / b;
  }
  else 
  {
    x = rgamma(a + 1.f, b);
    x = x * powf(rand() / (float)RAND_MAX, 1.f / a);
  }
  return x;
}

/*
   Metropolis algorithm for producing random a values. 
   The proposal distribution in normal with a variance that
   is adjusted at each step.
 */
float sample_a(float a, float b, int N, float log_sum)
{
  static float sigma = 2;

  float proposal = rnorm() * sigma + a;

  if(proposal <= 0) return a;

  float log_acceptance_ratio = (proposal - a) * log_sum +
    N * (proposal - a) * logf(b) -
    N * (lgamma(proposal) - lgamma(a));

  float U = rand() / float(RAND_MAX);

  if(logf(U) < log_acceptance_ratio){
    sigma *= 1.1;
    return proposal;
  } else {
    sigma /= 1.1;
    return a;
  }
}

/*
   Returns a random b sample (simply a draw from the appropriate
   gamma distribution)
 */
float sample_b(float a, int N, float flat_sum)
{
  float hyperA = N * a + 1;
  float hyperB = flat_sum;
  return rgamma(hyperA, hyperB);
}

/* 
   Generate a single Gamma distributed random variable by the Marsoglia 
   algorithm (George Marsaglia, Wai Wan Tsang; 2001).
 */
float rgamma(oneapi::mkl::rng::device::philox4x32x10<1> *state,
             float a, float b)
{
  float d = a - 1.f / 3.f;
  float c = 1.f / sqrt(9.f * d);
  bool flag = true;
  float V;

  oneapi::mkl::rng::device::gaussian<float> norm_distr;
  oneapi::mkl::rng::device::uniform<float> uniform_distr;
  auto engine = *state;

  while (flag) {
    // Generate a standard normal random variable
    float Z = oneapi::mkl::rng::device::generate(norm_distr, engine);
    if (Z > -1.f / c) {
      V = sycl::pow(1.f + c * Z, 3.f);
      float U = oneapi::mkl::rng::device::generate(uniform_distr, engine);
      flag = sycl::log(U) > (0.5f * Z * Z + d - d * V + d * sycl::log(V));
    }
  }
  return d * V / b;
}

/* 
   Initializes GPU random number generators 
 */
void setup_kernel(nd_item<1> &item,
                  oneapi::mkl::rng::device::philox4x32x10<1> *state,
                  int num_sample, unsigned int seed)
{
  int id = item.get_global_id(0);
  if (id < num_sample)
    /* Each thread gets same seed, a different sequence number, no offset */
    state[id] = oneapi::mkl::rng::device::philox4x32x10<1>(
                seed, {0, static_cast<std::uint64_t>(id * 8)});
}

/*
   Sample each theta from the appropriate gamma distribution
 */
void sample_theta(nd_item<1> &item,
                  oneapi::mkl::rng::device::philox4x32x10<1> *__restrict state,
                  float *__restrict theta,
                  const int *__restrict y,
                  const float *__restrict n,
                  float a, float b, int num_sample)
{
  int id = item.get_global_id(0);
  if(id < num_sample) {
    oneapi::mkl::rng::device::philox4x32x10<1> *s = state + id;
    const float hyperA = a + y[id];
    const float hyperB = b + n[id];
    float x;
    if (hyperA < 1.f) {
      oneapi::mkl::rng::device::uniform<float> uniform_distr;
      x = rgamma(s, hyperA + 1.f, hyperB) * 
          sycl::pow(oneapi::mkl::rng::device::generate(uniform_distr, *s), 1.f / hyperA);
    } else {
      x = rgamma(s, hyperA, hyperB);
    }
    theta[id] = x;
  }
}

/* 
   Sampling of a and b require the sum and product of all theta 
   values. This function performs parallel summations of 
   flat values and logs of theta for many blocks of length
   THREADS_PER_BLOCK_ADD. The CPU will then sum the block
   sums (see main method);    
 */
void sum_blocks(nd_item<1> &item, 
                float *__restrict flats,
                float *__restrict logs,
                const float *__restrict theta,
                float *__restrict flat_sums, 
                float *__restrict log_sums,
                int num_sample)
{
  int lid = item.get_local_id(0); 
  int gid = item.get_group(0); 
  int wgs = item.get_local_range(0); 
  int id = gid * wgs + lid;

  flats[lid] = (id < num_sample) ? theta[id] : 0.f;
  logs[lid] = (id < num_sample) ? sycl::log(theta[id]) : 0.f;

  item.barrier(access::fence_space::local_space);

  int i = wgs / 2;
  while (i != 0) {
    if (lid < i) {
      flats[lid] += flats[lid + i];
      logs[lid] += logs[lid + i];
    }
    i /= 2;
    item.barrier(access::fence_space::local_space);
  }

  if (lid == 0) {
    flat_sums[gid] = flats[0];
    log_sums[gid] = logs[0];
  }
}

int main(int argc, char *argv[]) {

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  const int seed = 123; // seed for random number generation
  srand(seed);

  int trials = 1000; // Default number of trials
  if(argc > 2) {
    trials = atoi(argv[2]);
  }

  /*------ Loading Data ------------------------------------------*/

  FILE *fp;
  if(argc > 1){
    fp = fopen(argv[1], "r");
  } else {
    printf("Please provide input filename\n");
    return 1;
  }

  if(fp == NULL){
    printf("Cannot read file \n");
    return 1;
  }

  int N = 0;
  char line[128];
  while( fgets (line, sizeof line, fp) != NULL ) { 
    N++; 
  }

  rewind(fp);

  size_t y_size = sizeof(int) * N;
  size_t n_size = sizeof(float) * N;
  int *y = (int*) malloc (y_size);
  float *n = (float*) malloc (n_size);
  float *host_fpsum = (float*) malloc (n_size);
  float *host_lpsum = (float*) malloc (n_size);

  for(int i = 0; i < N; i++) {
    fscanf(fp, "%d %f", &y[i], &n[i]);    
  }

  fclose(fp);

  /*------ Memory Allocations ------------------------------------*/
  float *dev_theta, *dev_fpsum, *dev_lpsum, *dev_n;
  int *dev_y;

  dev_y = malloc_device<int>(N, q);
  q.memcpy(dev_y, y, y_size);

  dev_n = malloc_device<float>(N, q);
  q.memcpy(dev_n, n, n_size);

  // Allocate memory for the partial flat and log sums
  int nSumBlocks = (N + (THREADS_PER_BLOCK_ADD - 1)) / THREADS_PER_BLOCK_ADD;
  range<1> sum_gws (nSumBlocks * THREADS_PER_BLOCK_ADD);
  range<1> sum_lws (THREADS_PER_BLOCK_ADD);

  dev_fpsum = malloc_device<float>(nSumBlocks, q);
  dev_lpsum = malloc_device<float>(nSumBlocks, q);

  /* Allocate space for theta on device and host */
  dev_theta = malloc_device<float>(N, q);

  /* Allocate space for random states on device */
  oneapi::mkl::rng::device::philox4x32x10<1> *devStates;
  devStates = malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(N, q);

  /*------ Setup RNG ---------------------------------------------*/

  int nBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  range<1> rng_gws (nBlocks * THREADS_PER_BLOCK);
  range<1> rng_lws (THREADS_PER_BLOCK);

  // Setup the rng machines (one for each thread)
  q.submit([&] (handler &cgh) {
    cgh.parallel_for(nd_range<1>(rng_gws, rng_lws), [=](nd_item<1> item) {
      setup_kernel(item, devStates, N, seed);
    });
  });

  /*------ MCMC Algorithm ----------------------------------------*/
  float a = 20; // Set starting value
  float b = 1;  // Set starting value

  double mean_a = 0.0, mean_b = 0.0;
  double total_time = 0.0;
  for(int i = 0; i < trials; i++) {
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &cgh) {
      cgh.parallel_for(nd_range<1>(rng_gws, rng_lws), [=](nd_item<1> item) {
        sample_theta(item, devStates, dev_theta, dev_y, dev_n, a, b, N);
      });
    });

    // Sum the flat and log values of theta
    q.submit([&](handler &cgh) {
      accessor<float, 1, access_mode::read_write, access::target::local>
          flats_acc(range<1>(THREADS_PER_BLOCK_ADD), cgh);
      accessor<float, 1, access_mode::read_write, access::target::local>
          logs_acc(range<1>(THREADS_PER_BLOCK_ADD), cgh);
      cgh.parallel_for(nd_range<1>(sum_gws, sum_lws), [=](nd_item<1> item) {
        sum_blocks(item, flats_acc.get_pointer(), logs_acc.get_pointer(),
                   dev_theta, dev_fpsum, dev_lpsum, N);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    q.memcpy(host_fpsum, dev_fpsum, nSumBlocks * sizeof(float));
    q.memcpy(host_lpsum, dev_lpsum, nSumBlocks * sizeof(float));
    q.wait();

    // The GPU summed blocks of theta values, now the CPU sums the blocks
    float flat_sum = 0;
    float log_sum = 0; 
    for(int j = 0; j < nSumBlocks; j++) {
      flat_sum += host_fpsum[j];
      log_sum += host_lpsum[j];
    }

    // Sample one random value from a's distribution
    a = sample_a(a, b, N, log_sum);
    mean_a += a;

    // And then from b's distribution given the new a
    b = sample_b(a, N, flat_sum);
    mean_b += b;
  }

  printf("Average execution time of kernels: %f (us)\n", (total_time * 1e-3f) / trials);
  printf("a = %lf (avg), b = %lf (avg)\n", mean_a / trials, mean_b / trials);

  /*------ Free Memory -------------------------------------------*/
  free(devStates, q);
  free(dev_theta, q);
  free(dev_fpsum, q);
  free(dev_lpsum, q);
  free(dev_y, q);
  free(dev_n, q);
  free(host_fpsum);
  free(host_lpsum);
  free(y);
  free(n);

  return EXIT_SUCCESS;
}
