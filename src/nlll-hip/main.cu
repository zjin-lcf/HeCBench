#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include "reference.h"

template <typename scalar_t, typename accscalar_t, 
          typename index_t, int NLL_LOSS_THREADS>
__global__
void nll_loss_forward_reduce2d_kernel(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ total_weight,
    const scalar_t* __restrict__ input,
    const index_t*  __restrict__ target,
    const scalar_t* __restrict__ weights,
    bool size_average,
    int64_t nframe,
    int64_t kdim,
    int64_t ignore_index)
{
  __shared__ accscalar_t sm_inputs[NLL_LOSS_THREADS],
                         acc_weight[NLL_LOSS_THREADS];

  int tid = threadIdx.x;
  sm_inputs[tid] = static_cast<accscalar_t>(0);
  acc_weight[tid] = static_cast<accscalar_t>(0);

  for (int i = tid; i < nframe; i += NLL_LOSS_THREADS) {
    index_t t = target[i];
    if (t != ignore_index) {
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sm_inputs[tid] -= input[i * kdim + t] * cur_weight;
      acc_weight[tid] += cur_weight;
    }
  }

  __syncthreads();

  if (tid == 0) {
    accscalar_t output_acc = 0;
    accscalar_t total_weight_acc = 0;
    for (int i = 0; i < NLL_LOSS_THREADS; ++i) {
      output_acc += sm_inputs[i];
      total_weight_acc += acc_weight[i];
    }
    *total_weight = static_cast<scalar_t>(total_weight_acc);
    if (size_average) {
      *output = static_cast<scalar_t>(output_acc / total_weight_acc);
    } else {
      *output = static_cast<scalar_t>(output_acc);
    }
  }
}

template <typename scalar_t, typename index_t, int GPU_THREADS>
void eval(const int64_t nframe,
          const int64_t kdim,
          const int64_t n_classes,
          const bool size_average,
          const int64_t ignore_index,
          const scalar_t r_output,
          const scalar_t r_total_weight,
          scalar_t *h_input,
          scalar_t *h_weights,
           index_t *h_target,
          const int repeat)
{
  int64_t input_size = nframe * kdim * n_classes;
  int64_t input_size_bytes = input_size * sizeof(scalar_t);

  int64_t weights_size = nframe;
  int64_t weights_size_bytes = weights_size * sizeof(scalar_t);

  int64_t target_size = nframe;
  int64_t target_size_bytes = target_size * sizeof(index_t);

  int output_size_bytes = sizeof(scalar_t);

  scalar_t h_output;
  scalar_t h_total_weight;

  scalar_t *d_output;
  scalar_t *d_total_weight;
  scalar_t *d_input;
   index_t *d_target;
  scalar_t *d_weights;

  hipMalloc((void**)&d_input, input_size_bytes); 
  hipMemcpy(d_input, h_input, input_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_weights, weights_size_bytes); 
  hipMemcpy(d_weights, h_weights, weights_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_target, target_size_bytes); 
  hipMemcpy(d_target, h_target, target_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_total_weight, output_size_bytes); 
  hipMalloc((void**)&d_output, output_size_bytes); 

  dim3 grid (1);
  dim3 block (GPU_THREADS);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    nll_loss_forward_reduce2d_kernel
      <scalar_t, scalar_t, index_t, GPU_THREADS>
      <<<grid, block>>>(d_output,
                        d_total_weight,
                        d_input,
                        d_target,
                        d_weights,
                        size_average,
                        nframe,
                        kdim,
                        ignore_index);
  }
  hipDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nThread block size: %d\n", GPU_THREADS);
  printf("Average execution time of nll loss forward reduce 2D kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  hipMemcpy(&h_output, d_output, output_size_bytes, hipMemcpyDeviceToHost);
  hipMemcpy(&h_total_weight, d_total_weight, output_size_bytes, hipMemcpyDeviceToHost);

  bool ok = true;
  if (fabs(h_output - r_output) > 1e-1 ||
      fabs(h_total_weight - r_total_weight) > 1e-1) {
    printf("%f %f %f %f\n", (float)h_output, (float)r_output, 
                            (float)h_total_weight, (float)r_total_weight);
    ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  hipFree(d_output);
  hipFree(d_total_weight);
  hipFree(d_input);
  hipFree(d_target);
  hipFree(d_weights);
}

template <typename scalar_t, typename index_t>
void driver(char** argv) {
  const int64_t nframe = atol(argv[1]);
  const int64_t kdim = atol(argv[2]);
  const int64_t n_classes = atol(argv[3]);
  const int repeat = atoi(argv[4]);

  const int64_t input_size = nframe * kdim * n_classes;
  const int64_t input_size_bytes = input_size * sizeof(scalar_t);

  const int64_t weights_size = nframe;
  const int64_t weights_size_bytes = weights_size * sizeof(scalar_t);

  const int64_t target_size = nframe;
  const int64_t target_size_bytes = target_size * sizeof(index_t);

  scalar_t *h_input = (scalar_t*) malloc (input_size_bytes);
  scalar_t *h_weights = (scalar_t*) malloc (weights_size_bytes);
  index_t *h_target = (index_t*) malloc (target_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<scalar_t> d1 (-1.f, 1.f);
  std::uniform_int_distribution<index_t> d2 (0, n_classes-1);

  printf("Initialization of input data may take a while..\n");
  for (int64_t i = 0; i < input_size; i++)
    h_input[i] = d1(g);

  for (int64_t i = 0; i < weights_size; i++)
    h_weights[i] = d1(g);

  for (int64_t i = 0; i < target_size; i++)
    h_target[i] = d2(g);

  const bool size_average = true;

  // the index may not necessarily be in the class range
  const int64_t ignore_index = n_classes / 2;
  
  // verify the loss function
  scalar_t r_output;
  scalar_t r_total_weight;

  reference<scalar_t, scalar_t, index_t>(
    &r_output, &r_total_weight,
    h_input, h_target, h_weights,
    size_average, nframe, kdim, ignore_index);

  #define EVAL(nThreads) \
  eval<scalar_t, index_t, nThreads>(nframe, kdim, n_classes, \
                                    size_average, ignore_index, \
                                    r_output, r_total_weight, \
                                    h_input, h_weights, h_target, repeat)
  EVAL(64);
  EVAL(128);
  EVAL(256);
  EVAL(512);
  EVAL(1024);

  free(h_input);
  free(h_target);
  free(h_weights);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <minibatch> <kdim> <classes> <repeat>\n", argv[0]);
    return 1;
  }

  printf("=========== Data type is FP32 ==========\n");
  driver<float, int>(argv);

  return 0;
}
