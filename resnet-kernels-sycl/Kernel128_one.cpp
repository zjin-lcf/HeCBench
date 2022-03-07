#include "util.h"

const char inputName128one[] = "data/input_one_14_1024.bin";
const char weightName128one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name128one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name128one[] = "data/bnScale_myKernel_one_1024.bin";

#define __syncthreads() item.barrier(access::fence_space::local_space)

void kernel_512_one_128(
  nd_item<2> &item,
        float *__restrict shared_,
  const float *__restrict A,
  const float *__restrict B,
  const float *__restrict bnBias,
  const float *__restrict bnScale,
        float *__restrict C) 
{
  int tile = item.get_group(1), 
      in_channel = item.get_local_id(1),
      line = item.get_local_id(0);
  int ind = line*128 + in_channel;

  float *__restrict weights = shared_ + 512*4,
        *__restrict output = weights + 128*64,
        *__restrict input = shared_;
  float *__restrict bias = output + 4*128,
        *__restrict scale = bias + 128;

  for (int i = 0; i < 4; i++)
    input[ind + i*512] = A[tile*2048 + i*512 + ind];
  bias[in_channel] = bnBias[in_channel];
  scale[in_channel] = bnScale[in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 512; k += 64) {
    const float *B_start = B + k*128;
    for (int i = 0; i < 16; i++)
      weights[ind + i*512] = B_start[i*512 + ind];
    __syncthreads();

    const float *A_start = input + k;
    for (int p = 0; p < 64; p++) {
      output[ind] += A_start[line*512 + p] * weights[in_channel + p*128];
    }
    __syncthreads();
  }

  float *C_start = C + tile*512, res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[ind] = res > 0 ? res : 0;
}

void kernel_128_one_512(
  nd_item<2> &item,
        float *__restrict shared_,
  const float *__restrict A,
  const float *__restrict B,
  const float *__restrict bnBias,
  const float *__restrict bnScale,
        float *__restrict C) 
{
  int tile = item.get_group(1), part = item.get_group(0),
      in_channel = item.get_local_id(1), line = item.get_local_id(0);
  int ind = line*128 + in_channel;

  float *weights = shared_ + 128*4, *output = weights + 128*64, *input = shared_;
  float *bias = output + 4*128, *scale = bias + 128;

  input[ind] = A[tile * 512 + ind];
  bias[in_channel] = bnBias[part*128 + in_channel];
  scale[in_channel] = bnScale[part*128+ in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 128; k += 64) {
    for (int i = 0; i < 16; i++)
      weights[ind + 512*i] = B[(k + i*4 + line)*512 + part*128 + in_channel];
    __syncthreads();

    float *A_start = input + k;
    for (int p = 0; p < 64; p++) {
      output[ind] += A_start[line*128 + p] * weights[in_channel + p*128];
    }
    __syncthreads();
  }

  float *C_start = C + tile*2048 + part*128;
  float res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[line * 512 + in_channel] = res;
}

int kernel_128_1_in(queue &q) {
  float *input = get_parameter(inputName128one, 14*14*512);
  float *weight = get_parameter(weightName128one, 128*512);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 128);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 128);

  int nInput = 14*14*512, nOutput = 14*14*128, nWeights = 128*512;
  float result[nOutput];

  uint64_t nT1 = 0, nT2 = 0;

  nT1 = getTimeMicroseconds64();

  buffer<float, 1> input_(input, nInput);
  buffer<float, 1> output_(nOutput);
  buffer<float, 1> weight_(weight, nWeights);
  buffer<float, 1> bnBias_(bnBias_myKernel, 128);
  buffer<float, 1> bnScale_(bnScale_myKernel, 128);

  range<2> gws (4, 128*49);
  range<2> lws (4, 128);

  q.submit([&] (handler &cgh) {
    auto i = input_.get_access<sycl_read>(cgh);
    auto w = weight_.get_access<sycl_read>(cgh);
    auto b = bnBias_.get_access<sycl_read>(cgh);
    auto s = bnScale_.get_access<sycl_read>(cgh);
    auto o = output_.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local>
      sm (4*512 + 64*128 + 4*128 + 2*128, cgh);
    cgh.parallel_for<class k512_128>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      kernel_512_one_128 (item, sm.get_pointer(), i.get_pointer(),
        w.get_pointer(), b.get_pointer(), s.get_pointer(), o.get_pointer());
    });
  });

  q.submit([&] (handler &cgh) {
    auto acc = output_.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  }).wait();

  nT2 = getTimeMicroseconds64();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(bnBias_myKernel);
  free(bnScale_myKernel);
  free(input);
  free(weight);

  return ((nT2-nT1) << 16);
}

int kernel_128_1_out(queue &q) {
  float *input = get_parameter(inputName128one, 14*14*128);
  float *weight = get_parameter(weightName128one, 128*512);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 512);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 512);

  int nInput = 14*14*128, nOutput = 14*14*512, nWeights = 128*512;
  float result[nOutput];

  uint64_t nT1 = 0, nT2 = 0;

  nT1 = getTimeMicroseconds64();
  
  buffer<float, 1> input_(input, nInput);
  buffer<float, 1> output_(nOutput);
  buffer<float, 1> weight_(weight, nWeights);
  buffer<float, 1> bnBias_(bnBias_myKernel, 512);
  buffer<float, 1> bnScale_(bnScale_myKernel, 512);

  range<2> gws (4*4, 128*49);
  range<2> lws (4, 128);

  q.submit([&] (handler &cgh) {
    auto i = input_.get_access<sycl_read>(cgh);
    auto w = weight_.get_access<sycl_read>(cgh);
    auto b = bnBias_.get_access<sycl_read>(cgh);
    auto s = bnScale_.get_access<sycl_read>(cgh);
    auto o = output_.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local>
      sm (4*128 + 64*128 + 4*128 + 2*128, cgh);
    cgh.parallel_for<class k128_512>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      kernel_128_one_512 (item, sm.get_pointer(), i.get_pointer(),
        w.get_pointer(), b.get_pointer(), s.get_pointer(), o.get_pointer());
    });
  });

  q.submit([&] (handler &cgh) {
    auto acc = output_.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  }).wait();

  nT2 = getTimeMicroseconds64();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(bnBias_myKernel);
  free(bnScale_myKernel);
  free(input);
  free(weight);

  return ((nT2-nT1) << 16);
}
