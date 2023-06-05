#include "util.h"

const char inputName256one[] = "data/input_one_14_1024.bin";
const char weightName256one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name256one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name256one[] = "data/bnScale_myKernel_one_1024.bin";

#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)

void kernel_1024_one_256(
  sycl::nd_item<2> &item,
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
  int ind = line*256 + in_channel;

  float *__restrict weights = shared_ + 1024*4,
        *__restrict output = weights + 256*16,
        *__restrict input = shared_;
  float *__restrict bias = output + 4*256,
        *__restrict scale = bias + 256;

  for (int i = 0; i < 4; i++)
    input[ind + i*1024] = A[tile*4096 + i*1024 + ind];
  bias[in_channel] = bnBias[in_channel];
  scale[in_channel] = bnScale[in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 1024; k += 16) {
    const float *B_start = B + k*256;
    for (int i = 0; i < 4; i++)
      weights[ind + i*1024] = B_start[i*1024 + ind];
    __syncthreads();

    const float *A_start = input + k;
    for (int p = 0; p < 16; p++) {
      output[ind] += A_start[line*1024 + p] * weights[in_channel + p*256];
    }
    __syncthreads();
  }

  float *C_start = C + tile*1024, res = scale[in_channel] * output[ind] + bias[in_channel];
  C_start[ind] = res > 0 ? res : 0;
}

void kernel_256_one_1024(
  sycl::nd_item<2> &item,
        float *__restrict shared_,
  const float *__restrict A,
  const float *__restrict B,
  const float *__restrict bnBias,
  const float *__restrict bnScale,
        float *__restrict C)
{
  int tile = item.get_group(1), part = item.get_group(0),
      in_channel = item.get_local_id(1), line = item.get_local_id(0);
  int ind = line*256 + in_channel;

  float *weights = shared_ + 256*4, *output = weights + 256*32, *input = shared_;
  float *bias = output + 4*256, *scale = bias + 256;

  input[ind] = A[tile * 1024 + ind];
  bias[in_channel] = bnBias[part*256 + in_channel];
  scale[in_channel] = bnScale[part*256+ in_channel];
  output[ind] = 0.0f;
  __syncthreads();

  for (int k = 0; k < 256; k += 32) {
    for (int i = 0; i < 8; i++)
      weights[ind + 1024*i] = B[(k + i*4 + line)*1024 + part*256 + in_channel];
    __syncthreads();

    float *A_start = input + k;
    for (int p = 0; p < 32; p++) {
      output[ind] += A_start[line*256 + p] * weights[in_channel + p*256];
    }
    __syncthreads();
  }

  float *C_start = C + tile*4096 + part*256;
  C_start[line * 1024 + in_channel] = scale[in_channel] * output[ind] + bias[in_channel];
}

void kernel_256_1_in(sycl::queue &q, double &time, double &ktime) {
  float *input = get_parameter(inputName256one, 14*14*1024);
  float *weight = get_parameter(weightName256one, 256*1024);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 256);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 256);

  int nInput = 14*14*1024, nOutput = 14*14*256, nWeights = 256*1024;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  float *input_ = sycl::malloc_device<float>(nInput, q);
  q.memcpy(input_, input, sizeof(float) * nInput);

  float *output_ = sycl::malloc_device<float>(nOutput, q);

  float *weight_ = sycl::malloc_device<float>(nWeights, q);
  q.memcpy(weight_, weight, sizeof(float) * nWeights);

  float *bnBias_ = sycl::malloc_device<float>(256, q);
  q.memcpy(bnBias_, bnBias_myKernel, sizeof(float) * 256);

  float *bnScale_ = sycl::malloc_device<float>(256, q);
  q.memcpy(bnScale_, bnScale_myKernel, sizeof(float) * 256);

  q.wait();
  auto kstart = std::chrono::steady_clock::now();

  sycl::range<2> gws (4, 256*49);
  sycl::range<2> lws (4, 256);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1>
      sm (sycl::range<1>(4*1024 + 16*256 + 4*256 + 2*256), cgh);
    cgh.parallel_for<class k1024>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      kernel_1024_one_256 (item, sm.get_pointer(), input_,
                           weight_, bnBias_, bnScale_, output_);
    });
  });

  q.wait();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  q.memcpy(result, output_, sizeof(float) * nOutput).wait();

  sycl::free(input_, q);
  sycl::free(output_, q);
  sycl::free(weight_, q);
  sycl::free(bnBias_, q);
  sycl::free(bnScale_, q);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(input);
  free(weight);
  free(bnBias_myKernel);
  free(bnScale_myKernel);
}

void kernel_256_1_out(sycl::queue &q, double &time, double &ktime) {
  float *input = get_parameter(inputName256one, 14*14*256);
  float *weight = get_parameter(weightName256one, 256*1024);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 1024);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 1024);

  int nInput = 14*14*256, nOutput = 14*14*1024, nWeights = 256*1024;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  float *input_ = sycl::malloc_device<float>(nInput, q);
  q.memcpy(input_, input, sizeof(float) * nInput);

  float *output_ = sycl::malloc_device<float>(nOutput, q);

  float *weight_ = sycl::malloc_device<float>(nWeights, q);
  q.memcpy(weight_, weight, sizeof(float) * nWeights);

  float *bnBias_ = sycl::malloc_device<float>(1024, q);
  q.memcpy(bnBias_, bnBias_myKernel, sizeof(float) * 1024);

  float *bnScale_ = sycl::malloc_device<float>(1024, q);
  q.memcpy(bnScale_, bnScale_myKernel, sizeof(float) * 1024);

  q.wait();
  auto kstart = std::chrono::steady_clock::now();

  sycl::range<2> gws (4*4, 256*49);
  sycl::range<2> lws (4, 256);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1>
      sm (sycl::range<1>(4*256 + 32*256 + 4*256 + 2*256), cgh);
    cgh.parallel_for<class k256_1024>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      kernel_256_one_1024 (item, sm.get_pointer(), input_,
                           weight_, bnBias_, bnScale_, output_);
    });
  });

  q.wait();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  q.memcpy(result, output_, sizeof(float) * nOutput).wait();

  sycl::free(input_, q);
  sycl::free(output_, q);
  sycl::free(weight_, q);
  sycl::free(bnBias_, q);
  sycl::free(bnScale_, q);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

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
}
