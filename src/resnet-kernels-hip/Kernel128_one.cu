#include "util.h"

const char inputName128one[] = "data/input_one_14_1024.bin";
const char weightName128one[] = "data/weight_one_1024.bin";
const char bnBias_myKernel_Name128one[] = "data/bnBias_myKernel_one_1024.bin";
const char bnScale_myKernel_Name128one[] = "data/bnScale_myKernel_one_1024.bin";

__global__ void kernel_512_one_128(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*128 + in_channel;

  extern __shared__ float shared_[];
  float *__restrict__ weights = shared_ + 512*4,
        *__restrict__ output = weights + 128*64,
        *__restrict__ input = shared_;
  float *__restrict__ bias = output + 4*128,
        *__restrict__ scale = bias + 128;

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

__global__ void kernel_128_one_512(
  const float *__restrict__ A,
  const float *__restrict__ B,
  const float *__restrict__ bnBias,
  const float *__restrict__ bnScale,
        float *__restrict__ C) 
{
  int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
  int ind = line*128 + in_channel;

  extern __shared__ float shared_[];
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

void kernel_128_1_in(double &time, double &ktime) {
  float *input = get_parameter(inputName128one, 14*14*512);
  float *weight = get_parameter(weightName128one, 128*512);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 128);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 128);

  float *input_, *output_, *weight_, *bnBias_, *bnScale_;

  int nInput = 14*14*512, nOutput = 14*14*128, nWeights = 128*512;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  hipMalloc((void **) &input_, nInput<<2);
  hipMalloc((void **) &output_, nOutput<<2);
  hipMalloc((void **) &weight_, nWeights<<2);
  hipMalloc((void **) &bnBias_, 128<<2);
  hipMalloc((void **) &bnScale_, 128<<2);

  hipMemcpy(input_, input, nInput<<2, hipMemcpyHostToDevice);
  hipMemcpy(weight_, weight, nWeights<<2, hipMemcpyHostToDevice);
  hipMemcpy(bnBias_, bnBias_myKernel, 128<<2, hipMemcpyHostToDevice);
  hipMemcpy(bnScale_, bnScale_myKernel, 128<<2, hipMemcpyHostToDevice);

  hipDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  kernel_512_one_128 <<<dim3(49), dim3(128, 4), (4*512 + 64*128 + 4*128 + 2*128)<<2 >>>(
      input_, weight_, bnBias_, bnScale_, output_);

  hipDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  hipMemcpy(result, output_, nOutput<<2, hipMemcpyDeviceToHost);

  hipFree(bnScale_);
  hipFree(bnBias_);
  hipFree(input_);
  hipFree(output_);
  hipFree(weight_);

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

void kernel_128_1_out(double &time, double &ktime) {
  float *input = get_parameter(inputName128one, 14*14*128);
  float *weight = get_parameter(weightName128one, 128*512);
  float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name128one, 512);
  float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name128one, 512);

  float *input_, *output_, *weight_, *bnBias_, *bnScale_;

  int nInput = 14*14*128, nOutput = 14*14*512, nWeights = 128*512;
  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  hipMalloc((void **) &input_, nInput<<2);
  hipMalloc((void **) &output_, nOutput<<2);
  hipMalloc((void **) &weight_, nWeights<<2);
  hipMalloc((void **) &bnBias_, 512<<2);
  hipMalloc((void **) &bnScale_, 512<<2);

  hipMemcpy(input_, input, nInput<<2, hipMemcpyHostToDevice);
  hipMemcpy(weight_, weight, nWeights<<2, hipMemcpyHostToDevice);
  hipMemcpy(bnBias_, bnBias_myKernel, 512<<2, hipMemcpyHostToDevice);
  hipMemcpy(bnScale_, bnScale_myKernel, 512<<2, hipMemcpyHostToDevice);

  hipDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  kernel_128_one_512 <<<dim3(49, 4), dim3(128, 4), (4*128 + 64*128 + 4*128 + 2*128)<<2 >>> (
      input_, weight_, bnBias_, bnScale_, output_);

  hipDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  hipMemcpy(result, output_, nOutput<<2, hipMemcpyDeviceToHost);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  hipFree(input_);
  hipFree(output_);
  hipFree(weight_);
  hipFree(bnScale_);
  hipFree(bnBias_);

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
