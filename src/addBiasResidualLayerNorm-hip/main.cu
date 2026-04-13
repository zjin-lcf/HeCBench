/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "kernels.h"
#include "reference.h"

template<typename T, int V, int WarpSize>
void invokeAddBiasResidualLayerNorm(
          T*     out,
    const T*     input,
    const T*     bias,
    const T*     gamma,
    const T*     beta,
    const float  layernorm_eps,
    int          m,
    int          n)
{
  dim3 grid(m);
  dim3 block(std::min(n, 256));

  if (V == 2 && (n % 8 == 0)) {
    if (n / 8 >= WarpSize)
      addBiasResidualPostLayerNormV2<T, WarpSize><<<grid, n / 8>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    else if (n / 8 == WarpSize / 2) // the case of 32 
      addBiasResidualPostLayerNormV2<T, 32><<<grid, 32>>>(out, input, bias, gamma, beta, layernorm_eps, n);
  }
  else {
    int num_trips = (n + block.x - 1) / block.x;
    if (num_trips == 1) {
      addBiasResidualPostLayerNorm<T, 1, WarpSize>
        <<<grid, block>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
    else if (num_trips == 2) {
      addBiasResidualPostLayerNorm<T, 2, WarpSize>
        <<<grid, block>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
    else {
      generalAddBiasResidualPostLayerNorm<T, WarpSize>
        <<<grid, block>>>(out, input, bias, gamma, beta, layernorm_eps, n);
    }
  }
}

template<typename T, int V>
void layer(int m, int n, int repeat) {

  std::mt19937 gen (19937);
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  const int input_size = m * n;
  const int output_size = m * n;
  const int input_size_bytes = input_size * sizeof(T);
  const int output_size_bytes = output_size * sizeof(T);
  const int bias_size_bytes = n * sizeof(T);
  const int beta_size_bytes = n * sizeof(T);
  const int gamma_size_bytes = n * sizeof(T);

  T *h_input = (T*) malloc (input_size_bytes);
  T *h_output = (T*) malloc (output_size_bytes);
  T *h_bias = (T*) malloc (bias_size_bytes);
  T *h_gamma = (T*) malloc (gamma_size_bytes);
  T *h_beta = (T*) malloc (beta_size_bytes);

  for (int i = 0; i < input_size; i++) {
    h_input[i] = floatToType<T>(dis(gen));
  }
  for (int i = 0; i < n; i++) {
    h_bias[i] = floatToType<T>(dis(gen));
    h_gamma[i] = floatToType<T>(dis(gen));
    h_beta[i] = floatToType<T>(dis(gen));
  }

  float layernorm_eps = 1e-6;
  T *d_input, *d_output, *d_bias, *d_gamma, *d_beta;
  hipMalloc((void**)&d_input, input_size_bytes);
  hipMemcpy(d_input, h_input, input_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_output,  output_size_bytes);
  hipMemset(d_output,  0, output_size_bytes); // need to reset output

  hipMalloc((void**)&d_bias,  bias_size_bytes);
  hipMemcpy(d_bias, h_bias, bias_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_gamma,  gamma_size_bytes);
  hipMemcpy(d_gamma, h_gamma, gamma_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_beta,  beta_size_bytes);
  hipMemcpy(d_beta, h_beta, beta_size_bytes, hipMemcpyHostToDevice);

  int WarpSize;
  hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize, 0);

  // warmup and verify
  T *r_output = (T*) calloc (output_size, sizeof(T));
  for (int i = 0; i < 100; i++) {
    if (WarpSize == 64)
      invokeAddBiasResidualLayerNorm<T, V, 64>
        (d_output, d_input, d_bias, d_gamma, d_beta, layernorm_eps, m, n);
    else
      invokeAddBiasResidualLayerNorm<T, V, 32>
        (d_output, d_input, d_bias, d_gamma, d_beta, layernorm_eps, m, n);
    reference<T>(r_output, h_input, h_bias, h_gamma, h_beta, layernorm_eps, m, n);
  }
  hipMemcpy(h_output, d_output, output_size_bytes, hipMemcpyDeviceToHost);

  bool ok = true;
  float error_bound = sizeof(T) >= 4 ? 1e-4f : 0.5f;
  for (int i = 0; i < output_size; i++) {
    if (fabsf(float(h_output[i]) - float(r_output[i])) > error_bound) {
      printf("i=%d %f != %f (ref)\n", i, float(h_output[i]), float(r_output[i]));
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  free(r_output);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    if (WarpSize == 64)
      invokeAddBiasResidualLayerNorm<T, V, 64>
        (d_output, d_input, d_bias, d_gamma, d_beta, layernorm_eps, m, n);
    else
      invokeAddBiasResidualLayerNorm<T, V, 32>
        (d_output, d_input, d_bias, d_gamma, d_beta, layernorm_eps, m, n);
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AddBiasResidualLayerNorm (%d x %d): %f (us)\n",
         m, n, (time * 1e-3f) / repeat);

  hipFree(d_input);
  hipFree(d_output);
  hipFree(d_bias);
  hipFree(d_gamma);
  hipFree(d_beta);

  free(h_input);
  free(h_output);
  free(h_bias);
  free(h_gamma);
  free(h_beta);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int m = 4096;  // batch size

  // The minimum value of n is 256
  for (int n = 256; n <= 8192; n = n * 2) {
    printf("---------------- float32 (version 1) -------------\n");
    layer<float, 1>(m, n, repeat);
    printf("---------------- float32 (version 2) -------------\n");
    layer<float, 2>(m, n, repeat);

    printf("---------------- float16 (version 1) -------------\n");
    layer<__half, 1>(m, n, repeat);
    printf("---------------- float16 (version 2) -------------\n");
    layer<__half, 2>(m, n, repeat);

    printf("---------------- bfloat16 (version 1) -------------\n");
    layer<__hip_bfloat16, 1>(m, n, repeat);
    printf("---------------- bfloat16 (version 2) -------------\n");
    layer<__hip_bfloat16, 2>(m, n, repeat);
    printf("\n");
  }
  return 0;
}
