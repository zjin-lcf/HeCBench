#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

__global__ 
void kernel1 (
    const float*__restrict key, 
    const float*__restrict query, 
    float*__restrict dot_product, 
    float*__restrict exp_sum, 
    const int n,
    const int d) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, expf(sum));
  }
}

__global__ 
void kernel2 (
    const float*__restrict exp_sum, 
    const float*__restrict dot_product, 
    float*__restrict score, 
    const int n)
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n)
    score[i] = expf(dot_product[i]) / exp_sum[0];
}

__global__ 
void kernel3 (
    const float*__restrict score, 
    const float*__restrict value, 
    float*__restrict output, 
    const int n,
    const int d) 
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;  
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}

float* attention_host(const float* key, const float* value, const float* query,
    const int n, const int d) 
{
  // intermediate
  float* dot_product = (float*) malloc (n * sizeof(float));
  float* score = (float*) malloc (n * sizeof(float));
  // result
  float* output = (float*) malloc (d * sizeof(float));

  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
  }

  float sum = 0;
  for (int i = 0; i < n; i++)
    sum += expf(dot_product[i]);

  for (int i = 0; i < n; i++)
    score[i] = expf(dot_product[i]) / sum;

  for (int j = 0; j < d; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }

  free(dot_product);
  free(score);
  return output;
}

float* attention_device(const float* key, const float* value, const float* query,
    const int n, const int d, const int repeat) 
{
  // input
  float *d_key;
  hipMalloc((void**)&d_key, n * d * sizeof(float)); 
  hipMemcpy(d_key, key, n * d * sizeof(float), hipMemcpyHostToDevice); 

  float *d_value;
  hipMalloc((void**)&d_value, n * d * sizeof(float)); 
  hipMemcpy(d_value, value, n * d * sizeof(float), hipMemcpyHostToDevice); 

  float *d_query;
  hipMalloc((void**)&d_query, d * sizeof(float)); 
  hipMemcpy(d_query, query, d * sizeof(float), hipMemcpyHostToDevice); 

  // intermediate
  float *d_dot_product;
  hipMalloc((void**)&d_dot_product, n * sizeof(float));

  float *d_score;
  hipMalloc((void**)&d_score, n * sizeof(float));

  float *d_exp_sum;
  hipMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (d * sizeof(float));
  float *d_output;
  hipMalloc((void**)&d_output, d * sizeof(float));

  dim3 n_grid((n+255)/256);
  dim3 n_block(256);
  dim3 d_grid((d+255)/256);
  dim3 d_block(256);

  for (int k = 0; k < repeat; k++) {
    hipMemset(d_exp_sum, 0, 4);

    hipLaunchKernelGGL(kernel1, dim3(n_grid), dim3(n_block), 0, 0, d_key, d_query, d_dot_product, d_exp_sum, n, d);

    hipLaunchKernelGGL(kernel2, dim3(n_grid), dim3(n_block), 0, 0, d_exp_sum, d_dot_product, d_score, n);

    hipLaunchKernelGGL(kernel3, dim3(d_grid), dim3(d_block), 0, 0, d_score, d_value, d_output, n, d);
  }

  hipMemcpy(output, d_output, d * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_score);
  hipFree(d_value);
  hipFree(d_output);
  hipFree(d_key);
  hipFree(d_dot_product);
  hipFree(d_exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  srand(2);
  for (int i = 0; i < n * d; i++) {
    key[i] = 0.1;
    value[i] = 0.3;
    if (rand() % 2)
      query[i % d] = value[i] + key[i] ;
    else
      query[i % d] = value[i] - key[i] ;
  }

  float* hout = attention_host(key, value, query, n, d);
  float* dout = attention_device(key, value, query, n, d, r);

  float rmse = 0;
  for (int i = 0; i < d; i++) 
    rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
  printf("RMSE = %f\n", sqrtf(rmse / d));

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
