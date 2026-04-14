#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"
#include "common.h"

inline void gpu_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define GPU_CHECK(err) (gpu_check(err, __FILE__, __LINE__))

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
  D* out_gpu = (D*)malloc(num_elements * sizeof(D));
  GPU_CHECK(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (std::fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
      printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
      nfaults++;
      if (nfaults >= max_int(10, n_print)) {
        break;
      }
    }
  }
  
  printf("%s\n", (nfaults == 0) ? "PASS" : "FAIL");
  free(out_gpu);
}


__global__
void silu_forward_kernel(const float* x, float* out, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float x_val = x[i];
  out[i] = x_val / (1.0f + expf(-x_val));
}

__global__
void silu_forward_kernel2(const float* x, float* out, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t vec_count = N / 4;
  if (i < vec_count) {
    float4 x4  = reinterpret_cast<const float4*>(x)[i];
    float4 t4;
    t4.x = x4.x / (1.0f + expf(-x4.x));
    t4.y = x4.y / (1.0f + expf(-x4.y));
    t4.z = x4.z / (1.0f + expf(-x4.z));
    t4.w = x4.w / (1.0f + expf(-x4.w));
    reinterpret_cast<float4*>(out)[i] = t4;
  }
  for (uint64_t j = 4 * vec_count + i; j < N; j += blockDim.x) {
    float x_val = x[j];
    out[j] = x_val / (1.0f + expf(-x_val));
  }
}

__global__
void silu_backward_kernel(const float* dout, const float* x, float* dx, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float out_val = dout[i];
  float x_val = x[i];
  float expx = expf(-x_val);
  float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);
  dx[i] = out_val * grad_silu;
}

__global__
void silu_backward_kernel2(const float* dout, const float* x, float* dx, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float x_val = x[i];
  // sigmoid: 1 / (1 + exp(-x))
  float sig = 1.0f / (1.0f + expf(-x_val));
  // gradient: sig * (1 + x * (1 - sig))
  float grad_silu = sig * (1.0f + x_val * (1.0f - sig));
  dx[i] = dout[i] * grad_silu;
}

__global__
void silu_backward_kernel3(const float* dout, const float* x, float* dx, uint64_t N)
{
  uint64_t i = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t vec_count = N / 4;
  if (i < vec_count) {
    float4 x4  = reinterpret_cast<const float4*>(x)[i];
    float4 do4  = reinterpret_cast<const float4*>(dout)[i];
    float4 sig4, grad_silu4;
    sig4.x = 1.0f / (1.0f + expf(-x4.x));
    sig4.y = 1.0f / (1.0f + expf(-x4.y));
    sig4.z = 1.0f / (1.0f + expf(-x4.z));
    sig4.w = 1.0f / (1.0f + expf(-x4.w));
    grad_silu4.x = sig4.x * (1.0f + x4.x * (1.0f - sig4.x));
    grad_silu4.y = sig4.y * (1.0f + x4.y * (1.0f - sig4.y));
    grad_silu4.z = sig4.z * (1.0f + x4.z * (1.0f - sig4.z));
    grad_silu4.w = sig4.w * (1.0f + x4.w * (1.0f - sig4.w));
    sig4.x = do4.x * grad_silu4.x;
    sig4.y = do4.y * grad_silu4.y;
    sig4.z = do4.z * grad_silu4.z;
    sig4.w = do4.w * grad_silu4.w;
    reinterpret_cast<float4*>(dx)[i] = sig4;
  }
  for (uint64_t j = 4 * vec_count + i; j < N; j += blockDim.x) {
    float x_val = x[j];
    float sig = 1.0f / (1.0f + expf(-x_val));
    float grad_silu = sig * (1.0f + x_val * (1.0f - sig));
    dx[j] = dout[j] * grad_silu;
  }
}

void silu_forward(const float* x, float* out, uint64_t N, int block_size)
{
  int n_blk = ceil_div(N, block_size);
  silu_forward_kernel<<<n_blk, block_size>>>(x, out, N);
}

void silu_forward2(const float* x, float* out, uint64_t N, int block_size)
{
  int n_blk = ceil_div(N/4, block_size);
  silu_forward_kernel2<<<n_blk, block_size>>>(x, out, N);
}

void silu_backward(const float* dout, const float* x, float* dx, uint64_t N, int block_size)
{
  int n_blk = ceil_div(N, block_size);
  silu_backward_kernel<<<n_blk, block_size>>>(dout, x, dx, N);
}

void silu_backward2(const float* dout, const float* x, float* dx, uint64_t N, int block_size)
{
  int n_blk = ceil_div(N, block_size);
  silu_backward_kernel2<<<n_blk, block_size>>>(dout, x, dx, N);
}

void silu_backward3(const float* dout, const float* x, float* dx, uint64_t N, int block_size)
{
  int n_blk = ceil_div(N/4, block_size);
  silu_backward_kernel3<<<n_blk, block_size>>>(dout, x, dx, N);
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Usage: %s <batch size> <number of channels> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  uint64_t B = atoi(argv[1]);
  uint64_t C = atoi(argv[2]);
  uint64_t H = atoi(argv[3]);
  uint64_t W = atoi(argv[4]);
  int repeat = atoi(argv[5]);
  uint64_t N = B * C * H * W;

  srand(0);
  float* x = make_random_float(N);
  float* dout = make_random_float(N);
  float* out = (float*)malloc(N * sizeof(float));
  float* dx = (float*)malloc(N * sizeof(float));

  silu_forward_reference(x, out, N);
  silu_backward_reference(dout, x, dx, N);

  // allocate device memory
  float *d_x, *d_out, *d_dout, *d_dx;
  GPU_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_dout, N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_dx, N * sizeof(float)));
  GPU_CHECK(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_dout, dout, N * sizeof(float), cudaMemcpyHostToDevice));

  int block_sizes[] = {64, 128, 256, 512, 1024};
  printf("Checking forward pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_forward(d_x, d_out, N, block_size);
    validate_result(d_out, out, "out", N);
  }

  printf("Checking forward2 pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_forward2(d_x, d_out, N, block_size);
    validate_result(d_out, out, "out", N);
  }

  printf("Checking backward pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_backward(d_dout, d_x, d_dx, N, block_size);
    validate_result(d_dx, dx, "dx", N);
  }
  printf("Checking backward2 pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_backward2(d_dout, d_x, d_dx, N, block_size);
    validate_result(d_dx, dx, "dx", N);
  }
  printf("Checking backward3 pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_backward3(d_dout, d_x, d_dx, N, block_size);
    validate_result(d_dx, dx, "dx", N);
  }

  printf("\nForward pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(repeat, silu_forward, d_x, d_out, N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nForward2 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(repeat, silu_forward2, d_x, d_out, N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nBackward pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(repeat, silu_backward,
                                          d_dout, d_x, d_dx,
                                          N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nBackward2 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(repeat, silu_backward2,
                                          d_dout, d_x, d_dx,
                                          N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nBackward3 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(repeat, silu_backward3,
                                          d_dout, d_x, d_dx,
                                          N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  free(x);
  free(out);
  free(dout);
  free(dx);

  GPU_CHECK(cudaFree(d_x));
  GPU_CHECK(cudaFree(d_out));
  GPU_CHECK(cudaFree(d_dout));
  GPU_CHECK(cudaFree(d_dx));

  return 0;
}
