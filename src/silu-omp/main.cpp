#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <utility>
#include <omp.h>
#include "reference.h"
#include "common.h"

// ---------------- Benchmark ----------------
template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

// ---------------- Validation ----------------
template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
  #pragma omp target update from (device_result[0:num_elements])
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (std::fabs(cpu_reference[i] - (T)device_result[i]) > tolerance &&
        std::isfinite(cpu_reference[i])) {
      printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n",
             name, i, cpu_reference[i], (T)device_result[i]);
      nfaults++;
      if (nfaults >= max_int(10, n_print)) break;
    }
  }
  printf("%s\n", (nfaults == 0) ? "PASS" : "FAIL");
}

static inline int get_num_teams(uint64_t N, int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

// ---------------- Forward ----------------
void silu_forward(const float* x, float* out, uint64_t N, int block_size)
{
  int num_teams = get_num_teams(N, block_size);

  #pragma omp target teams distribute parallel for \
      num_teams(num_teams) num_threads(block_size)
  for (uint64_t i = 0; i < N; i++) {
    float x_val = x[i];
    out[i] = x_val / (1.0f + expf(-x_val));
  }
}

// ---------------- Forward (vectorized) ----------------
void silu_forward2(const float* x, float* out, uint64_t N, int block_size)
{
  uint64_t vec_count = N / 4;
  int num_teams = get_num_teams(vec_count, block_size);

  #pragma omp target teams distribute parallel for \
      num_teams(num_teams) num_threads(block_size)
  for (uint64_t i = 0; i < vec_count; i++) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      float x_val = x[4*i + k];
      out[4*i + k] = x_val / (1.0f + expf(-x_val));
    }
  }

  // tail
  #pragma omp target teams distribute parallel for \
   num_teams(1) num_threads(N - 4 * vec_count)
  for (uint64_t i = 4 * vec_count; i < N; i++) {
    float x_val = x[i];
    out[i] = x_val / (1.0f + expf(-x_val));
  }
}

// ---------------- Backward ----------------
void silu_backward(const float* dout, const float* x, float* dx,
                   uint64_t N, int block_size)
{
  int num_teams = get_num_teams(N, block_size);

  #pragma omp target teams distribute parallel for \
      num_teams(num_teams) num_threads(block_size)
  for (uint64_t i = 0; i < N; i++) {
    float x_val = x[i];
    float sig = 1.0f / (1.0f + expf(-x_val));
    float grad = sig * (1.0f + x_val * (1.0f - sig));
    dx[i] = dout[i] * grad;
  }
}

// ---------------- Backward2 (vectorized) ----------------
void silu_backward2(const float* dout, const float* x, float* dx,
                    uint64_t N, int block_size)
{
  uint64_t vec_count = N / 4;
  int num_teams = get_num_teams(vec_count, block_size);

  #pragma omp target teams distribute parallel for \
      num_teams(num_teams) num_threads(block_size)
  for (uint64_t i = 0; i < vec_count; i++) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      float x_val = x[4*i + k];
      float sig = 1.0f / (1.0f + expf(-x_val));
      float grad = sig * (1.0f + x_val * (1.0f - sig));
      dx[4*i + k] = dout[4*i + k] * grad;
    }
  }

  // tail
  #pragma omp target teams distribute parallel for \
      num_teams(1) num_threads(N - 4 * vec_count)
  for (uint64_t i = 4 * vec_count; i < N; i++) {
    float xv = x[i];
    float sig = 1.0f / (1.0f + expf(-xv));
    float grad = sig * (1.0f + xv * (1.0f - sig));
    dx[i] = dout[i] * grad;
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Usage: %s <B> <C> <H> <W> <repeat>\n", argv[0]);
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
  float* d_out = (float*)malloc(N * sizeof(float));
  float* dx  = (float*)malloc(N * sizeof(float));
  float* d_dx  = (float*)malloc(N * sizeof(float));

  silu_forward_reference(x, out, N);
  silu_backward_reference(dout, x, dx, N);

  int block_sizes[] = {64,128,256,512,1024};

  #pragma omp target data map(to: x[0:N], dout[0:N]) \
                          map(alloc: d_out[0:N], d_dx[0:N])
  {
    printf("Checking forward pass\n");
    for (int bs: block_sizes) {
      printf("Checking block size %d\n", bs);
      silu_forward(x, d_out, N, bs);
      validate_result(d_out, out, "out", N);
    }

    printf("Checking forward2 pass\n");
    for (int bs: block_sizes) {
      printf("Checking block size %d\n", bs);
      silu_forward2(x, d_out, N, bs);
      validate_result(d_out, out, "out", N);
    }

    printf("Checking backward pass\n");
    for (int bs: block_sizes) {
      printf("Checking block size %d\n", bs);
      silu_backward(dout, x, d_dx, N, bs);
      validate_result(d_dx, dx, "dx", N);
    }

    printf("Checking backward2 pass\n");
    for (int bs: block_sizes) {
      printf("Checking block size %d\n", bs);
      silu_backward2(dout, x, d_dx, N, bs);
      validate_result(d_dx, dx, "dx", N);
    }

    printf("\nForward pass benchmarks:\n");
    for (int bs: block_sizes) {
      float t = benchmark_kernel(repeat, silu_forward, x, d_out, N, bs);
      printf("block_size %4d | time %.4f ms\n", bs, t);
    }

    printf("\nForward2 pass benchmarks:\n");
    for (int bs: block_sizes) {
      float t = benchmark_kernel(repeat, silu_forward2, x, d_out, N, bs);
      printf("block_size %4d | time %.4f ms\n", bs, t);
    }

    printf("\nBackward pass benchmarks:\n");
    for (int bs: block_sizes) {
      float t = benchmark_kernel(repeat, silu_backward, dout, x, d_dx, N, bs);
      printf("block_size %4d | time %.4f ms\n", bs, t);
    }

    printf("\nBackward2 pass benchmarks:\n");
    for (int bs: block_sizes) {
      float t = benchmark_kernel(repeat, silu_backward2, dout, x, d_dx, N, bs);
      printf("block_size %4d | time %.4f ms\n", bs, t);
    }
  }

  free(x);
  free(out);
  free(d_out);
  free(dout);
  free(dx);
  free(d_dx);

  return 0;
}
