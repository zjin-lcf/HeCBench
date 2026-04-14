#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "reference.h"
#include "common.h"

template<class Kernel, class... KernelArgs>
float benchmark_kernel(sycl::queue &q, int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

template<class D, class T>
void validate_result(sycl::queue &q,
                     D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
  D* out_gpu = (D*)malloc(num_elements * sizeof(D));
  q.memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();
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

void silu_forward_kernel(const float* x, float* out, uint64_t N,
                         sycl::nd_item<1> &item)
{
  uint64_t i = item.get_global_id(0);
  if (i >= N) return;
  float x_val = x[i];
  out[i] = x_val / (1.0f + sycl::exp(-x_val));
}

void silu_forward_kernel2(const float* x, float* out, uint64_t N,
                         sycl::nd_item<1> &item)
{
  uint64_t i = item.get_global_id(0);
  uint64_t vec_count = N / 4;
  if (i < vec_count) {
    sycl::float4 x4  = reinterpret_cast<const sycl::float4*>(x)[i];
    sycl::float4 t4;
    t4.x() = x4.x() / (1.0f + sycl::exp(-x4.x()));
    t4.y() = x4.y() / (1.0f + sycl::exp(-x4.y()));
    t4.z() = x4.z() / (1.0f + sycl::exp(-x4.z()));
    t4.w() = x4.w() / (1.0f + sycl::exp(-x4.w()));
    reinterpret_cast<sycl::float4*>(out)[i] = t4;
  }
  for (uint64_t j = 4 * vec_count + i; j < N; j += item.get_local_range(0)) {
    float x_val = x[j];
    out[j] = x_val / (1.0f + sycl::exp(-x_val));
  }
}

void silu_backward_kernel(const float* dout, const float* x, float* dx, uint64_t N,
                          sycl::nd_item<1> &item)
{
  uint64_t i = item.get_global_id(0);
  if (i >= N) return;
  float out_val = dout[i];
  float x_val = x[i];
  float expx = sycl::exp(-x_val);
  float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);
  dx[i] = out_val * grad_silu;
}

void silu_backward_kernel2(const float* dout, const float* x, float* dx, uint64_t N,
                           sycl::nd_item<1> &item)
{
  uint64_t i = item.get_global_id(0);
  if (i >= N) return;
  float x_val = x[i];
  // sigmoid: 1 / (1 + exp(-x))
  float sig = 1.0f / (1.0f + sycl::exp(-x_val));
  // gradient: sig * (1 + x * (1 - sig))
  float grad_silu = sig * (1.0f + x_val * (1.0f - sig));
  dx[i] = dout[i] * grad_silu;
}

void silu_backward_kernel3(const float* dout, const float* x, float* dx, uint64_t N,
                           sycl::nd_item<1> &item)
{
  uint64_t i = item.get_global_id(0);
  uint64_t vec_count = N / 4;
  if (i < vec_count) {
    sycl::float4 x4  = reinterpret_cast<const sycl::float4*>(x)[i];
    sycl::float4 do4  = reinterpret_cast<const sycl::float4*>(dout)[i];
    sycl::float4 sig4, grad_silu4;
    sig4.x() = 1.0f / (1.0f + sycl::exp(-x4.x()));
    sig4.y() = 1.0f / (1.0f + sycl::exp(-x4.y()));
    sig4.z() = 1.0f / (1.0f + sycl::exp(-x4.z()));
    sig4.w() = 1.0f / (1.0f + sycl::exp(-x4.w()));
    grad_silu4.x() = sig4.x() * (1.0f + x4.x() * (1.0f - sig4.x()));
    grad_silu4.y() = sig4.y() * (1.0f + x4.y() * (1.0f - sig4.y()));
    grad_silu4.z() = sig4.z() * (1.0f + x4.z() * (1.0f - sig4.z()));
    grad_silu4.w() = sig4.w() * (1.0f + x4.w() * (1.0f - sig4.w()));
    sig4.x() = do4.x() * grad_silu4.x();
    sig4.y() = do4.y() * grad_silu4.y();
    sig4.z() = do4.z() * grad_silu4.z();
    sig4.w() = do4.w() * grad_silu4.w();
    reinterpret_cast<sycl::float4*>(dx)[i] = sig4;
  }
  for (uint64_t j = 4 * vec_count + i; j < N; j += item.get_local_range(0)) {
    float x_val = x[j];
    float sig = 1.0f / (1.0f + sycl::exp(-x_val));
    float grad_silu = sig * (1.0f + x_val * (1.0f - sig));
    dx[j] = dout[j] * grad_silu;
  }
}

void silu_forward(
    sycl::queue &q,
    const float* x,
    float* out,
    uint64_t N,
    int block_size)
{
  int gridDim = ceil_div(N, block_size);
  q.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(gridDim * block_size),
                      sycl::range<1>(block_size)),
    [=](sycl::nd_item<1> item) {
      silu_forward_kernel(x, out, N, item);
  });
}

void silu_forward2(
    sycl::queue &q,
    const float* x,
    float* out,
    uint64_t N,
    int block_size)
{
  int gridDim = ceil_div(N, block_size);
  q.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(gridDim * block_size),
                      sycl::range<1>(block_size)),
    [=](sycl::nd_item<1> item) {
      silu_forward_kernel2(x, out, N, item);
  });
}

void silu_backward(
    sycl::queue &q,
    const float* dout,
    const float* x,
    float* dx,
    uint64_t N,
    int block_size)
{
  int gridDim = ceil_div(N, block_size);
  q.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(gridDim * block_size),
                      sycl::range<1>(block_size)),
    [=](sycl::nd_item<1> item) {
      silu_backward_kernel(dout, x, dx, N, item);
  });
}

void silu_backward2(
    sycl::queue &q,
    const float* dout,
    const float* x,
    float* dx,
    uint64_t N,
    int block_size)
{
  int gridDim = ceil_div(N, block_size);
  q.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(gridDim * block_size),
                      sycl::range<1>(block_size)),
    [=](sycl::nd_item<1> item) {
      silu_backward_kernel2(dout, x, dx, N, item);
  });
}

void silu_backward3(
    sycl::queue &q,
    const float* dout,
    const float* x,
    float* dx,
    uint64_t N,
    int block_size)
{
  int gridDim = ceil_div(N, block_size);
  q.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(gridDim * block_size),
                      sycl::range<1>(block_size)),
    [=](sycl::nd_item<1> item) {
      silu_backward_kernel3(dout, x, dx, N, item);
  });
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_x, *d_out, *d_dout, *d_dx;
  d_x = sycl::malloc_device<float>(N, q);
  d_out = sycl::malloc_device<float>(N, q);
  d_dout = sycl::malloc_device<float>(N, q);
  d_dx = sycl::malloc_device<float>(N, q);
  q.memcpy(d_x, x, N * sizeof(float));
  q.memcpy(d_dout, dout, N * sizeof(float));

  int block_sizes[] = {64, 128, 256, 512, 1024};
  printf("Checking forward pass\n");
  for (int block_size : block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_forward(q, d_x, d_out, N, block_size);
    validate_result(q, d_out, out, "out", N);
  }

  printf("Checking forward2 pass\n");
  for (int block_size : block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_forward2(q, d_x, d_out, N, block_size);
    validate_result(q, d_out, out, "out", N);
  }

  printf("Checking backward pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_backward(q, d_dout, d_x, d_dx, N, block_size);
    validate_result(q, d_dx, dx, "dx", N);
  }
  printf("Checking backward2 pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_backward2(q, d_dout, d_x, d_dx, N, block_size);
    validate_result(q, d_dx, dx, "dx", N);
  }
  printf("Checking backward3 pass\n");
  for (int block_size: block_sizes) {
    printf("Checking block size %d\n", block_size);
    silu_backward3(q, d_dout, d_x, d_dx, N, block_size);
    validate_result(q, d_dx, dx, "dx", N);
  }

  printf("\nForward pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(q, repeat, silu_forward, q, d_x, d_out, N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nForward2 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(q, repeat, silu_forward2, q, d_x, d_out, N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nBackward pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(q, repeat, silu_backward, q,
                                          d_dout, d_x, d_dx,
                                          N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nBackward2 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(q, repeat, silu_backward2, q,
                                          d_dout, d_x, d_dx,
                                          N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  printf("\nBackward3 pass benchmarks:\n");
  for (int block_size: block_sizes) {
    float elapsed_time = benchmark_kernel(q, repeat, silu_backward3, q,
                                          d_dout, d_x, d_dx,
                                          N, block_size);

    printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
  }

  sycl::free(d_x, q);
  sycl::free(d_out, q);
  sycl::free(d_dout, q);
  sycl::free(d_dx, q);

  free(x);
  free(out);
  free(dout);
  free(dx);

  return 0;
}
