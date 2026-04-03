#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <omp.h>
#include "reference.h"
#include "common.h"

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

template<class D, class T>
void validate_result(D* out_gpu, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
  #pragma omp target update from (out_gpu[0:num_elements]) 
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (std::fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
      printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
      nfaults++;
      if (nfaults >= max_int(10, n_print)) {
        return;
      }
    }
  }
}

void upsample_forward_kernel(
    const float* __restrict__ x,
          float* __restrict__ out,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W, 
    int block_size)
{
  
  auto img_size = H * W;
  auto N = B * C * img_size;
  auto H_out = H*2;
  auto W_out = W*2;
  auto img_out_size = H_out * W_out;

  #pragma omp target teams distribute parallel for num_threads(block_size)
  for (uint64_t flat_idx = 0; flat_idx < N; flat_idx++) {
    auto b = flat_idx / (C * img_size);
    auto c = (flat_idx / img_size) % C;
    auto i = (flat_idx / W) % H;
    auto j = flat_idx % W;

    x   += b * C * img_size + c * img_size + i * W + j;
    out += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

    float x_val = x[0];

    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        out[ii * W_out + jj] = x_val;
      }
    }
  }
}

void upsample_forward_kernel2(const float* __restrict x,
                                    float* __restrict out,
                              uint64_t B, uint64_t C, uint64_t H, uint64_t W,
                              int block_size)
{
  const uint64_t H_out = H * 2;
  const uint64_t W_out = W * 2;

  #pragma omp target teams distribute parallel for collapse(3) num_threads(block_size)
  for (uint64_t bc = 0; bc < B * C; ++bc) {
    for (uint64_t in_y = 0; in_y < H; ++in_y) {
      for (uint64_t in_x = 0; in_x < W; ++in_x) {

        const uint64_t b = bc / C;
        const uint64_t c = bc % C;

        const float val = x[((b * C + c) * H + in_y) * W + in_x];

        const uint64_t out_base =
          ((b * C + c) * H_out + (in_y * 2)) * W_out + (in_x * 2);

        float* o = out + out_base;
        o[0] = val;
        o[1] = val;
        o[W_out] = val;
        o[W_out + 1] = val;
      }
    }
  }
}


void upsample_backward_kernel(
    const float* __restrict__ dout,
          float* __restrict__ dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W,
    int block_size)
{
  auto img_in_size = H * W;
  auto N = B * C * img_in_size;
  auto H_out = H*2;
  auto W_out = W*2;
  auto img_out_size = H_out * W_out;

  #pragma omp target teams distribute parallel for num_threads(block_size)
  for (uint64_t flat_idx = 0; flat_idx < N; flat_idx++) {

    auto b = flat_idx / (C * img_in_size);
    auto c = (flat_idx / img_in_size) % C;
    auto i = (flat_idx / W) % H;
    auto j = flat_idx % W;

    // move pointers
    dx += b * C * img_in_size + c * img_in_size + i * W + j;
    dout += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

    float dout_sum = 0.0f;
    for (int ii = 0; ii < 2; ii++) {
      for (int jj = 0; jj < 2; jj++) {
        dout_sum += dout[ii * W_out + jj];
      }
    }
    dx[0] = dout_sum;
  }
}

void upsample_backward_kernel2(
    const float* __restrict__ dout,
          float* __restrict__ dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W,
    int block_size)
{
  auto W_out = W * 2;
  auto H_out = H * 2;

  #pragma omp target teams distribute parallel for collapse(3) num_threads(block_size)
  for (uint64_t bc = 0; bc < B * C; ++bc) {
    for (uint64_t in_y = 0; in_y < H; ++in_y) {
      for (uint64_t in_x = 0; in_x < W; ++in_x) {

        const uint64_t b = bc / C;
        const uint64_t c = bc % C;

        // Base pointer into dout for this thread's 2x2 tile
        const auto out_base = b * (C * H_out * W_out)
                                + c * (H_out * W_out)
                                + (in_y * 2) * W_out
                                + (in_x * 2);

        auto row0 = dout + out_base;
        auto row1 = row0 + W_out;
        auto grad = row0[0] + row0[1] + row1[0] + row1[1];

        dx[(b * C  + c) * H * W + in_y * W + in_x] = grad;
      }
    }
  }
}

void upsample_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size)
{
  upsample_forward_kernel(x, out, B, C, H, W, block_size);
}

void upsample_forward2(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size_x,
    int block_size_y)
{
  int block_size = block_size_x * block_size_y;
  upsample_forward_kernel2(x, out, B, C, H, W, block_size);
}

void upsample_backward1(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size)
{
  upsample_backward_kernel(dout, dx, B, C, H, W, block_size);
}

void upsample_backward2(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size_x,
    int block_size_y)
{
  int block_size = block_size_x * block_size_y;
  upsample_backward_kernel2(dout, dx, B, C, H, W, block_size);
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Usage: %s <batch size> <number of channels> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }
  int B = atoi(argv[1]);
  int C = atoi(argv[2]);
  int H = atoi(argv[3]);
  int W = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  srand(0);
  uint64_t S = (uint64_t)B * C * H * W;
  float* x = make_random_float(S);
  float* dout = make_random_float(S * 4);
  float* out = (float*)malloc(S * 4 * sizeof(float));
  float* d_out = (float*)malloc(S * 4 * sizeof(float)); // forward device result
  float* dx = (float*)malloc(S * sizeof(float));
  float* d_dx = (float*)malloc(S * sizeof(float)); // backward device result

  upsample_forward_reference(x, out, B, C, H, W);

  #pragma omp target data map(to: x[0:S], dout[0:S*4]) \
                          map(alloc: d_out[0:S*4], d_dx[0:S])
  {
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    printf("Checking forward pass\n");
    for (int block_size : block_sizes) {
      printf("Checking block size %d\n", block_size);
      upsample_forward1(d_out, x, B, C, H, W, block_size);
      validate_result(d_out, out, "out", B * C * (H/2) * (W/2));
    }
    printf("Forward1 pass: all results match\n\n");

    printf("Forward1 pass benchmarks:\n");
    for (int block_size : block_sizes) {

      auto elapsed_time = benchmark_kernel(repeat, upsample_forward1, d_out, x, B, C, H, W, block_size);

      float gflops = (float)S / elapsed_time * 1e3f / 1e9f;
      printf("block_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed_time, gflops);
    }

    printf("\n─────────────────────────────────────────────────────\n");
    int block2D_sizes[] = {8, 16, 32};
    for (int block_size : block2D_sizes) {
      printf("Checking block size %d\n", block_size);
      upsample_forward2(d_out, x, B, C, H, W, block_size, block_size);
      validate_result(d_out, out, "out", B * C * (H/2) * (W/2));
    }
    printf("Forward2 pass: all results match\n\n");
    printf("Forward2 pass benchmarks:\n");
    for (int block_size : block2D_sizes) {

      auto elapsed_time = benchmark_kernel(repeat, upsample_forward2, d_out, x, B, C, H, W, block_size, block_size);

      float gflops = (float)S / elapsed_time * 1e3f / 1e9f;
      printf("block2D_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed_time, gflops);
    }

    printf("\n─────────────────────────────────────────────────────\n");
    printf("Checking backward pass\n");
    upsample_backward_reference(dout, dx, B, C, H, W);
    for (int block_size : block_sizes) {
      printf("Checking block size %d\n", block_size);
      upsample_backward1(d_dx, dout, B, C, H, W, block_size);
      validate_result(d_dx, dx, "dx", S);
    }
    printf("Backward pass: all results match\n\n");
    printf("All results match. Starting benchmarks.\n\n");

    printf("\nBackward pass benchmarks:\n");
    for (int block_size : block_sizes) {

      auto elapsed_time = benchmark_kernel(repeat, upsample_backward1, d_dx, dout, B, C, H, W, block_size);

      float gflops = (float)S / elapsed_time * 1e3f / 1e9f;
      printf("block_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed_time, gflops);
    }

    printf("\n─────────────────────────────────────────────────────\n");
    printf("Checking backward2 pass\n");
    for (int block_size : block2D_sizes) {
      printf("Checking block size %d\n", block_size);
      upsample_backward2(d_dx, dout, B, C, H, W, block_size, block_size);
      validate_result(d_dx, dx, "dx", S);
    }
    printf("Backward2 pass: all results match\n\n");
    printf("All results match. Starting benchmarks.\n\n");

    printf("\nBackward2 pass benchmarks:\n");
    for (int block_size : block2D_sizes) {

      auto elapsed_time = benchmark_kernel(repeat, upsample_backward2, d_dx, dout, B, C, H, W, block_size, block_size);

      float gflops = (float)S / elapsed_time * 1e3f / 1e9f;
      printf("block2D_size %4d | time %.4f ms | gflops %.2f\n", block_size, elapsed_time, gflops);
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
