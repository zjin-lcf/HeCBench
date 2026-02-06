/*
Kernels for layernorm forward pass.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include "reference.h"
#include "utils.cuh"
#include "reduce.cuh"

template <int UNROLL>
__global__
void layernorm_forward_kernel0(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C)
{
    const int BLOCKSIZE = blockDim.x;
    const int bid       = blockIdx.x;
    const int warp_id   = threadIdx.x / THREADS_PER_WARP;
    const int lane_id   = threadIdx.x % THREADS_PER_WARP;

    const float *input_ptr  = inp + bid * C;
    const float *weight_ptr = weight;
    const float *bias_ptr   = bias;
    float       *output_ptr = out + bid * C;

    const int start_offset = warp_id * THREADS_PER_WARP * UNROLL + lane_id * UNROLL;
    float     ld_input_regs[UNROLL];
    float     local_sum = 0.0f;
    for (int64_t offset = start_offset; offset < C; offset += (BLOCKSIZE * UNROLL)) {
        load_data<float, UNROLL>(input_ptr + offset, ld_input_regs);
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float val = static_cast<float>(ld_input_regs[i]);
            local_sum += val;
        }
    }

    const float mean_sum = BlockReduce<SumOp, float>(local_sum) / static_cast<float>(C);
    if (threadIdx.x == 0)
      __stcs(mean + bid, mean_sum);

    local_sum = 0.0f;
    for (int64_t offset = start_offset; offset < C; offset += (BLOCKSIZE * UNROLL)) {
        load_data<float, UNROLL>(input_ptr + offset, ld_input_regs);
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            const float diff = static_cast<float>(ld_input_regs[i]) - mean_sum;
            local_sum += diff * diff;
        }
    }
    const float mean_square = BlockReduce<SumOp, float>(local_sum) / static_cast<float>(C);
    float s = rsqrtf(mean_square + 1e-5f);
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + bid, s);
    }

    float st_regs[UNROLL];
    float ld_weight_regs[UNROLL];
    float ld_bias_regs[UNROLL];

    for (int64_t offset = start_offset; offset < C; offset += (BLOCKSIZE * UNROLL)) {
        load_data<float, UNROLL>(input_ptr + offset, ld_input_regs);
        load_data<float, UNROLL>(weight_ptr + offset, ld_weight_regs);
        load_data<float, UNROLL>(bias_ptr + offset, ld_bias_regs);

#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            float n = (static_cast<float>(ld_input_regs[i]) - mean_sum) * s;
            st_regs[i] = n * static_cast<float>(ld_weight_regs[i]) +
                         static_cast<float>(ld_bias_regs[i]);
        }
        store_data<float, UNROLL>(output_ptr + offset, st_regs);
    }
}

__global__ void layernorm_forward_kernel1(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

// same as kernel 1 but uses var(x) == mean(x**2) - mean(x)**2
__global__ void layernorm_forward_kernel2(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0; // stores sum(x)
    float sum2 = 0.0; // stores sum(x**2)
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = cg::reduce(warp, sum, cg::plus<float>{}); // sum(x)
    sum2 = cg::reduce(warp, sum2, cg::plus<float>{}); // sum(x**2)
    sum /= C; // mean(x)
    sum2 /= C; // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = rsqrtf(var + 1e-5f);

    // store the mean, no need to cache it
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

void layernorm_forward0(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    constexpr int UNROLL = sizeof(float4) / sizeof(float);
    if (C % UNROLL == 0)
      layernorm_forward_kernel0<UNROLL><<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    else
      layernorm_forward_kernel0<1><<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaDeviceSynchronize());
}

void layernorm_forward1(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaDeviceSynchronize());
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel2<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaDeviceSynchronize());
}


// kernel version dispatch
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 0:
            layernorm_forward0(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {

  if (argc != 5) {
    printf("Usage: %s <batch size> <sequence length> <channel length> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t B = atoi(argv[1]);
  const size_t T = atoi(argv[2]);
  const size_t C = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  // create host memory of random numbers
  srand(0);
  float* out = (float*)malloc(B * T * C * sizeof(float));
  float* mean = (float*)malloc(B * T * sizeof(float));
  float* rstd = (float*)malloc(B * T * sizeof(float));
  float* inp = make_random_float(B * T * C);
  float* weight = make_random_float(C);
  float* bias = make_random_float(C);

  // move to GPU
  float* d_out;
  float* d_mean;
  float* d_rstd;
  float* d_inp;
  float* d_weight;
  float* d_bias;
  cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
  cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
  cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
  cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

  const int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  // evaluate kernels
  for (int kernel_num = 0; kernel_num < 3; kernel_num++) {
    printf("Using kernel %d\n", kernel_num);

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int block_size : block_sizes) {
      printf("Checking block size %d.\n", block_size);

      layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

      validate_result(d_out, out, "out", B * T * C, 1e-5f);
      validate_result(d_mean, mean, "mean", B * T, 1e-5f);
      validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int block_size : block_sizes) {

      float elapsed_time = benchmark_kernel(repeat, layernorm_forward,
                                            kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                            B, T, C, block_size);

      // estimate the memory bandwidth achieved
      // e.g. A100 40GB PCIe is advertised at 1,555GB/s
      long memory_ops = (2 * B * T * C) * 4; // *4 for float
      float memory_bandwidth = memory_ops / elapsed_time / 1e6;

      printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }
  }

  // free memory
  free(out);
  free(mean);
  free(rstd);
  free(inp);
  free(weight);
  free(bias);
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_mean));
  cudaCheck(cudaFree(d_rstd));
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_weight));
  cudaCheck(cudaFree(d_bias));

  return 0;
}
