/*
Kernels for layernorm forward pass.
*/

#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <chrono>
#include "common.hpp"
#include "reference.h"

void layernorm_forward_kernel1(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float m = sum / C;
    if(warp.leader() && mean != nullptr) {
        mean[idx] = m;
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if(warp.leader() && rstd != nullptr) {
        rstd[idx] = s;
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.get_local_linear_id(); c < C; c += warp.get_max_local_range()[0]) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
void layernorm_forward_kernel2(sycl::nd_item<1> id, float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                               const float*  __restrict__ inp, const float*  __restrict__ weight,
                               const float* __restrict__ bias, int N, int C) {
    sycl::sub_group warp = id.get_sub_group();
    int idx = id.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0f;  // stores sum(x)
    float sum2 = 0.0f; // stores sum(x**2)
    for (int i = warp.get_local_linear_id(); i < C; i += warp.get_max_local_range()[0]) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});   // sum(x)
    sum2 = sycl::reduce_over_group(warp, sum2, sycl::plus<float>{}); // sum(x**2)
    sum /= C;   // mean(x)
    sum2 /= C;  // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = sycl::rsqrt(var + 1e-5f);

    // store the mean, no need to cache it
    if(warp.leader() && mean != nullptr) {
        mean[idx] = m;
    }
    // store the rstd, no need to cache it
    if(warp.leader() && rstd != nullptr) {
        rstd[idx] = s;
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.get_local_linear_id(); c < C; c += warp.get_max_local_range()[0]) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel1(id, out, mean, rstd, inp, weight, bias, N, C);
    }).wait();
}

void layernorm_forward2(sycl::queue &q, float* out, float* mean, float* rstd,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C,
                        const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    q.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> id) {
        layernorm_forward_kernel2(id, out, mean, rstd, inp, weight, bias, N, C);
    }).wait();
}

// kernel version dispatch
void layernorm_forward(int kernel_num,
                       sycl::queue &q,
                       float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(q, out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            std::cout << "Invalid kernel number\n";
            std::exit(1);
    }
}

// ----------------------------------------------------------------------------
// Main

int main(int argc, char** argv) {
    srand(0);

    int B = 8; // batch size
    int T = 1024; // sequence length
    int C = 768; // embedding size

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // Device memory allocation
    float* d_out = sycl::malloc_device<float>(B * T * C, q);
    float* d_mean = sycl::malloc_device<float>(B * T, q);
    float* d_rstd = sycl::malloc_device<float>(B * T, q);
    float* d_inp = sycl::malloc_device<float>(B * T * C, q);
    float* d_weight = sycl::malloc_device<float>(C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);

    // Copy data to device
    q.memcpy(d_inp, inp, B * T * C * sizeof(float)).wait();
    q.memcpy(d_weight, weight, C * sizeof(float)).wait();
    q.memcpy(d_bias, bias, C * sizeof(float)).wait();

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1)
        kernel_num = atoi(argv[1]);
    std::cout << "Using kernel version: " << kernel_num << std::endl;

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int block_size : block_sizes) {
        std::cout << "Checking block size " <<  block_size << '\n';

        layernorm_forward(kernel_num, q, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    std::cout << "All results match. Starting benchmarks.\n\n";

    // time the kernel at different block sizes
    for (int block_size : block_sizes) {
        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(
                repeat_times,
                layernorm_forward, // kernel
                // kernel params
                kernel_num, q, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                B, T, C, block_size
        );

        // napkin math: estimate the memory bandwidth achieved
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        std::cout << "block_size " << block_size << " | time " << elapsed_time << " ms | bandwidth " << memory_bandwidth << " GB/s" << std::endl;
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);

    sycl::free(d_out, q);
    sycl::free(d_mean, q);
    sycl::free(d_rstd, q);
    sycl::free(d_inp, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);

    return 0;
}
