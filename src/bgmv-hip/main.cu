#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "config.h"
#include "reference.h"
#include "kernels.h"

#define GPU_CHECK(x)                                                    \
  do {                                                                  \
    hipError_t _e = (x);                                                \
    if (_e != hipSuccess) {                                             \
      fprintf(stderr, "HIP error %s:%d  %s\n",                          \
              __FILE__, __LINE__, hipGetErrorString(_e));               \
      exit(1);                                                          \
    }                                                                   \
  } while (0)


int main(int argc, char* argv[])
{
    Config cfg = parse_args(argc, argv);

    const int T  = cfg.num_tokens;
    const int H  = cfg.hidden_size;
    const int R  = cfg.lora_rank;
    const int NL = cfg.num_loras;

    printf("\n=== BGMV Benchmark ===\n");
    printf("  op          : %s\n", cfg.op);
    printf("  num_tokens  : %d\n", T);
    printf("  hidden_size : %d\n", H);
    printf("  lora_rank   : %d\n", R);
    printf("  num_loras   : %d\n", NL);
    printf("  repeat      : %d\n", cfg.repeat);
    printf("  add_to_output  : %s\n", cfg.add_to_output ? "true" : "false");
    printf("------------------------------------------------------------------\n");

    // initialize random data on host
    std::mt19937 rng(19937);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    auto rand_vec_f32 = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    };
    auto f32_to_f16 = [](const std::vector<float>& f) {
        std::vector<__half> h(f.size());
        for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
        return h;
    };

    // lora_indices (per token)
    std::vector<int> h_lora_indices(T);
    std::uniform_int_distribution<int>    lora_dist(0, NL - 1); // valid indices
    for (auto& x : h_lora_indices) x = lora_dist(rng);

    auto alloc_f16 = [](size_t n) {
        __half* p; GPU_CHECK(hipMalloc(&p, n * sizeof(__half))); return p;
    };
    auto alloc_f32 = [](size_t n) {
        float* p; GPU_CHECK(hipMalloc(&p, n * sizeof(float))); return p;
    };

    int* d_lora_indices;
    GPU_CHECK(hipMalloc(&d_lora_indices, T * sizeof(int)));
    GPU_CHECK(hipMemcpy(d_lora_indices, h_lora_indices.data(),
                          T * sizeof(int), hipMemcpyHostToDevice));

    bool is_shrink        = !strcmp(cfg.op, "bgmv_shrink");
    bool is_expand        = !strcmp(cfg.op, "bgmv_expand");

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // BGMV-Shrink:  input[T,H] x weight[NL,R,H] -> output[T,R]
    if (is_shrink) {
        size_t input_size = (size_t)T * H;
        size_t weight_size = (size_t)NL * R * H;
        size_t output_size = (size_t)T * R;

        auto h_inp_f32  = rand_vec_f32(input_size);
        auto h_wt_f32   = rand_vec_f32(weight_size);
        auto h_inp_f16  = f32_to_f16(h_inp_f32);
        auto h_wt_f16   = f32_to_f16(h_wt_f32);

        __half* d_inp = alloc_f16(input_size);
        __half* d_wt  = alloc_f16(weight_size);
        float*  d_out = alloc_f32(output_size);

        GPU_CHECK(hipMemcpy(d_inp, h_inp_f16.data(), input_size * sizeof(__half), hipMemcpyHostToDevice));
        GPU_CHECK(hipMemcpy(d_wt,  h_wt_f16.data(),  weight_size * sizeof(__half), hipMemcpyHostToDevice));

        std::vector<float> h_out_gpu(output_size);
        std::vector<float> h_out_ref(output_size, 0.f);
        ref_bgmv_shrink_cpu(h_out_ref.data(), h_inp_f32.data(), h_wt_f32.data(),
                               h_lora_indices.data(), T, H, R, cfg.scaling);

        // Warmup
        for (int block_size : block_sizes) {
          GPU_CHECK(hipMemset(d_out, 0, output_size * sizeof(float)));
          switch (block_size) {
            case 32: { 
              launch_bgmv_shrink<32>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
              break;
            }
            case 64: { 
              launch_bgmv_shrink<64>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
              break;
            }
            case 128: { 
              launch_bgmv_shrink<128>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
              break;
            }
            case 256: { 
              launch_bgmv_shrink<256>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
              break;
            }
            case 512: { 
              launch_bgmv_shrink<512>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
              break;
            }
            case 1024: { 
              launch_bgmv_shrink<1024>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
              break;
            }
          }

          GPU_CHECK(hipMemcpy(h_out_gpu.data(), d_out, output_size * sizeof(float), hipMemcpyDeviceToHost));

          float max_err = 0.f;
          for (size_t i = 0; i < output_size; i++)
              max_err = fmaxf(max_err, fabsf(h_out_gpu[i] - h_out_ref[i]));
          printf("block_size %4d | correctness check max_err = %e  => %s\n",
                 block_size, max_err, max_err < 0.1f ? "PASS" : "FAIL");
        }

        // Timed loop
        for (int block_size : block_sizes) {
          GPU_CHECK(hipMemset(d_out, 0, output_size * sizeof(float)));
          GPU_CHECK(hipDeviceSynchronize());

          auto start = std::chrono::steady_clock::now();

          for (int i = 0; i < cfg.repeat; i++) {
            switch (block_size) {
              case 32: { 
                launch_bgmv_shrink<32>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
                break;
              }
              case 64: { 
                launch_bgmv_shrink<64>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
                break;
              }
              case 128: { 
                launch_bgmv_shrink<128>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
                break;
              }
              case 256: { 
                launch_bgmv_shrink<256>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
                break;
              }
              case 512: { 
                launch_bgmv_shrink<512>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
                break;
              }
              case 1024: { 
                launch_bgmv_shrink<1024>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.scaling, cfg.vectorize);
                break;
              }
            }
          }

          GPU_CHECK(hipDeviceSynchronize());
          auto end = std::chrono::steady_clock::now();
          auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

          // Bandwidth estimate
          size_t bytes_read  = input_size * sizeof(__half)     // input
                             + weight_size * sizeof(__half)    // weights (worst case all loaded)
                             + T * sizeof(int);
          size_t bytes_write = output_size * sizeof(float);
          double total_bytes = (bytes_read + bytes_write) * cfg.repeat;
          double bw_gb_s     = total_bytes / time;

          printf("bgmv_shrink  | block_size %4d | avg latency: %.3f us  (over %d iters) | bandwidth (approx): %.1f GB/s\n",
                 block_size, time * 1e-3f / cfg.repeat, cfg.repeat, bw_gb_s);

        }

        GPU_CHECK(hipFree(d_inp));
        GPU_CHECK(hipFree(d_wt));
        GPU_CHECK(hipFree(d_out));
    }

    // BGMV-Expand:  input[T,R] x weight[NL,H,R] -> output[T,H]
    else if (is_expand) {
        size_t input_size = (size_t)T * R;
        size_t weight_size = (size_t)NL * H * R;
        size_t output_size = (size_t)T * H;

        auto h_inp_f32 = rand_vec_f32(input_size);
        auto h_wt_f32  = rand_vec_f32(weight_size);
        auto h_wt_f16  = f32_to_f16(h_wt_f32);

        float*  d_inp = alloc_f32(input_size);
        __half* d_wt  = alloc_f16(weight_size);
        __half* d_out = alloc_f16(output_size);

        GPU_CHECK(hipMemcpy(d_inp, h_inp_f32.data(), input_size * sizeof(float), hipMemcpyHostToDevice));
        GPU_CHECK(hipMemcpy(d_wt,  h_wt_f16.data(),  weight_size * sizeof(__half), hipMemcpyHostToDevice));

        std::vector<__half> h_out_gpu(output_size);
        std::vector<float> h_out_ref(output_size, 0.f);
        ref_bgmv_expand_cpu(h_out_ref.data(), h_inp_f32.data(), h_wt_f32.data(),
                            h_lora_indices.data(), T, H, R, cfg.scaling);

        for (int block_size : block_sizes) {

          GPU_CHECK(hipMemset(d_out, 0, output_size * sizeof(__half)));

          switch (block_size) {
            case 32: { 
              launch_bgmv_expand<32>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
              break;
            }
            case 64: { 
              launch_bgmv_expand<64>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
              break;
            }
            case 128: { 
              launch_bgmv_expand<128>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
              break;
            }
            case 256: { 
              launch_bgmv_expand<256>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
              break;
            }
            case 512: { 
              launch_bgmv_expand<512>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
              break;
            }
            case 1024: { 
              launch_bgmv_expand<1024>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
              break;
            }
          }

          GPU_CHECK(hipMemcpy(h_out_gpu.data(), d_out, output_size * sizeof(__half), hipMemcpyDeviceToHost));

          float max_err = 0.f;
          for (size_t i = 0; i < output_size; i++)
              max_err = fmaxf(max_err, fabsf((float)h_out_gpu[i] - h_out_ref[i]));
          printf("block_size %4d | correctness check max_err = %e  => %s\n",
                 block_size, max_err, max_err < 0.1f ? "PASS" : "FAIL");
        }

        // Timed loop
        for (int block_size : block_sizes) {
          GPU_CHECK(hipMemset(d_out, 0, output_size * sizeof(__half)));
          GPU_CHECK(hipDeviceSynchronize());

          auto start = std::chrono::steady_clock::now();

          for (int i = 0; i < cfg.repeat; i++) {
            switch (block_size) { 
              case 32: { 
                launch_bgmv_expand<32>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
                break;
              }
              case 64: { 
                launch_bgmv_expand<64>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
                break;
              }
              case 128: { 
                launch_bgmv_expand<128>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
                break;
              }
              case 256: { 
                launch_bgmv_expand<256>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
                break;
              }
              case 512: { 
                launch_bgmv_expand<512>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
                break;
              }
              case 1024: { 
                launch_bgmv_expand<1024>(d_out, d_inp, d_wt, d_lora_indices, T, H, R, cfg.add_to_output);
                break;
              }
            }
          }
          
          GPU_CHECK(hipDeviceSynchronize());
          auto end = std::chrono::steady_clock::now();
          auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

          size_t bytes_read  = input_size * sizeof(float)  +   // input
                               weight_size * sizeof(__half) +  // weights (worst case all loaded)
                               T * sizeof(int);
          size_t bytes_write = output_size * sizeof(__half);
          double total_bytes = (bytes_read + bytes_write) * cfg.repeat;
          double bw_gb_s     = total_bytes / time;
          printf("bgmv_expand | block_size %4d | add_to_output=%s  avg latency: %.3f us | bandwidth (approx): %.1f GB/s\n",
                 block_size, cfg.add_to_output ? "true" : "false", time * 1e-3f / cfg.repeat, bw_gb_s);
        }

        GPU_CHECK(hipFree(d_inp));
        GPU_CHECK(hipFree(d_wt));
        GPU_CHECK(hipFree(d_out));
    }
    else {
        fprintf(stderr, "Unknown op '%s'\n", cfg.op);
        usage(argv[0]);
        return 1;
    }
    printf("------------------------------------------------------------------\n");

    GPU_CHECK(hipFree(d_lora_indices));
    return 0;
}
