#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "config.h"
#include "reference.h"
#include "kernels.h"

int main(int argc, char* argv[])
{
    Config cfg = parse_args(argc, argv);

    const int T  = cfg.num_tokens;
    const int H  = cfg.hidden_size;
    const int R  = cfg.lora_rank;
    const int NL = cfg.num_loras;

    printf("\n=== BGMV Benchmark (OpenMP Target Offloading) ===\n");
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
        std::vector<half> h(f.size());
        for (size_t i = 0; i < f.size(); i++) h[i] = half(f[i]);
        return h;
    };

    // lora_indices (per token)
    std::vector<int> h_lora_indices_vector(T);
    std::uniform_int_distribution<int> lora_dist(0, NL - 1); // valid indices
    for (auto& x : h_lora_indices_vector) x = lora_dist(rng);

    auto h_lora_indices = h_lora_indices_vector.data();

    #pragma omp target enter data map(to: h_lora_indices[0:T])

    bool is_shrink = !strcmp(cfg.op, "bgmv_shrink");
    bool is_expand = !strcmp(cfg.op, "bgmv_expand");

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // BGMV-Shrink:  input[T,H] x weight[NL,R,H] -> output[T,R]
    if (is_shrink) {
        size_t input_size = (size_t)T * H;
        size_t weight_size = (size_t)NL * R * H;
        size_t output_size = (size_t)T * R;

        auto h_inp_f32_vector = rand_vec_f32(input_size);
        auto h_wt_f32_vector = rand_vec_f32(weight_size);
        auto h_inp_f16_vector = f32_to_f16(h_inp_f32_vector);
        auto h_wt_f16_vector = f32_to_f16(h_wt_f32_vector);

        std::vector<float> h_out_ref_vector(output_size, 0.f);
        std::vector<float> h_out_vector(output_size); // device

        auto *h_out_ref = h_out_ref_vector.data();
        auto *h_out = h_out_vector.data();
        auto *h_inp_f32 = h_inp_f32_vector.data();
        auto *h_inp_f16 = h_inp_f16_vector.data();
        auto *h_wt_f32 = h_wt_f32_vector.data();
        auto *h_wt_f16 = h_wt_f16_vector.data();

        // Output on host for reference
        ref_bgmv_shrink_cpu(h_out_ref, h_inp_f32, h_wt_f32,
                            h_lora_indices, T, H, R, cfg.scaling);

        #pragma omp target enter data map(to: h_inp_f16[0:input_size], h_wt_f16[0:weight_size]) \
                                      map(alloc: h_out[0:output_size])

        // Warmup and correctness check
        for (int block_size : block_sizes) {
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < output_size; i++)
                h_out[i] = 0.f;
            
            // Launch kernel with specific block size (team size)
            launch_bgmv_shrink(h_out, h_inp_f16, h_wt_f16,
                               h_lora_indices, T, H, R, cfg.scaling, cfg.vectorize, block_size);

            #pragma omp target update from (h_out[0:output_size])
            float max_err = 0.f;
            for (size_t i = 0; i < output_size; i++)
                max_err = fmaxf(max_err, fabsf(h_out[i] - h_out_ref[i]));
            printf("block_size %4d | correctness check max_err = %e  => %s\n",
                   block_size, max_err, max_err < 0.1f ? "PASS" : "FAIL");
        }

        // Timed loop
        for (int block_size : block_sizes) {
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < output_size; i++)
                h_out[i] = 0.f;
            
            auto start = std::chrono::steady_clock::now();

            for (int i = 0; i < cfg.repeat; i++) {
                
                launch_bgmv_shrink(h_out, h_inp_f16, h_wt_f16,
                                   h_lora_indices, T, H, R, cfg.scaling, cfg.vectorize, block_size);
            }

            auto end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            // Bandwidth estimate
            size_t bytes_read = input_size * sizeof(half)     // input
                                   + weight_size * sizeof(half)    // weights (worst case all loaded)
                                   + T * sizeof(int);
            size_t bytes_write = output_size * sizeof(float);
            double total_bytes = (bytes_read + bytes_write) * cfg.repeat;
            double bw_gb_s = total_bytes / time;

            printf("bgmv_shrink  | block_size %4d | avg latency: %.3f us  (over %d iters) | bandwidth (approx): %.1f GB/s\n",
                   block_size, time * 1e-3f / cfg.repeat, cfg.repeat, bw_gb_s);
        }
        #pragma omp target exit data map(delete: h_inp_f16[0:input_size], h_wt_f16[0:weight_size], \
                                                 h_out[0:output_size])
    }
    // BGMV-Expand:  input[T,R] x weight[NL,H,R] -> output[T,H]
    else if (is_expand) {
        size_t input_size = (size_t)T * R;
        size_t weight_size = (size_t)NL * H * R;
        size_t output_size = (size_t)T * H;

        auto h_inp_f32_vector = rand_vec_f32(input_size);
        auto h_wt_f32_vector = rand_vec_f32(weight_size);
        auto h_wt_f16_vector = f32_to_f16(h_wt_f32_vector);

        std::vector<float> h_out_ref_vector(output_size, 0.f);
        std::vector<half> h_out_vector(output_size); // device

        auto *h_out_ref = h_out_ref_vector.data();
        auto *h_out = h_out_vector.data();
        auto *h_inp_f32 = h_inp_f32_vector.data();
        auto *h_wt_f32 = h_wt_f32_vector.data();
        auto *h_wt_f16 = h_wt_f16_vector.data();

        ref_bgmv_expand_cpu(h_out_ref, h_inp_f32, h_wt_f32,
                            h_lora_indices, T, H, R, cfg.add_to_output);

        #pragma omp target enter data map(to: h_inp_f32[0:input_size], h_wt_f16[0:weight_size]) \
                                      map(alloc: h_out[0:output_size])

        // Warmup and correctness check
        for (int block_size : block_sizes) {
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < output_size; i++)
                h_out[i] = half(0.f);
            
            launch_bgmv_expand(h_out, h_inp_f32, h_wt_f16,
                               h_lora_indices, T, H, R, cfg.add_to_output, block_size);
            
            #pragma omp target update from (h_out[0:output_size])

            float max_err = 0.f;
            for (size_t i = 0; i < output_size; i++)
                max_err = fmaxf(max_err, fabsf((float)h_out[i] - h_out_ref[i]));
            printf("block_size %4d | correctness check max_err = %e  => %s\n",
                   block_size, max_err, max_err < 0.1f ? "PASS" : "FAIL");
        }

        // Timed loop
        for (int block_size : block_sizes) {
            #pragma omp target teams distribute parallel for
            for (size_t i = 0; i < output_size; i++)
                h_out[i] = half(0.f);
            
            auto start = std::chrono::steady_clock::now();

            for (int i = 0; i < cfg.repeat; i++) {
                
                launch_bgmv_expand(h_out, h_inp_f32, h_wt_f16,
                                   h_lora_indices, T, H, R, cfg.add_to_output, block_size);
            }

            auto end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            size_t bytes_read = input_size * sizeof(float)  +   // input
                               weight_size * sizeof(half) +  // weights (worst case all loaded)
                               T * sizeof(int);
            size_t bytes_write = output_size * sizeof(half);
            double total_bytes = (bytes_read + bytes_write) * cfg.repeat;
            double bw_gb_s = total_bytes / time;
            printf("bgmv_expand | block_size %4d | add_to_output=%s  avg latency: %.3f us | bandwidth (approx): %.1f GB/s\n",
                   block_size, cfg.add_to_output ? "true" : "false", time * 1e-3f / cfg.repeat, bw_gb_s);
        }
        #pragma omp target exit data map(delete: h_inp_f32[0:input_size], h_wt_f16[0:weight_size], \
                                                 h_out[0:output_size])
    }
    else {
        fprintf(stderr, "Unknown op '%s'\n", cfg.op);
        usage(argv[0]);
        return 1;
    }
    printf("------------------------------------------------------------------\n");

    #pragma omp target exit data map(release: h_lora_indices[0:T])
    return 0;
}
