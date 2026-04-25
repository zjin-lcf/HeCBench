#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "utils.h"
#include "kernels.h"

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void reshape_and_cache(sycl::queue &q,
                       int num_tokens, int num_heads, int head_size,
                       int block_size, int x, int repeat) try {
    assert(head_size % x == 0);

    int key_stride = num_heads * head_size;
    int value_stride = key_stride;
    int headsize_div_x = head_size / x;

    uint64_t key_size = (uint64_t)num_tokens * num_heads * head_size;
    uint64_t value_size = key_size;

    uint64_t num_blocks = (num_tokens + block_size - 1) / block_size;

    uint64_t key_cache_size = num_blocks * num_heads * head_size * block_size;

    uint64_t value_cache_size = key_cache_size;

    std::vector<scalar_t> h_key(key_size);
    std::vector<scalar_t> h_value(value_size);
    std::vector<int64_t> h_slot_mapping(num_tokens);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    scalar_t key_max = 0, val_max = 0;
    for (uint64_t i = 0; i < key_size; i++) {
        h_key[i] = dist(gen);
        h_value[i] = dist(gen);
        key_max = std::max(key_max, h_key[i]);
        val_max = std::max(val_max, h_value[i]);
    }
    float k_scale = (float)key_max / 64.f;
    float v_scale = (float)val_max / 64.f;

    for (int i = 0; i < num_tokens; i++) {
        h_slot_mapping[i] = i;
    }

    std::shuffle(h_slot_mapping.begin(), h_slot_mapping.end(), gen);

    scalar_t *d_key = sycl::malloc_device<scalar_t>(key_size, q);
    scalar_t *d_value = sycl::malloc_device<scalar_t>(value_size, q);
    cache_t *d_key_cache = sycl::malloc_device<cache_t>(key_cache_size, q);
    cache_t *d_value_cache = sycl::malloc_device<cache_t>(value_cache_size, q);
    int64_t* d_slot_mapping = sycl::malloc_device<int64_t>(num_tokens, q);

    q.memset(d_key_cache, 0, key_cache_size * sizeof(cache_t));
    q.memset(d_key_cache, 0, value_cache_size * sizeof(cache_t));
    q.memcpy(d_key, h_key.data(), key_size * sizeof(scalar_t));
    q.memcpy(d_value, h_value.data(), value_size * sizeof(scalar_t));
    q.memcpy(d_slot_mapping, h_slot_mapping.data(), num_tokens * sizeof(int64_t));

    int threadsPerBlock = std::min(num_heads * headsize_div_x, 512);
    sycl::range<1> gws (threadsPerBlock * num_tokens);
    sycl::range<1> lws (threadsPerBlock);

    auto kFn = [=](sycl::nd_item<1> item) {
      reshape_and_cache_kernel<scalar_t, cache_t, kv_dt>(
        d_key, d_value, d_key_cache, d_value_cache, d_slot_mapping,
        key_stride, value_stride, num_heads, head_size, block_size,
        x, k_scale, v_scale, item);
    };

    // warmup
    for (int i = 0; i < 30; i++) {
      q.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
    }

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of reshape and cache kernel: %f (us)\n", (time * 1e-3f) / repeat);

    uint64_t bytes = sizeof(cache_t) * (key_cache_size + value_cache_size) +
                     sizeof(int64_t) * num_tokens +
                     sizeof(scalar_t) * (key_size + value_size);
    printf("Average bandwidth of reshape and cache kernel: %f (GB/s)\n", bytes * repeat * 1.f / time);

    std::vector<cache_t> h_key_cache(key_cache_size);
    std::vector<cache_t> h_value_cache(value_cache_size);

    q.memcpy(h_key_cache.data(), d_key_cache, key_cache_size * sizeof(cache_t));

    q.memcpy(h_value_cache.data(), d_value_cache, value_cache_size * sizeof(cache_t));
    q.wait();

    sycl::free(d_key, q);
    sycl::free(d_value, q);
    sycl::free(d_key_cache, q);
    sycl::free(d_value_cache, q);
    sycl::free(d_slot_mapping, q);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <number of tokens> <number of heads> <head size> <repeat>\n", argv[0]);
        return 1;
    }
    const int num_tokens = atoi(argv[1]);
    const int num_heads = atoi(argv[2]);
    const int head_size = atoi(argv[3]);
    const int repeat = atoi(argv[4]);

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    for (int block_size = 16; block_size <= 32; block_size *= 2) {
        for (int x = 1; x <= 8; x *= 2) {
            printf("Block size: %d, Head size divisor: %d\n", block_size, x);
            printf("─────────────────────────────────────────────────────\n");
            printf("Reshape FP32 -> FP8 KV cache\n");
            reshape_and_cache<float, uint8_t, Fp8KVCacheDataType::kFp8E4M3>(
              q, num_tokens, num_heads, head_size, block_size, x, repeat);

            printf("Reshape BF16 -> FP8 KV cache\n");
            reshape_and_cache<sycl::ext::oneapi::bfloat16, uint8_t,
                              Fp8KVCacheDataType::kFp8E4M3>(
              q, num_tokens, num_heads, head_size, block_size, x, repeat);
            printf("─────────────────────────────────────────────────────\n");
        }
    }

    return 0;
}
