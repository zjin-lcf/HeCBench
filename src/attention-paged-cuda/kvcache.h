/*
def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    set_random_seed(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches
struct KVCaches {
    void** key_caches;    // array of device pointers
    void** value_caches;
};
*/


__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float uint32_to_uniform(uint32_t x) {
    // Convert to (0,1)
    return (x >> 8) * 0x1.0p-24f;
}

__device__ __forceinline__ float normal_from_uniform( uint32_t& rng) {
    float u1 = uint32_to_uniform(xorshift32(rng));
    float u2 = uint32_to_uniform(xorshift32(rng));

    // Box–Muller
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 6.28318530718f * u2;
    return r * cosf(theta);           // N(0,1)
}

template <typename T>
__global__ void uniform_fill_kernel(
    T* data,
    int64_t n,
    float low,
    float high,
    uint32_t seed
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;

    uint32_t r = xorshift32(rng);
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = (T)v;
}

template <typename T>
__global__ void norm_fill_kernel(
    T* data,
    int n,
    uint32_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;
    data[idx] = (T)normal_from_uniform(rng);
}

template<typename T>
struct KVCaches {
    T** key_caches;
    T** value_caches;
};

template <typename T>
KVCaches<T> create_kv_caches_with_random(
    int num_blocks,
    int block_size,
    int num_layers,
    int num_heads,
    int head_size,
    unsigned long seed,
    int &kv_block_stride,
    int &kv_head_stride
) {
    float scale = std::pow((float)head_size, -0.5f);

    int element_size = sizeof(T);
    int x = 16 / element_size;

    //printf("Key cache shape:\n");
    //printf("[%d %d %d %d %d]\n", num_blocks, num_heads, head_size/x, block_size, x);
    int64_t key_elems = (int64_t) num_blocks * num_heads * (head_size / x) * block_size * x;
    kv_block_stride = key_elems / num_blocks;
    kv_head_stride = kv_block_stride / num_heads;

    //printf("Value cache shape:\n");
    //printf("[%d %d %d %d]\n", num_blocks, num_heads, head_size, block_size);
    int64_t value_elems = (int64_t)num_blocks * num_heads * head_size * block_size;

    T** key_caches_h = (T**)malloc(num_layers * sizeof(T*));
    T** value_caches_h = (T**)malloc(num_layers * sizeof(T*));

    int threads = 256;

    for (int l = 0; l < num_layers; ++l) {
        T* key_cache_d;
        T* value_cache_d;
        GPU_CHECK(cudaMalloc(&key_cache_d, key_elems * element_size));
        GPU_CHECK(cudaMalloc(&value_cache_d, value_elems * element_size));

        int64_t key_blocks = (key_elems + threads - 1) / threads;
        int64_t val_blocks = (value_elems + threads - 1) / threads;

        uniform_fill_kernel<T>
            <<<key_blocks, threads>>>(
                (T*)key_cache_d,
                key_elems,
                -scale,
                scale,
                seed + l);

        uniform_fill_kernel<T>
            <<<val_blocks, threads>>>(
                (T*)value_cache_d,
                value_elems,
                -scale,
                scale,
                seed + l);

        key_caches_h[l] = key_cache_d;
        value_caches_h[l] = value_cache_d;
    }

    KVCaches<T> out;
    out.key_caches = key_caches_h;
    out.value_caches = value_caches_h;
    return out;
}
