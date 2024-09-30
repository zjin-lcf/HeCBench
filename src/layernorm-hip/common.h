#include <cfloat>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// warp-level reduction for summing values
__device__ inline float warpReduceSum(cooperative_groups::thread_block_tile<32> &warp, float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_xor(val, offset);
    }
    return val;
}

// provide cache hints where possible
#define __stcs(ptr, val) patched_stcs(ptr, val)
#define __ldcs(ptr) patched_ldcs(ptr)
static __device__ __forceinline__ void patched_stcs(float *addr, float val) {
    __builtin_nontemporal_store(val, addr);
}

static __device__ __forceinline__ float patched_ldcs(const float *addr) {
    return __builtin_nontemporal_load(addr);
}

// ----------------------------------------------------------------------------
// checking utils

// HIP error checking
void hip_check(hipError_t error, const char *file, int line) {
    if (error != hipSuccess) {
        printf("[HIP ERROR] at file %s:%d:\n%s\n", file, line,
               hipGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define hipCheck(err) (hip_check(err, __FILE__, __LINE__))

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    hipCheck(hipMemcpy(out_gpu, device_result, num_elements * sizeof(D), hipMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (std::size_t i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!std::isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        //if (i < 5) { printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]); }
        
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        kernel(std::forward<KernelArgs>(kernel_args)...);

        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> duration = stop - start;
        elapsed_time += duration.count();
    }

    return elapsed_time / repeats;
}
