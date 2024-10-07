#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

template<class D, class T>
void validate_result(D* out_gpu, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance = 1e-4) {

    // Copy results from device to host
    #pragma omp target update from (out_gpu[0:num_elements]) 
    
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079f;
#endif

    for (std::size_t i = 0; i < num_elements; ++i) {
        // Skip masked elements
        if (!std::isfinite(cpu_reference[i])) {
            continue;
        }

        // Print the first few comparisons
        //if (i < 5) { std::cout << cpu_reference[i] << " " << static_cast<T>(out_gpu[i]) << std::endl; }

        // Effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + std::fabs(cpu_reference[i]) * epsilon;

        // Ensure correctness for all elements
        if (std::fabs(cpu_reference[i] - static_cast<T>(out_gpu[i])) > t_eff) {
            std::cerr << "Mismatch of " << name << " at " << i << ": CPU_ref: " << cpu_reference[i] << " vs GPU: " << static_cast<T>(out_gpu[i]) << std::endl;
            nfaults++;
            if (nfaults >= 10) {
                std::exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        std::exit(EXIT_FAILURE);
    }
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // Start recording the timing of the kernel
        auto start = std::chrono::high_resolution_clock::now();

        kernel(std::forward<KernelArgs>(kernel_args)...);

        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> duration = stop - start;
        elapsed_time += duration.count();
    }

    return elapsed_time / repeats;
}

//
template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}
