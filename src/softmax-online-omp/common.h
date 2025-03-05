#include <cfloat>
#include <chrono>
#include <cmath>
#include <utility>

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// ----------------------------------------------------------------------------
// checking utils

template<class D, class T>
void validate_result(D* out_gpu, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    #pragma omp target update from (out_gpu[0:num_elements])
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
