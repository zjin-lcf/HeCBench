#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <omp.h>

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX * 2.f - 1.f; // range -1..1
    }
    return arr;
}

// ----------------------------------------------------------------------------
// checking utils

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    #pragma omp target update from (device_result[0:num_elements]) 
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
        if (fabs(cpu_reference[i] - (T)device_result[i]) > t_eff) {
            printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)device_result[i]);
            nfaults ++;
            if (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
}
