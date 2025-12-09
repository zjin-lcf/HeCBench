# permutate-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME permutate
    MODEL hip
    SOURCES gpu_permutation_testing.cu
    CATEGORIES algorithms
)
