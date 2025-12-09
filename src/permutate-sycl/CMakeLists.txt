# permutate-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME permutate
    MODEL sycl
    SOURCES gpu_permutation_testing.cpp
    CATEGORIES algorithms
)
