# permutate-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME permutate
    MODEL omp
    SOURCES gpu_permutation_testing.cpp
    CATEGORIES algorithms
)
