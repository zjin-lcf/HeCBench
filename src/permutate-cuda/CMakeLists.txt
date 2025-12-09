# permutate-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME permutate
    MODEL cuda
    SOURCES gpu_permutation_testing.cu main.cpp permutation_testing.cpp statistical_test.cpp utils.cpp
    CATEGORIES algorithms
)
