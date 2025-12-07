# lebesgue-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lebesgue
    MODEL omp
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
