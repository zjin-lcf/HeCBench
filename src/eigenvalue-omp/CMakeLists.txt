# eigenvalue-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME eigenvalue
    MODEL omp
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES algorithms
)
