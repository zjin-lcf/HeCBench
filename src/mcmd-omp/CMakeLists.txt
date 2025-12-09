# mcmd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcmd
    MODEL omp
    SOURCES force_kernel.cpp
    CATEGORIES algorithms
)
