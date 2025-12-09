# pso-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pso
    MODEL omp
    SOURCES kernel_gpu.cpp
    CATEGORIES algorithms
)
