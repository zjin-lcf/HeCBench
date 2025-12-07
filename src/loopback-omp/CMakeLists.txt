# loopback-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
