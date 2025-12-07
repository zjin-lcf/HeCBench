# blas-gemm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
