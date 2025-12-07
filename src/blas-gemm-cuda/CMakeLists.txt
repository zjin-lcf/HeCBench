# blas-gemm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
