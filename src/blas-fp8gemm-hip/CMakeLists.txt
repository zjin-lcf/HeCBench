# blas-fp8gemm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-fp8gemm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
