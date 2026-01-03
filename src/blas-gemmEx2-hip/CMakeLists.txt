# blas-gemmEx2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmEx2
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblaslt
)
