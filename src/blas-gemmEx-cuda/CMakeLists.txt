# blas-gemmEx-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmEx
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
    LINK_LIBRARIES hipblasLt
)
