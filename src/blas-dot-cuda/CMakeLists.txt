# blas-dot-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-dot
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
