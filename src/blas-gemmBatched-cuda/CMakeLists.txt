# blas-gemmBatched-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmBatched
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
