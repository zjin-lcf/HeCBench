# blas-gemmStridedBatched-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmStridedBatched
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
