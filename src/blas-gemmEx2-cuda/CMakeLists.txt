# blas-gemmEx2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmEx2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
