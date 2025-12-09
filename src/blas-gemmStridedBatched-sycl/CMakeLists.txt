# blas-gemmStridedBatched-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmStridedBatched
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
