# blas-gemmBatched-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmBatched
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
