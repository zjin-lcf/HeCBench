# blas-gemmEx2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmEx2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
