# blas-gemm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
