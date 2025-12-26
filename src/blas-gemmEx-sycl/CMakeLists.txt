# blas-gemmEx-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-gemmEx
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
