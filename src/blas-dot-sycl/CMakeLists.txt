# blas-dot-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blas-dot
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
