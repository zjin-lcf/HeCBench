# simpleSpmv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleSpmv
    MODEL sycl
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
