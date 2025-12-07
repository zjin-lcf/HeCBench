# svd3x3-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
