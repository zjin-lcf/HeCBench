# hbc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hbc
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
