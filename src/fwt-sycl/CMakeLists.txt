# fwt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL sycl
    SOURCES kernels.cpp main.cpp reference.cpp
    CATEGORIES algorithms
)
