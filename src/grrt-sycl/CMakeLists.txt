# grrt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
