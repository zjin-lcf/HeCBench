# bn-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
