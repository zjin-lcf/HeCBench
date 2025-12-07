# loopback-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
