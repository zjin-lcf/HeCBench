# sampling-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
