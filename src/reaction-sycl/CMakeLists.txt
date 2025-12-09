# reaction-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME reaction
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
