# mcmd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mcmd
    MODEL sycl
    SOURCES force_kernel.cpp
    CATEGORIES algorithms
)
