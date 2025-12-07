# ced-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ced
    MODEL sycl
    SOURCES kernel_reference.cpp main.cpp
    CATEGORIES algorithms
)
