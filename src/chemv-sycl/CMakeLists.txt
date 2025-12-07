# chemv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME chemv
    MODEL sycl
    SOURCES kernel.cpp main.cpp
    CATEGORIES algorithms
)
