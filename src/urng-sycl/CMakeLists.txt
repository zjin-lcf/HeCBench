# urng-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME urng
    MODEL sycl
    SOURCES kernel.cpp main.cpp
    CATEGORIES algorithms
)
