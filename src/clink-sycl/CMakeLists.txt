# clink-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME clink
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
