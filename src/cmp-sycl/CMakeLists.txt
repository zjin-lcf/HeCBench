# cmp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cmp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
