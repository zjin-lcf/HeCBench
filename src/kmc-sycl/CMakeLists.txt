# kmc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
