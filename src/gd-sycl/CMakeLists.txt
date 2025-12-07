# gd-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gd
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
