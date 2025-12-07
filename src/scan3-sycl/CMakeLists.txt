# scan3-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan3
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
