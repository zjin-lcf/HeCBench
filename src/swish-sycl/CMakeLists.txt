# swish-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME swish
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
