# qkv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME qkv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
