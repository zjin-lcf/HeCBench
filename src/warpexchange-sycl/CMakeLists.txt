# warpexchange-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME warpexchange
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
