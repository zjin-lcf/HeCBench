# dxtc2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dxtc2
    MODEL sycl
    SOURCES block.cpp kernel.cpp main.cpp
    CATEGORIES algorithms
)
