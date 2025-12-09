# bscan-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bscan
    MODEL sycl
    SOURCES main-wave64.cpp main.cpp
    CATEGORIES algorithms
)
