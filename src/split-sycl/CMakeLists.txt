# split-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME split
    MODEL sycl
    SOURCES main.cpp verify.cpp
    CATEGORIES algorithms
)
