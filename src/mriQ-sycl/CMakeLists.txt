# mriQ-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mriQ
    MODEL sycl
    SOURCES computeQ.cpp main.cpp
    CATEGORIES algorithms
)
