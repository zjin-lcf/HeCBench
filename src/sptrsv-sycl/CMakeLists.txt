# sptrsv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sptrsv
    MODEL sycl
    SOURCES main.cpp sptrsv_syncfree.cpp
    CATEGORIES algorithms
)
