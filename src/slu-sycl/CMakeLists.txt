# slu-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME slu
    MODEL sycl
    SOURCES numeric.cpp
    CATEGORIES algorithms
)
