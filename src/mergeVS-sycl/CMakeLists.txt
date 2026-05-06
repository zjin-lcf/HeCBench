# mergeVS-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mergeVS
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
