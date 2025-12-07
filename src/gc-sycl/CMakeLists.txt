# gc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
