# dwconv-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dwconv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
