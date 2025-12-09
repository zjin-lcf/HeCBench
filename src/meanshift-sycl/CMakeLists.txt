# meanshift-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME meanshift
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
