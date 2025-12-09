# stddev-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME stddev
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
