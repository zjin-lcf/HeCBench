# f8cast-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME f8cast
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
