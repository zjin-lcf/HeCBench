# sad-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sad
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
