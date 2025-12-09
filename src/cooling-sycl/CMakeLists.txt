# cooling-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cooling
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
