# openmp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME openmp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
