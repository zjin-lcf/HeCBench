# gels-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gels
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
