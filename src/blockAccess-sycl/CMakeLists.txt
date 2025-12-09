# blockAccess-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockAccess
    MODEL sycl
    SOURCES main.cpp main2.cpp
    CATEGORIES algorithms
)
