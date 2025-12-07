# axhelm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME axhelm
    MODEL sycl
    SOURCES main.cpp meshBasis.cpp
    CATEGORIES algorithms
)
