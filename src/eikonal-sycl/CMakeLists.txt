# eikonal-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME eikonal
    MODEL sycl
    SOURCES StructuredEikonal.cpp fim.cpp kernel.cpp main.cpp
    CATEGORIES algorithms
)
