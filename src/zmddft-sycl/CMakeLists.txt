# zmddft-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME zmddft
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
