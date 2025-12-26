# ntt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ntt
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -Wno-register
)
