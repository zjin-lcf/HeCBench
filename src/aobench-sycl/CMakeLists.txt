# aobench-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME aobench
    MODEL sycl
    SOURCES ao.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
