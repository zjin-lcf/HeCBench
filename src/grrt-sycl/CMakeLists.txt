# grrt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
