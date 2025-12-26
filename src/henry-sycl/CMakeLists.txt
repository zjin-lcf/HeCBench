# henry-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
