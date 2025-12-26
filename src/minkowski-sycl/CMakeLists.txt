# minkowski-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minkowski
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
