# haversine-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME haversine
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffast-math
)
