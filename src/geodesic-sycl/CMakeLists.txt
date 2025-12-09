# geodesic-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME geodesic
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
