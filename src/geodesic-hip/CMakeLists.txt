# geodesic-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME geodesic
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
