# haversine-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME haversine
    MODEL hip
    SOURCES distance.cu
    CATEGORIES algorithms
)
