# minkowski-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME minkowski
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
