# determinant-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME determinant
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
