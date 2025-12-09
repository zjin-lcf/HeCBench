# stddev-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME stddev
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
