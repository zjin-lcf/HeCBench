# henry-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
