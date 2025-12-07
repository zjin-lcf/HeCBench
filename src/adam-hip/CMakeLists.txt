# adam-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME adam
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
