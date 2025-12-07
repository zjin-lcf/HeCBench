# adamw-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME adamw
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
