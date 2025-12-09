# attention-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME attention
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
