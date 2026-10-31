# rmsnorm-fused-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm-fused
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
