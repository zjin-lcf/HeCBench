# pso-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pso
    MODEL hip
    SOURCES kernel_gpu.cu
    CATEGORIES algorithms
)
