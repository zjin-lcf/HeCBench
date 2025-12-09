# f16sp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16sp
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
