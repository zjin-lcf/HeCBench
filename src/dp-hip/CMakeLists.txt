# dp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp
    MODEL hip
    SOURCES main.cu shrUtils.cu
    CATEGORIES algorithms
)
