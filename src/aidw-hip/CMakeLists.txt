# aidw-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME aidw
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
