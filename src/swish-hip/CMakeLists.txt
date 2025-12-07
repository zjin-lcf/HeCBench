# swish-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME swish
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
