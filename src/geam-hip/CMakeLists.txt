# geam-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME geam
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
