# ccl-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccl
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
