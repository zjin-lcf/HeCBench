# testSNAP-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME testSNAP
    MODEL hip
    SOURCES main.cu utils.cu
    CATEGORIES algorithms
)
