# sptrsv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sptrsv
    MODEL hip
    SOURCES sptrsv_syncfree.cu
    CATEGORIES algorithms
)
