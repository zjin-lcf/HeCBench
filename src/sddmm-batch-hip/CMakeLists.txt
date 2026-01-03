# sddmm-batch-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sddmm-batch
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    LINK_LIBRARIES hipsparse
)
