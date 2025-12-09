# spmv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spmv
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
