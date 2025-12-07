# hogbom-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hogbom
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
