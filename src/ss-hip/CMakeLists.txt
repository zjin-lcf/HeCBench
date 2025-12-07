# ss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ss
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
