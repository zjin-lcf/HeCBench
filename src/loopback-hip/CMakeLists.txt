# loopback-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
