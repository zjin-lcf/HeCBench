# degrid-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME degrid
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
