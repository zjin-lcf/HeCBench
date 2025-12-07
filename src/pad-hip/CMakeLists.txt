# pad-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pad
    MODEL hip
    SOURCES kernel.cu main.cu
    CATEGORIES algorithms
)
