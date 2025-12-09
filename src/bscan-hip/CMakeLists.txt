# bscan-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bscan
    MODEL hip
    SOURCES main-wave64.cu main.cu
    CATEGORIES algorithms
)
