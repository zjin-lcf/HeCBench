# mriQ-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mriQ
    MODEL hip
    SOURCES computeQ.cu main.cu
    CATEGORIES algorithms
)
