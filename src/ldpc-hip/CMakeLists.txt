# ldpc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ldpc
    MODEL hip
    SOURCES cpu.cu main.cu
    CATEGORIES algorithms
)
