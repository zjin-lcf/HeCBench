# ldpc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ldpc
    MODEL hip
    SOURCES cpu.cu kernel.cu main.cu
    CATEGORIES algorithms
)
