# aes-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME aes
    MODEL hip
    SOURCES kernels.cu main.cu reference.cu utils.cu
    CATEGORIES cryptography
)
