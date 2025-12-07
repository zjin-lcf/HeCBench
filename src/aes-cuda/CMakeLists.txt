# aes-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME aes
    MODEL cuda
    SOURCES kernels.cu main.cu reference.cu utils.cu
    CATEGORIES cryptography
)
