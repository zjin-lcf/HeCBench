# ldpc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ldpc
    MODEL cuda
    SOURCES cpu.cu main.cu
    CATEGORIES algorithms
)
