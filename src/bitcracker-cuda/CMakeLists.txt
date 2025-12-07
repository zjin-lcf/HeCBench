# bitcracker-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitcracker
    MODEL cuda
    SOURCES attack.cu main.cu utils.cu w_blocks.cu
    CATEGORIES algorithms
)
