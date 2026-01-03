# bitcracker-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitcracker
    MODEL hip
    SOURCES attack.cu main.cu utils.cu w_blocks.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES algorithms
)
