# mriQ-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mriQ
    MODEL cuda
    SOURCES computeQ.cu main.cu
    CATEGORIES algorithms
)
