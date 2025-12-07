# opticalFlow-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME opticalFlow
    MODEL cuda
    SOURCES flowCUDA.cu flowGold.cu main.cu
    CATEGORIES algorithms
)
