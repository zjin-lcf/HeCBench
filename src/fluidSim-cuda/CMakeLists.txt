# fluidSim-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fluidSim
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -DVERIFY
)
