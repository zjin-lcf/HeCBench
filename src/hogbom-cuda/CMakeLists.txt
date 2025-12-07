# hogbom-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hogbom
    MODEL cuda
    SOURCES kernels.cu main.cpp reference.cpp timer.cpp
    CATEGORIES algorithms
)
