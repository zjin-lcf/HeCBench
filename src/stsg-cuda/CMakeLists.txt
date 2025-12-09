# stsg-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME stsg
    MODEL cuda
    SOURCES Filter.cu main.cu
    CATEGORIES algorithms
)
