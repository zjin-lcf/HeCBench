# adv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL cuda
    SOURCES adv.cu main.cu
    CATEGORIES algorithms
)
