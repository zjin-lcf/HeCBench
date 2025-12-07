# minisweep-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minisweep
    MODEL cuda
    SOURCES kernels.cu main.cu utils.cu
    CATEGORIES algorithms
)
