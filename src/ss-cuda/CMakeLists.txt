# ss-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ss
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
