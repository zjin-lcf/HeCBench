# loopback-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
