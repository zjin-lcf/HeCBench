# gmm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL cuda
    SOURCES cluster.cu gaussian_kernel.cu main.cu readData.cu
    CATEGORIES algorithms
)
