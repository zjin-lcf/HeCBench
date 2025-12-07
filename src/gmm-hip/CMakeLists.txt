# gmm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL hip
    SOURCES cluster.cu gaussian_kernel.cu main.cu readData.cu
    CATEGORIES algorithms
)
