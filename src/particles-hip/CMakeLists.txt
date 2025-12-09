# particles-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL hip
    SOURCES bitonicSort.cu bitonicSort_kernels.cu main.cu particles.cu particles_kernels.cu
    CATEGORIES algorithms
)
