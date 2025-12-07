# particles-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL cuda
    SOURCES bitonicSort.cu bitonicSort_kernels.cu main.cu particles.cu particles_kernels.cu
    CATEGORIES algorithms
)
