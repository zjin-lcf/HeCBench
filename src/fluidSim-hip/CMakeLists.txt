# fluidSim-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fluidSim
    MODEL hip
    SOURCES kernels.cu
    CATEGORIES algorithms
)
