# columnarSolver-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL hip
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
