# columnarSolver-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL cuda
    SOURCES kernels.cu main.cu
    CATEGORIES algorithms
)
