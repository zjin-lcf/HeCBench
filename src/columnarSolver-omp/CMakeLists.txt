# columnarSolver-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
