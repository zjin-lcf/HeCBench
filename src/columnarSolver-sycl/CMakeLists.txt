# columnarSolver-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME columnarSolver
    MODEL sycl
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
