# tridiagonal-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tridiagonal
    MODEL omp
    SOURCES cmd_arg_reader.cpp cyclic_kernels.cpp main.cpp pcr_kernels.cpp shrUtils.cpp sweep_kernels.cpp
    CATEGORIES algorithms
)
