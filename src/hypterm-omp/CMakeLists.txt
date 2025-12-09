# hypterm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hypterm
    MODEL omp
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
