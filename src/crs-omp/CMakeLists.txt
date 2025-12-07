# crs-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME crs
    MODEL omp
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES algorithms
)
