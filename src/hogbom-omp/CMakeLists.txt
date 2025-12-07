# hogbom-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hogbom
    MODEL omp
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
