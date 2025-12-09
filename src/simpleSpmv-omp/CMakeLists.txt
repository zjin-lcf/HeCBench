# simpleSpmv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleSpmv
    MODEL omp
    SOURCES kernels.cpp
    CATEGORIES algorithms
)
