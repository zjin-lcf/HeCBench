# grrt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
