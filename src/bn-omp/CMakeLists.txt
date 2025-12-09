# bn-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL omp
    SOURCES kernels.cpp main.cpp
    CATEGORIES algorithms
)
