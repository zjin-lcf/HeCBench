# fwt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL omp
    SOURCES kernels.cpp main.cpp main2.cpp reference.cpp
    CATEGORIES algorithms
)
