# rushlarsen-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rushlarsen
    MODEL omp
    SOURCES kernels.cpp main.cpp reference.cpp utils.cpp
    CATEGORIES algorithms
)
