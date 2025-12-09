# sobel-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sobel
    MODEL omp
    SOURCES kernels.cpp main.cpp reference.cpp
    CATEGORIES algorithms
)
