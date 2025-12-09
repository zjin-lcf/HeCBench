# gd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gd
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
