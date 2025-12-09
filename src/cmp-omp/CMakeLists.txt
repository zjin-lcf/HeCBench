# cmp-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cmp
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
