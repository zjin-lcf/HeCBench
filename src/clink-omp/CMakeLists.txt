# clink-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME clink
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
