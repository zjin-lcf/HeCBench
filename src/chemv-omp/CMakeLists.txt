# chemv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME chemv
    MODEL omp
    SOURCES kernel.cpp main.cpp
    CATEGORIES algorithms
)
