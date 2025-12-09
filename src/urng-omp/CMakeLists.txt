# urng-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME urng
    MODEL omp
    SOURCES kernel.cpp main.cpp
    CATEGORIES algorithms
)
