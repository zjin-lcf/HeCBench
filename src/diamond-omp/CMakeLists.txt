# diamond-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME diamond
    MODEL omp
    SOURCES masking.cpp
    CATEGORIES algorithms
)
