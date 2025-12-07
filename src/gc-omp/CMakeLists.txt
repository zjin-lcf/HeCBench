# gc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gc
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
