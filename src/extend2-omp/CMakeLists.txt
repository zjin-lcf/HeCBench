# extend2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME extend2
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
