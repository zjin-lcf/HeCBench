# sad-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sad
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
