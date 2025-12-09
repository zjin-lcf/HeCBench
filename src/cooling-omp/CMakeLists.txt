# cooling-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cooling
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
