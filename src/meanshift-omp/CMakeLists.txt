# meanshift-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME meanshift
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
