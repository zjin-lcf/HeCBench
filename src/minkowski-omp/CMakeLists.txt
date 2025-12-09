# minkowski-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minkowski
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
