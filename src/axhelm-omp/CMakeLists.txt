# axhelm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME axhelm
    MODEL omp
    SOURCES main.cpp meshBasis.cpp
    CATEGORIES algorithms
)
