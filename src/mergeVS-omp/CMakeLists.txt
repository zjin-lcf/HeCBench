# mergeVS-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mergeVS
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
