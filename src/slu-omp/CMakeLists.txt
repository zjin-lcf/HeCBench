# slu-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME slu
    MODEL omp
    SOURCES numeric.cpp
    CATEGORIES algorithms
)
