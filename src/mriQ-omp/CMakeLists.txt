# mriQ-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mriQ
    MODEL omp
    SOURCES computeQ.cpp main.cpp
    CATEGORIES algorithms
)
