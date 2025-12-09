# sptrsv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sptrsv
    MODEL omp
    SOURCES sptrsv_syncfree.cpp
    CATEGORIES algorithms
)
