# testSNAP-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME testSNAP
    MODEL omp
    SOURCES main.cpp utils.cpp
    CATEGORIES algorithms
)
