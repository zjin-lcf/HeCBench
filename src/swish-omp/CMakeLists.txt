# swish-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME swish
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
