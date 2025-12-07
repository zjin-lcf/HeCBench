# wordcount-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME wordcount
    MODEL omp
    SOURCES wc.cpp
    CATEGORIES algorithms
)
