# b+tree-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME b+tree
    MODEL omp
    SOURCES main.c
    CATEGORIES algorithms
)
