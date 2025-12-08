# myocyte-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME myocyte
    MODEL omp
    SOURCES main.c
    CATEGORIES algorithms
)
