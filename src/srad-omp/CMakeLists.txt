# srad-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME srad
    MODEL omp
    SOURCES main.c
    CATEGORIES algorithms
)
