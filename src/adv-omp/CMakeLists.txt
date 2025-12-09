# adv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME adv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
