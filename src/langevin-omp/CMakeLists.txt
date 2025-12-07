# langevin-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
