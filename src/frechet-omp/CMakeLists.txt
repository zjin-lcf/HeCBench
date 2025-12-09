# frechet-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME frechet
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
