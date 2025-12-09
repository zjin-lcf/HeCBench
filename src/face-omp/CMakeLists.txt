# face-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME face
    MODEL omp
    SOURCES haar.cpp
    CATEGORIES algorithms
)
