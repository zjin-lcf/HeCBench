# cm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cm
    MODEL omp
    SOURCES process.cpp
    CATEGORIES algorithms
)
