# lavaMD-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lavaMD
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
