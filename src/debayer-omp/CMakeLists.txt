# debayer-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME debayer
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
