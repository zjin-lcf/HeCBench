# gabor-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
