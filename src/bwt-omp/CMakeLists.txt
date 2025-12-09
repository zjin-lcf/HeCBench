# bwt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bwt
    MODEL omp
    SOURCES bwt.cpp
    CATEGORIES algorithms
)
