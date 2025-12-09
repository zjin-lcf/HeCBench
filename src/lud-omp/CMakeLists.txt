# lud-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lud
    MODEL omp
    SOURCES lud.cpp
    CATEGORIES algorithms
)
