# heartwall-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME heartwall
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
