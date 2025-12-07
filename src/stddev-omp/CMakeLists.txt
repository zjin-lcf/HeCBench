# stddev-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME stddev
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
