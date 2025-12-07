# henry-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
