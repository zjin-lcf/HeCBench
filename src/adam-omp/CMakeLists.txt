# adam-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME adam
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
