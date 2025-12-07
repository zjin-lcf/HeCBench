# attention-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME attention
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
