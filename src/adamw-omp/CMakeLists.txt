# adamw-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME adamw
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
