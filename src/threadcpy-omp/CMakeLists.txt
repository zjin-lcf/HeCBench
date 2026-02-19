# threadcpy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadcpy
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bandwidth
)
