# mace-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mace
    MODEL omp
    SOURCES main.cpp
    CATEGORIES machine-learning
)
