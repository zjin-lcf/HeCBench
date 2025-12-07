# xsbench-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME xsbench
    MODEL omp
    SOURCES Simulation.cpp
    CATEGORIES algorithms
)
