# xsbench-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME xsbench
    MODEL sycl
    SOURCES Simulation.cpp
    CATEGORIES algorithms
)
