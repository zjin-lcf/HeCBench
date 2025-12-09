# xsbench-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME xsbench
    MODEL hip
    SOURCES Simulation.cu
    CATEGORIES algorithms
)
