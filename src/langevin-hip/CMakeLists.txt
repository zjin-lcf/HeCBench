# langevin-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
