# openmp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME openmp
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
