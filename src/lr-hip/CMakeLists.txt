# lr-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lr
    MODEL hip
    SOURCES linear_par.cu
    CATEGORIES algorithms
)
