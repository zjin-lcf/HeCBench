# p4-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME p4
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
