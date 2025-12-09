# cm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cm
    MODEL hip
    SOURCES process.cu
    CATEGORIES algorithms
)
