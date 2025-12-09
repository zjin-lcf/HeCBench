# split-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME split
    MODEL hip
    SOURCES main.cu verify.cu
    CATEGORIES algorithms
)
