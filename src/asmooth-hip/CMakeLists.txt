# asmooth-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME asmooth
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
