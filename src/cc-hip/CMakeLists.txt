# cc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cc
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
