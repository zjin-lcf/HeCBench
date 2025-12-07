# gabor-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
