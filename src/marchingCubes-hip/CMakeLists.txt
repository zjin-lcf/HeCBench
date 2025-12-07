# marchingCubes-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME marchingCubes
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
