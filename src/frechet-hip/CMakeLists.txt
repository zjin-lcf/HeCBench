# frechet-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME frechet
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
