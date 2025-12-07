# floydwarshall2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall2
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography graph
)
