# stsg-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME stsg
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
