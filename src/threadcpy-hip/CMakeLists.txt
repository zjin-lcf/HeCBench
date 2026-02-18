# threadcpy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadcpy
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
