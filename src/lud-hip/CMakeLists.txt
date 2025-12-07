# lud-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lud
    MODEL hip
    SOURCES lud.cu
    CATEGORIES algorithms
)
