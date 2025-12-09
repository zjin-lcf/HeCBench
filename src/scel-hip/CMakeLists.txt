# scel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scel
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
