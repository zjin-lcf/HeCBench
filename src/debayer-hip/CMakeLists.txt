# debayer-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME debayer
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
