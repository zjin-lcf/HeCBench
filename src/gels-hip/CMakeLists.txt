# gels-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gels
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
