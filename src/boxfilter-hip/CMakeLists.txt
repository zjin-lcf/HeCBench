# boxfilter-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME boxfilter
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
