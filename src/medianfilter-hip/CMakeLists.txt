# medianfilter-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME medianfilter
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
