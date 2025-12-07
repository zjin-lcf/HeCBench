# wordcount-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME wordcount
    MODEL hip
    SOURCES wc.cu
    CATEGORIES algorithms
)
