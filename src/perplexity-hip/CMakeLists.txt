# perplexity-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME perplexity
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
